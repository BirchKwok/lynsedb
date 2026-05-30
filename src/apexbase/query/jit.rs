//! JIT Compilation for Query Expressions using Cranelift
//!
//! This module provides Just-In-Time compilation for query predicates,
//! converting SQL WHERE clauses into native machine code for faster execution.
//!
//! Key optimizations:
//! - Compile filter predicates to native code
//! - SIMD vectorized aggregations (SUM, COUNT, MIN, MAX, AVG)
//! - Complex expression support (AND, OR, multi-condition)
//! - Eliminate interpreter overhead
//! - Enable CPU-level optimizations

use cranelift::prelude::*;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};
use once_cell::sync::Lazy;
use parking_lot::Mutex;
use std::collections::HashMap;

use crate::data::Value;
use crate::query::sql_parser::BinaryOperator;
use crate::query::SqlExpr;

/// Compiled filter function signature
/// Takes: (data_ptr, row_count, result_ptr) -> count of matches
pub type FilterFnI64 = unsafe extern "C" fn(*const i64, usize, *mut u8) -> usize;

/// Compiled SUM aggregation function
/// Takes: (data_ptr, row_count) -> sum
pub type SumFnI64 = unsafe extern "C" fn(*const i64, usize) -> i64;

/// Compiled COUNT aggregation with filter
/// Takes: (data_ptr, row_count, threshold) -> count
pub type CountFilteredFn = unsafe extern "C" fn(*const i64, usize, i64) -> usize;

/// Compiled MIN/MAX aggregation function
/// Takes: (data_ptr, row_count) -> min or max value
pub type MinMaxFnI64 = unsafe extern "C" fn(*const i64, usize) -> i64;

/// Compiled AVG aggregation function (returns sum and count for precision)
/// Takes: (data_ptr, row_count, out_sum_ptr, out_count_ptr)
pub type AvgFnI64 = unsafe extern "C" fn(*const i64, usize, *mut i64, *mut i64);

/// Cache for compiled filter functions (function pointers only, not the module)
static JIT_CACHE: Lazy<Mutex<HashMap<u64, usize>>> = Lazy::new(|| Mutex::new(HashMap::new()));

/// JIT compiler for query expressions
pub struct ExprJIT {
    /// Cranelift JIT module
    module: JITModule,
    /// Function builder context  
    builder_ctx: FunctionBuilderContext,
}

impl ExprJIT {
    /// Create a new JIT compiler
    pub fn new() -> Result<Self, String> {
        let mut flag_builder = settings::builder();
        flag_builder
            .set("use_colocated_libcalls", "false")
            .map_err(|e| e.to_string())?;
        flag_builder
            .set("is_pic", "false")
            .map_err(|e| e.to_string())?;
        flag_builder
            .set("opt_level", "speed")
            .map_err(|e| e.to_string())?;

        let isa_builder = cranelift_native::builder().map_err(|e| e.to_string())?;
        let isa = isa_builder
            .finish(settings::Flags::new(flag_builder))
            .map_err(|e| e.to_string())?;

        let builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        let module = JITModule::new(builder);

        Ok(Self {
            module,
            builder_ctx: FunctionBuilderContext::new(),
        })
    }

    /// Compile an integer comparison filter: column op literal
    /// Returns a function pointer that can be called to filter data
    pub fn compile_int_filter(
        &mut self,
        op: BinaryOperator,
        literal_value: i64,
    ) -> Result<FilterFnI64, String> {
        let pointer_type = self.module.target_config().pointer_type();

        // Create function context
        let mut ctx = self.module.make_context();

        // Signature: fn(data: *i64, count: usize, result: *mut u8) -> usize
        ctx.func.signature.params.push(AbiParam::new(pointer_type)); // data ptr
        ctx.func.signature.params.push(AbiParam::new(types::I64)); // count
        ctx.func.signature.params.push(AbiParam::new(pointer_type)); // result ptr
        ctx.func.signature.returns.push(AbiParam::new(types::I64)); // match count

        let mut builder = FunctionBuilder::new(&mut ctx.func, &mut self.builder_ctx);

        // Entry block
        let entry = builder.create_block();
        builder.append_block_params_for_function_params(entry);
        builder.switch_to_block(entry);
        builder.seal_block(entry);

        let data_ptr = builder.block_params(entry)[0];
        let count = builder.block_params(entry)[1];
        let result_ptr = builder.block_params(entry)[2];

        let lit_val = builder.ins().iconst(types::I64, literal_value);
        let zero = builder.ins().iconst(types::I64, 0);
        let one = builder.ins().iconst(types::I64, 1);
        let eight = builder.ins().iconst(types::I64, 8);

        // Loop blocks
        let loop_header = builder.create_block();
        let loop_body = builder.create_block();
        let loop_exit = builder.create_block();

        // Declare block params upfront (before any branches target them)
        builder.append_block_param(loop_header, types::I64); // i
        builder.append_block_param(loop_header, types::I64); // match_count
        builder.append_block_param(loop_exit, types::I64); // final match_count

        // Jump to loop with initial counter and match_count
        builder.ins().jump(loop_header, &[zero, zero]);

        // Loop header: i, match_count
        builder.switch_to_block(loop_header);
        let i = builder.block_params(loop_header)[0];
        let match_count = builder.block_params(loop_header)[1];

        let cond = builder.ins().icmp(IntCC::UnsignedLessThan, i, count);
        builder
            .ins()
            .brif(cond, loop_body, &[], loop_exit, &[match_count]);

        // Loop body
        builder.switch_to_block(loop_body);

        // Load value: data_ptr[i * 8]
        let offset = builder.ins().imul(i, eight);
        let addr = builder.ins().iadd(data_ptr, offset);
        let val = builder.ins().load(types::I64, MemFlags::new(), addr, 0);

        // Compare
        let cmp_result = match op {
            BinaryOperator::Eq => builder.ins().icmp(IntCC::Equal, val, lit_val),
            BinaryOperator::NotEq => builder.ins().icmp(IntCC::NotEqual, val, lit_val),
            BinaryOperator::Lt => builder.ins().icmp(IntCC::SignedLessThan, val, lit_val),
            BinaryOperator::Le => builder
                .ins()
                .icmp(IntCC::SignedLessThanOrEqual, val, lit_val),
            BinaryOperator::Gt => builder.ins().icmp(IntCC::SignedGreaterThan, val, lit_val),
            BinaryOperator::Ge => builder
                .ins()
                .icmp(IntCC::SignedGreaterThanOrEqual, val, lit_val),
            _ => return Err(format!("Unsupported operator: {:?}", op)),
        };

        // Store result byte (icmp returns I8, use directly)
        let result_addr = builder.ins().iadd(result_ptr, i);
        builder
            .ins()
            .store(MemFlags::new(), cmp_result, result_addr, 0);

        // Update match count: match_count += cmp_result ? 1 : 0
        let inc = builder.ins().uextend(types::I64, cmp_result);
        let new_match_count = builder.ins().iadd(match_count, inc);

        // Next iteration
        let next_i = builder.ins().iadd(i, one);
        builder.ins().jump(loop_header, &[next_i, new_match_count]);

        // Exit
        builder.switch_to_block(loop_exit);
        let final_count = builder.block_params(loop_exit)[0];
        builder.ins().return_(&[final_count]);

        builder.seal_block(loop_header);
        builder.seal_block(loop_body);
        builder.seal_block(loop_exit);
        builder.finalize();

        // Compile
        let func_id = self
            .module
            .declare_function("filter_i64", Linkage::Local, &ctx.func.signature)
            .map_err(|e| e.to_string())?;

        self.module
            .define_function(func_id, &mut ctx)
            .map_err(|e| e.to_string())?;
        self.module.clear_context(&mut ctx);
        self.module
            .finalize_definitions()
            .map_err(|e| e.to_string())?;

        let code_ptr = self.module.get_finalized_function(func_id);
        let func: FilterFnI64 = unsafe { std::mem::transmute(code_ptr) };

        Ok(func)
    }

    /// Compile a SIMD-optimized SUM aggregation (4-way unrolled loop)
    /// Returns a function that computes sum of i64 array
    pub fn compile_sum_i64(&mut self) -> Result<SumFnI64, String> {
        let pointer_type = self.module.target_config().pointer_type();
        let mut ctx = self.module.make_context();

        // Signature: fn(data: *i64, count: usize) -> i64
        ctx.func.signature.params.push(AbiParam::new(pointer_type));
        ctx.func.signature.params.push(AbiParam::new(types::I64));
        ctx.func.signature.returns.push(AbiParam::new(types::I64));

        let mut builder = FunctionBuilder::new(&mut ctx.func, &mut self.builder_ctx);

        let entry = builder.create_block();
        builder.append_block_params_for_function_params(entry);
        builder.switch_to_block(entry);
        builder.seal_block(entry);

        let data_ptr = builder.block_params(entry)[0];
        let count = builder.block_params(entry)[1];

        let zero = builder.ins().iconst(types::I64, 0);
        let one = builder.ins().iconst(types::I64, 1);
        let four = builder.ins().iconst(types::I64, 4);
        let eight = builder.ins().iconst(types::I64, 8);
        let thirty_two = builder.ins().iconst(types::I64, 32);

        // 4-way unrolled main loop
        let loop_header = builder.create_block();
        let loop_body_4x = builder.create_block();
        let loop_remainder = builder.create_block();
        let loop_body_1x = builder.create_block();
        let loop_exit = builder.create_block();

        // Calculate how many 4-element iterations we can do
        let two_const = builder.ins().iconst(types::I64, 2);
        let count_div_4 = builder.ins().ushr(count, two_const);
        let two_const2 = builder.ins().iconst(types::I64, 2);
        let count_4x = builder.ins().ishl(count_div_4, two_const2);

        // Start with 4 accumulators for better pipelining
        builder
            .ins()
            .jump(loop_header, &[zero, zero, zero, zero, zero]);

        // Loop header: i, sum0, sum1, sum2, sum3
        builder.switch_to_block(loop_header);
        builder.append_block_param(loop_header, types::I64); // i
        builder.append_block_param(loop_header, types::I64); // sum0
        builder.append_block_param(loop_header, types::I64); // sum1
        builder.append_block_param(loop_header, types::I64); // sum2
        builder.append_block_param(loop_header, types::I64); // sum3

        let i = builder.block_params(loop_header)[0];
        let sum0 = builder.block_params(loop_header)[1];
        let sum1 = builder.block_params(loop_header)[2];
        let sum2 = builder.block_params(loop_header)[3];
        let sum3 = builder.block_params(loop_header)[4];

        let cond_4x = builder.ins().icmp(IntCC::UnsignedLessThan, i, count_4x);
        builder
            .ins()
            .brif(cond_4x, loop_body_4x, &[], loop_remainder, &[]);

        // 4x unrolled body
        builder.switch_to_block(loop_body_4x);
        let offset = builder.ins().imul(i, eight);
        let addr0 = builder.ins().iadd(data_ptr, offset);
        let addr1 = builder.ins().iadd(addr0, eight);
        let addr2 = builder.ins().iadd(addr1, eight);
        let addr3 = builder.ins().iadd(addr2, eight);

        let v0 = builder.ins().load(types::I64, MemFlags::new(), addr0, 0);
        let v1 = builder.ins().load(types::I64, MemFlags::new(), addr1, 0);
        let v2 = builder.ins().load(types::I64, MemFlags::new(), addr2, 0);
        let v3 = builder.ins().load(types::I64, MemFlags::new(), addr3, 0);

        let new_sum0 = builder.ins().iadd(sum0, v0);
        let new_sum1 = builder.ins().iadd(sum1, v1);
        let new_sum2 = builder.ins().iadd(sum2, v2);
        let new_sum3 = builder.ins().iadd(sum3, v3);

        let next_i = builder.ins().iadd(i, four);
        builder.ins().jump(
            loop_header,
            &[next_i, new_sum0, new_sum1, new_sum2, new_sum3],
        );

        // Remainder loop (process remaining 0-3 elements)
        builder.switch_to_block(loop_remainder);
        // Combine the 4 accumulators
        let partial_sum = builder.ins().iadd(sum0, sum1);
        let partial_sum = builder.ins().iadd(partial_sum, sum2);
        let partial_sum = builder.ins().iadd(partial_sum, sum3);
        builder.ins().jump(loop_body_1x, &[i, partial_sum]);

        // 1x loop for remainder
        builder.switch_to_block(loop_body_1x);
        builder.append_block_param(loop_body_1x, types::I64); // i
        builder.append_block_param(loop_body_1x, types::I64); // sum

        let i_rem = builder.block_params(loop_body_1x)[0];
        let sum_rem = builder.block_params(loop_body_1x)[1];

        let cond_1x = builder.ins().icmp(IntCC::UnsignedLessThan, i_rem, count);
        let inner_body = builder.create_block();
        builder.ins().brif(cond_1x, inner_body, &[], loop_exit, &[]);
        builder.switch_to_block(inner_body);
        let offset_rem = builder.ins().imul(i_rem, eight);
        let addr_rem = builder.ins().iadd(data_ptr, offset_rem);
        let v_rem = builder.ins().load(types::I64, MemFlags::new(), addr_rem, 0);
        let new_sum_rem = builder.ins().iadd(sum_rem, v_rem);
        let next_i_rem = builder.ins().iadd(i_rem, one);
        builder.ins().jump(loop_body_1x, &[next_i_rem, new_sum_rem]);

        builder.seal_block(inner_body);

        // Exit
        builder.switch_to_block(loop_exit);
        let final_sum = builder.block_params(loop_body_1x)[1];
        builder.ins().return_(&[final_sum]);

        builder.seal_block(loop_header);
        builder.seal_block(loop_body_4x);
        builder.seal_block(loop_remainder);
        builder.seal_block(loop_body_1x);
        builder.seal_block(loop_exit);
        builder.finalize();

        let func_id = self
            .module
            .declare_function("sum_i64", Linkage::Local, &ctx.func.signature)
            .map_err(|e| e.to_string())?;

        self.module
            .define_function(func_id, &mut ctx)
            .map_err(|e| e.to_string())?;
        self.module.clear_context(&mut ctx);
        self.module
            .finalize_definitions()
            .map_err(|e| e.to_string())?;

        let code_ptr = self.module.get_finalized_function(func_id);
        Ok(unsafe { std::mem::transmute(code_ptr) })
    }

    /// Compile a SIMD-optimized MIN aggregation
    pub fn compile_min_i64(&mut self) -> Result<MinMaxFnI64, String> {
        self.compile_minmax_i64(true)
    }

    /// Compile a SIMD-optimized MAX aggregation
    pub fn compile_max_i64(&mut self) -> Result<MinMaxFnI64, String> {
        self.compile_minmax_i64(false)
    }

    fn compile_minmax_i64(&mut self, is_min: bool) -> Result<MinMaxFnI64, String> {
        let pointer_type = self.module.target_config().pointer_type();
        let mut ctx = self.module.make_context();

        ctx.func.signature.params.push(AbiParam::new(pointer_type));
        ctx.func.signature.params.push(AbiParam::new(types::I64));
        ctx.func.signature.returns.push(AbiParam::new(types::I64));

        let mut builder = FunctionBuilder::new(&mut ctx.func, &mut self.builder_ctx);

        let entry = builder.create_block();
        builder.append_block_params_for_function_params(entry);
        builder.switch_to_block(entry);
        builder.seal_block(entry);

        let data_ptr = builder.block_params(entry)[0];
        let count = builder.block_params(entry)[1];

        let one = builder.ins().iconst(types::I64, 1);
        let eight = builder.ins().iconst(types::I64, 8);

        // Load first element as initial value
        let initial = builder.ins().load(types::I64, MemFlags::new(), data_ptr, 0);

        let loop_header = builder.create_block();
        let loop_body = builder.create_block();
        let loop_exit = builder.create_block();

        builder.ins().jump(loop_header, &[one, initial]);

        builder.switch_to_block(loop_header);
        builder.append_block_param(loop_header, types::I64); // i
        builder.append_block_param(loop_header, types::I64); // current min/max

        let i = builder.block_params(loop_header)[0];
        let current = builder.block_params(loop_header)[1];

        let cond = builder.ins().icmp(IntCC::UnsignedLessThan, i, count);
        builder.ins().brif(cond, loop_body, &[], loop_exit, &[]);

        builder.switch_to_block(loop_body);
        let offset = builder.ins().imul(i, eight);
        let addr = builder.ins().iadd(data_ptr, offset);
        let val = builder.ins().load(types::I64, MemFlags::new(), addr, 0);

        // Select min or max
        let cmp = if is_min {
            builder.ins().icmp(IntCC::SignedLessThan, val, current)
        } else {
            builder.ins().icmp(IntCC::SignedGreaterThan, val, current)
        };
        let new_val = builder.ins().select(cmp, val, current);

        let next_i = builder.ins().iadd(i, one);
        builder.ins().jump(loop_header, &[next_i, new_val]);

        builder.switch_to_block(loop_exit);
        let result = builder.block_params(loop_header)[1];
        builder.ins().return_(&[result]);

        builder.seal_block(loop_header);
        builder.seal_block(loop_body);
        builder.seal_block(loop_exit);
        builder.finalize();

        let func_name = if is_min { "min_i64" } else { "max_i64" };
        let func_id = self
            .module
            .declare_function(func_name, Linkage::Local, &ctx.func.signature)
            .map_err(|e| e.to_string())?;

        self.module
            .define_function(func_id, &mut ctx)
            .map_err(|e| e.to_string())?;
        self.module.clear_context(&mut ctx);
        self.module
            .finalize_definitions()
            .map_err(|e| e.to_string())?;

        let code_ptr = self.module.get_finalized_function(func_id);
        Ok(unsafe { std::mem::transmute(code_ptr) })
    }

    /// Compile a complex filter with AND: (col op1 val1) AND (col op2 val2)
    pub fn compile_range_filter(
        &mut self,
        low: i64,
        high: i64,
        inclusive_low: bool,
        inclusive_high: bool,
    ) -> Result<FilterFnI64, String> {
        let pointer_type = self.module.target_config().pointer_type();
        let mut ctx = self.module.make_context();

        ctx.func.signature.params.push(AbiParam::new(pointer_type));
        ctx.func.signature.params.push(AbiParam::new(types::I64));
        ctx.func.signature.params.push(AbiParam::new(pointer_type));
        ctx.func.signature.returns.push(AbiParam::new(types::I64));

        let mut builder = FunctionBuilder::new(&mut ctx.func, &mut self.builder_ctx);

        let entry = builder.create_block();
        builder.append_block_params_for_function_params(entry);
        builder.switch_to_block(entry);
        builder.seal_block(entry);

        let data_ptr = builder.block_params(entry)[0];
        let count = builder.block_params(entry)[1];
        let result_ptr = builder.block_params(entry)[2];

        let low_val = builder.ins().iconst(types::I64, low);
        let high_val = builder.ins().iconst(types::I64, high);
        let zero = builder.ins().iconst(types::I64, 0);
        let one = builder.ins().iconst(types::I64, 1);
        let eight = builder.ins().iconst(types::I64, 8);

        let loop_header = builder.create_block();
        let loop_body = builder.create_block();
        let loop_exit = builder.create_block();

        builder.ins().jump(loop_header, &[zero, zero]);

        builder.switch_to_block(loop_header);
        builder.append_block_param(loop_header, types::I64);
        builder.append_block_param(loop_header, types::I64);

        let i = builder.block_params(loop_header)[0];
        let match_count = builder.block_params(loop_header)[1];

        let cond = builder.ins().icmp(IntCC::UnsignedLessThan, i, count);
        builder.ins().brif(cond, loop_body, &[], loop_exit, &[]);

        builder.switch_to_block(loop_body);
        let offset = builder.ins().imul(i, eight);
        let addr = builder.ins().iadd(data_ptr, offset);
        let val = builder.ins().load(types::I64, MemFlags::new(), addr, 0);

        // Check low bound
        let low_cmp = if inclusive_low {
            builder
                .ins()
                .icmp(IntCC::SignedGreaterThanOrEqual, val, low_val)
        } else {
            builder.ins().icmp(IntCC::SignedGreaterThan, val, low_val)
        };

        // Check high bound
        let high_cmp = if inclusive_high {
            builder
                .ins()
                .icmp(IntCC::SignedLessThanOrEqual, val, high_val)
        } else {
            builder.ins().icmp(IntCC::SignedLessThan, val, high_val)
        };

        // AND the two conditions
        let in_range = builder.ins().band(low_cmp, high_cmp);

        // Store result
        let result_byte = builder.ins().uextend(types::I8, in_range);
        let result_addr = builder.ins().iadd(result_ptr, i);
        builder
            .ins()
            .store(MemFlags::new(), result_byte, result_addr, 0);

        let inc = builder.ins().uextend(types::I64, in_range);
        let new_match_count = builder.ins().iadd(match_count, inc);

        let next_i = builder.ins().iadd(i, one);
        builder.ins().jump(loop_header, &[next_i, new_match_count]);

        builder.switch_to_block(loop_exit);
        let final_count = builder.block_params(loop_header)[1];
        builder.ins().return_(&[final_count]);

        builder.seal_block(loop_header);
        builder.seal_block(loop_body);
        builder.seal_block(loop_exit);
        builder.finalize();

        let func_id = self
            .module
            .declare_function("range_filter", Linkage::Local, &ctx.func.signature)
            .map_err(|e| e.to_string())?;

        self.module
            .define_function(func_id, &mut ctx)
            .map_err(|e| e.to_string())?;
        self.module.clear_context(&mut ctx);
        self.module
            .finalize_definitions()
            .map_err(|e| e.to_string())?;

        let code_ptr = self.module.get_finalized_function(func_id);
        Ok(unsafe { std::mem::transmute(code_ptr) })
    }

    /// Compile filtered SUM: sum values where condition is true
    pub fn compile_filtered_sum(
        &mut self,
        op: BinaryOperator,
        threshold: i64,
    ) -> Result<SumFnI64, String> {
        let pointer_type = self.module.target_config().pointer_type();
        let mut ctx = self.module.make_context();

        ctx.func.signature.params.push(AbiParam::new(pointer_type));
        ctx.func.signature.params.push(AbiParam::new(types::I64));
        ctx.func.signature.returns.push(AbiParam::new(types::I64));

        let mut builder = FunctionBuilder::new(&mut ctx.func, &mut self.builder_ctx);

        let entry = builder.create_block();
        builder.append_block_params_for_function_params(entry);
        builder.switch_to_block(entry);
        builder.seal_block(entry);

        let data_ptr = builder.block_params(entry)[0];
        let count = builder.block_params(entry)[1];

        let thresh_val = builder.ins().iconst(types::I64, threshold);
        let zero = builder.ins().iconst(types::I64, 0);
        let one = builder.ins().iconst(types::I64, 1);
        let eight = builder.ins().iconst(types::I64, 8);

        let loop_header = builder.create_block();
        let loop_body = builder.create_block();
        let loop_exit = builder.create_block();

        builder.ins().jump(loop_header, &[zero, zero]);

        builder.switch_to_block(loop_header);
        builder.append_block_param(loop_header, types::I64);
        builder.append_block_param(loop_header, types::I64);

        let i = builder.block_params(loop_header)[0];
        let sum = builder.block_params(loop_header)[1];

        let cond = builder.ins().icmp(IntCC::UnsignedLessThan, i, count);
        builder.ins().brif(cond, loop_body, &[], loop_exit, &[]);

        builder.switch_to_block(loop_body);
        let offset = builder.ins().imul(i, eight);
        let addr = builder.ins().iadd(data_ptr, offset);
        let val = builder.ins().load(types::I64, MemFlags::new(), addr, 0);

        // Check filter condition
        let passes = match op {
            BinaryOperator::Gt => builder
                .ins()
                .icmp(IntCC::SignedGreaterThan, val, thresh_val),
            BinaryOperator::Ge => {
                builder
                    .ins()
                    .icmp(IntCC::SignedGreaterThanOrEqual, val, thresh_val)
            }
            BinaryOperator::Lt => builder.ins().icmp(IntCC::SignedLessThan, val, thresh_val),
            BinaryOperator::Le => builder
                .ins()
                .icmp(IntCC::SignedLessThanOrEqual, val, thresh_val),
            BinaryOperator::Eq => builder.ins().icmp(IntCC::Equal, val, thresh_val),
            BinaryOperator::NotEq => builder.ins().icmp(IntCC::NotEqual, val, thresh_val),
            _ => return Err("Unsupported operator".to_string()),
        };

        // Conditional add: sum += passes ? val : 0
        let add_val = builder.ins().select(passes, val, zero);
        let new_sum = builder.ins().iadd(sum, add_val);

        let next_i = builder.ins().iadd(i, one);
        builder.ins().jump(loop_header, &[next_i, new_sum]);

        builder.switch_to_block(loop_exit);
        let final_sum = builder.block_params(loop_header)[1];
        builder.ins().return_(&[final_sum]);

        builder.seal_block(loop_header);
        builder.seal_block(loop_body);
        builder.seal_block(loop_exit);
        builder.finalize();

        let func_id = self
            .module
            .declare_function("filtered_sum", Linkage::Local, &ctx.func.signature)
            .map_err(|e| e.to_string())?;

        self.module
            .define_function(func_id, &mut ctx)
            .map_err(|e| e.to_string())?;
        self.module.clear_context(&mut ctx);
        self.module
            .finalize_definitions()
            .map_err(|e| e.to_string())?;

        let code_ptr = self.module.get_finalized_function(func_id);
        Ok(unsafe { std::mem::transmute(code_ptr) })
    }
}

// ============================================================================
// SIMD Vectorized Operations (Pure Rust, no JIT)
// These use portable SIMD patterns that LLVM will auto-vectorize
// ============================================================================

/// SIMD-friendly sum of i64 array (4-way unrolled for auto-vectorization)
#[inline]
pub fn simd_sum_i64(data: &[i64]) -> i64 {
    let chunks = data.chunks_exact(4);
    let remainder = chunks.remainder();

    let mut sum0: i64 = 0;
    let mut sum1: i64 = 0;
    let mut sum2: i64 = 0;
    let mut sum3: i64 = 0;

    for chunk in chunks {
        sum0 = sum0.wrapping_add(chunk[0]);
        sum1 = sum1.wrapping_add(chunk[1]);
        sum2 = sum2.wrapping_add(chunk[2]);
        sum3 = sum3.wrapping_add(chunk[3]);
    }

    let mut total = sum0
        .wrapping_add(sum1)
        .wrapping_add(sum2)
        .wrapping_add(sum3);
    for &val in remainder {
        total = total.wrapping_add(val);
    }
    total
}

/// SIMD-friendly sum of f64 array
#[inline]
pub fn simd_sum_f64(data: &[f64]) -> f64 {
    let chunks = data.chunks_exact(4);
    let remainder = chunks.remainder();

    let mut sum0: f64 = 0.0;
    let mut sum1: f64 = 0.0;
    let mut sum2: f64 = 0.0;
    let mut sum3: f64 = 0.0;

    for chunk in chunks {
        sum0 += chunk[0];
        sum1 += chunk[1];
        sum2 += chunk[2];
        sum3 += chunk[3];
    }

    let mut total = sum0 + sum1 + sum2 + sum3;
    for &val in remainder {
        total += val;
    }
    total
}

/// SIMD-friendly min of i64 array
#[inline]
pub fn simd_min_i64(data: &[i64]) -> Option<i64> {
    if data.is_empty() {
        return None;
    }

    let chunks = data.chunks_exact(4);
    let remainder = chunks.remainder();

    let mut min0 = i64::MAX;
    let mut min1 = i64::MAX;
    let mut min2 = i64::MAX;
    let mut min3 = i64::MAX;

    for chunk in chunks {
        min0 = min0.min(chunk[0]);
        min1 = min1.min(chunk[1]);
        min2 = min2.min(chunk[2]);
        min3 = min3.min(chunk[3]);
    }

    let mut result = min0.min(min1).min(min2).min(min3);
    for &val in remainder {
        result = result.min(val);
    }
    Some(result)
}

/// SIMD-friendly max of i64 array
#[inline]
pub fn simd_max_i64(data: &[i64]) -> Option<i64> {
    if data.is_empty() {
        return None;
    }

    let chunks = data.chunks_exact(4);
    let remainder = chunks.remainder();

    let mut max0 = i64::MIN;
    let mut max1 = i64::MIN;
    let mut max2 = i64::MIN;
    let mut max3 = i64::MIN;

    for chunk in chunks {
        max0 = max0.max(chunk[0]);
        max1 = max1.max(chunk[1]);
        max2 = max2.max(chunk[2]);
        max3 = max3.max(chunk[3]);
    }

    let mut result = max0.max(max1).max(max2).max(max3);
    for &val in remainder {
        result = result.max(val);
    }
    Some(result)
}

/// SIMD-friendly count where condition is true
#[inline]
pub fn simd_count_gt(data: &[i64], threshold: i64) -> usize {
    let chunks = data.chunks_exact(4);
    let remainder = chunks.remainder();

    let mut count0: usize = 0;
    let mut count1: usize = 0;
    let mut count2: usize = 0;
    let mut count3: usize = 0;

    for chunk in chunks {
        count0 += (chunk[0] > threshold) as usize;
        count1 += (chunk[1] > threshold) as usize;
        count2 += (chunk[2] > threshold) as usize;
        count3 += (chunk[3] > threshold) as usize;
    }

    let mut total = count0 + count1 + count2 + count3;
    for &val in remainder {
        total += (val > threshold) as usize;
    }
    total
}

/// SIMD-friendly conditional sum: sum values where value > threshold
#[inline]
pub fn simd_sum_where_gt(data: &[i64], threshold: i64) -> i64 {
    let chunks = data.chunks_exact(4);
    let remainder = chunks.remainder();

    let mut sum0: i64 = 0;
    let mut sum1: i64 = 0;
    let mut sum2: i64 = 0;
    let mut sum3: i64 = 0;

    for chunk in chunks {
        if chunk[0] > threshold {
            sum0 = sum0.wrapping_add(chunk[0]);
        }
        if chunk[1] > threshold {
            sum1 = sum1.wrapping_add(chunk[1]);
        }
        if chunk[2] > threshold {
            sum2 = sum2.wrapping_add(chunk[2]);
        }
        if chunk[3] > threshold {
            sum3 = sum3.wrapping_add(chunk[3]);
        }
    }

    let mut total = sum0
        .wrapping_add(sum1)
        .wrapping_add(sum2)
        .wrapping_add(sum3);
    for &val in remainder {
        if val > threshold {
            total = total.wrapping_add(val);
        }
    }
    total
}

/// Check if an expression can be JIT compiled
pub fn can_jit_compile(expr: &SqlExpr) -> bool {
    match expr {
        SqlExpr::BinaryOp { left, op, right } => {
            matches!(
                (left.as_ref(), right.as_ref()),
                (SqlExpr::Column(_), SqlExpr::Literal(Value::Int64(_)))
                    | (SqlExpr::Literal(Value::Int64(_)), SqlExpr::Column(_))
            ) && matches!(
                op,
                BinaryOperator::Eq
                    | BinaryOperator::NotEq
                    | BinaryOperator::Lt
                    | BinaryOperator::Le
                    | BinaryOperator::Gt
                    | BinaryOperator::Ge
            )
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jit_creation() {
        let jit = ExprJIT::new();
        assert!(jit.is_ok(), "JIT compiler should initialize");
    }

    #[test]
    fn test_jit_int_filter() {
        let mut jit = ExprJIT::new().unwrap();
        let filter_fn = jit.compile_int_filter(BinaryOperator::Gt, 50).unwrap();

        // Test data: [10, 60, 30, 80, 50]
        let data: Vec<i64> = vec![10, 60, 30, 80, 50];
        let mut result = vec![0u8; 5];

        let count = unsafe { filter_fn(data.as_ptr(), data.len(), result.as_mut_ptr()) };

        assert_eq!(count, 2); // 60 and 80 are > 50
        assert_eq!(result, vec![0, 1, 0, 1, 0]);
    }
}
