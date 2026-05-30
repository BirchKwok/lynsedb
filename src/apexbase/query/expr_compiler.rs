use crate::data::Value;
use crate::query::filter::{CompareOp, Filter};
use crate::query::sql_parser::{BinaryOperator, SqlExpr, UnaryOperator};
use crate::ApexError;

pub fn sql_expr_to_filter(expr: &SqlExpr) -> Result<Filter, ApexError> {
    match expr {
        SqlExpr::BinaryOp { left, op, right } => match op {
            BinaryOperator::And => {
                let left_filter = sql_expr_to_filter(left)?;
                let right_filter = sql_expr_to_filter(right)?;

                // Flatten nested ANDs for better optimization
                let mut filters = Vec::new();
                match left_filter {
                    Filter::And(inner) => filters.extend(inner),
                    other => filters.push(other),
                }
                match right_filter {
                    Filter::And(inner) => filters.extend(inner),
                    other => filters.push(other),
                }
                Ok(Filter::And(filters))
            }
            BinaryOperator::Or => {
                let left_filter = sql_expr_to_filter(left)?;
                let right_filter = sql_expr_to_filter(right)?;
                Ok(Filter::Or(vec![left_filter, right_filter]))
            }
            BinaryOperator::Eq
            | BinaryOperator::NotEq
            | BinaryOperator::Lt
            | BinaryOperator::Le
            | BinaryOperator::Gt
            | BinaryOperator::Ge => {
                let field = match left.as_ref() {
                    SqlExpr::Column(name) => name.clone(),
                    _ => {
                        return Err(ApexError::QueryParseError(
                            "Left side of comparison must be column".to_string(),
                        ))
                    }
                };
                let value = match right.as_ref() {
                    SqlExpr::Literal(v) => v.clone(),
                    _ => {
                        return Err(ApexError::QueryParseError(
                            "Right side of comparison must be literal".to_string(),
                        ))
                    }
                };

                let compare_op = match op {
                    BinaryOperator::Eq => CompareOp::Equal,
                    BinaryOperator::NotEq => CompareOp::NotEqual,
                    BinaryOperator::Lt => CompareOp::LessThan,
                    BinaryOperator::Le => CompareOp::LessEqual,
                    BinaryOperator::Gt => CompareOp::GreaterThan,
                    BinaryOperator::Ge => CompareOp::GreaterEqual,
                    _ => unreachable!(),
                };

                Ok(Filter::Compare {
                    field,
                    op: compare_op,
                    value,
                })
            }
            _ => Err(ApexError::QueryParseError(format!(
                "Unsupported binary operator: {:?}",
                op
            ))),
        },
        SqlExpr::UnaryOp { op, expr } => match op {
            UnaryOperator::Not => {
                let inner = sql_expr_to_filter(expr)?;
                Ok(Filter::Not(Box::new(inner)))
            }
            _ => Err(ApexError::QueryParseError(format!(
                "Unsupported unary operator: {:?}",
                op
            ))),
        },
        SqlExpr::Like {
            column,
            pattern,
            negated,
        } => {
            let filter = Filter::Like {
                field: column.clone(),
                pattern: pattern.clone(),
            };
            if *negated {
                Ok(Filter::Not(Box::new(filter)))
            } else {
                Ok(filter)
            }
        }
        SqlExpr::Regexp {
            column,
            pattern,
            negated,
        } => {
            let filter = Filter::Regexp {
                field: column.clone(),
                pattern: pattern.clone(),
            };
            if *negated {
                Ok(Filter::Not(Box::new(filter)))
            } else {
                Ok(filter)
            }
        }
        SqlExpr::In {
            column,
            values,
            negated,
        } => {
            let filter = Filter::In {
                field: column.clone(),
                values: values.clone(),
            };
            if *negated {
                Ok(Filter::Not(Box::new(filter)))
            } else {
                Ok(filter)
            }
        }
        SqlExpr::Between {
            column,
            low,
            high,
            negated,
        } => {
            let low_val = match low.as_ref() {
                SqlExpr::Literal(v) => v.clone(),
                _ => {
                    return Err(ApexError::QueryParseError(
                        "BETWEEN bounds must be literals".to_string(),
                    ))
                }
            };
            let high_val = match high.as_ref() {
                SqlExpr::Literal(v) => v.clone(),
                _ => {
                    return Err(ApexError::QueryParseError(
                        "BETWEEN bounds must be literals".to_string(),
                    ))
                }
            };

            // Use native Range filter for single-pass BETWEEN evaluation
            let filter = Filter::Range {
                field: column.clone(),
                low: low_val,
                high: high_val,
                low_inclusive: true,
                high_inclusive: true,
            };
            if *negated {
                Ok(Filter::Not(Box::new(filter)))
            } else {
                Ok(filter)
            }
        }
        SqlExpr::IsNull { column, negated } => {
            let filter = Filter::Compare {
                field: column.clone(),
                op: CompareOp::Equal,
                value: Value::Null,
            };
            if *negated {
                Ok(Filter::Not(Box::new(filter)))
            } else {
                Ok(filter)
            }
        }
        SqlExpr::Paren(inner) => sql_expr_to_filter(inner),
        SqlExpr::Literal(Value::Bool(true)) => Ok(Filter::True),
        SqlExpr::Literal(Value::Bool(false)) => Ok(Filter::False),
        _ => Err(ApexError::QueryParseError(format!(
            "Cannot convert expression to filter: {:?}",
            expr
        ))),
    }
}
