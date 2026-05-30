//! Query Scheduler - enables parallel query execution using thread pool
//!
//! This module provides a thread-pool based query scheduler that allows
//! multiple queries to execute concurrently in Rust, bypassing Python's GIL.

use parking_lot::{Condvar, Mutex};
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
use std::thread;

use crate::query::executor::{ApexExecutor, ApexResult};
use crate::query::sql_parser::SqlParser;

/// Maximum number of worker threads
pub const DEFAULT_THREADS: usize = 4;

/// Query execution result
pub enum QueryResult {
    Data(arrow::record_batch::RecordBatch),
    Error(String),
    Done,
}

/// A query task submitted to the scheduler
pub struct QueryTask {
    pub sql: String,
    pub table_path: PathBuf,
    pub result_sender: Sender<QueryResult>,
}

/// Thread pool based query executor
pub struct ThreadPoolExecutor {
    worker_handles: Vec<thread::JoinHandle<()>>,
    task_queue: Arc<Mutex<VecDeque<QueryTask>>>,
    condvar: Arc<Condvar>,
    shutdown_flag: Arc<AtomicBool>,
    active_count: Arc<AtomicUsize>,
}

impl ThreadPoolExecutor {
    /// Create a new thread pool with specified number of threads
    pub fn new(num_threads: usize) -> Self {
        let task_queue = Arc::new(Mutex::new(VecDeque::new()));
        let condvar = Arc::new(Condvar::new());
        let shutdown_flag = Arc::new(AtomicBool::new(false));
        let active_count = Arc::new(AtomicUsize::new(0));

        let mut worker_handles = Vec::with_capacity(num_threads);

        for _ in 0..num_threads {
            let queue = Arc::clone(&task_queue);
            let cvar = Arc::clone(&condvar);
            let flag = Arc::clone(&shutdown_flag);
            let active = Arc::clone(&active_count);

            let handle = thread::spawn(move || {
                loop {
                    let task = {
                        let mut queue = queue.lock();
                        // Park on condvar until a task arrives or shutdown is requested
                        while queue.is_empty() && !flag.load(Ordering::Relaxed) {
                            cvar.wait(&mut queue);
                        }
                        if flag.load(Ordering::Relaxed) {
                            break;
                        }
                        queue.pop_front()
                    };

                    if let Some(t) = task {
                        active.fetch_add(1, Ordering::Relaxed);
                        Self::execute_query(t);
                        active.fetch_sub(1, Ordering::Relaxed);
                    }
                }
            });

            worker_handles.push(handle);
        }

        Self {
            worker_handles,
            task_queue,
            condvar,
            shutdown_flag,
            active_count,
        }
    }

    /// Execute a single query
    fn execute_query(task: QueryTask) {
        let result = (|| -> Result<QueryResult, String> {
            // Use the execute function that takes storage_path
            let exec_result =
                ApexExecutor::execute(&task.sql, &task.table_path).map_err(|e| e.to_string())?;

            match exec_result {
                ApexResult::Data(batch) => Ok(QueryResult::Data(batch)),
                ApexResult::Empty(_) => {
                    let schema = Arc::new(arrow::datatypes::Schema::empty());
                    let batch = arrow::record_batch::RecordBatch::new_empty(schema);
                    Ok(QueryResult::Data(batch))
                }
                ApexResult::Scalar(val) => {
                    let schema = Arc::new(arrow::datatypes::Schema::new(vec![
                        arrow::datatypes::Field::new(
                            "result",
                            arrow::datatypes::DataType::Int64,
                            false,
                        ),
                    ]));
                    let array: arrow::array::ArrayRef =
                        Arc::new(arrow::array::Int64Array::from(vec![val]));
                    let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![array])
                        .map_err(|e| e.to_string())?;
                    Ok(QueryResult::Data(batch))
                }
            }
        })();

        let result = match result {
            Ok(r) => r,
            Err(e) => QueryResult::Error(e),
        };

        let _ = task.result_sender.send(result);
    }

    /// Submit a query to the thread pool
    pub fn submit(&self, sql: String, table_path: PathBuf) -> Receiver<QueryResult> {
        let (sender, receiver) = channel();

        let task = QueryTask {
            sql,
            table_path,
            result_sender: sender,
        };

        {
            let mut queue = self.task_queue.lock();
            queue.push_back(task);
        }
        self.condvar.notify_one();

        receiver
    }

    /// Get number of active queries
    pub fn active_count(&self) -> usize {
        self.active_count.load(Ordering::Relaxed)
    }

    /// Shutdown the thread pool
    pub fn shutdown(&mut self) {
        self.shutdown_flag.store(true, Ordering::Relaxed);
        self.condvar.notify_all();
        for handle in self.worker_handles.drain(..) {
            let _ = handle.join();
        }
    }
}

impl Drop for ThreadPoolExecutor {
    fn drop(&mut self) {
        // catch_unwind: on Windows, thread-local destructors may run after
        // parking_lot's internal TLS is torn down, causing a panic in join().
        // Absorb it so the process exits cleanly after all tests pass.
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.shutdown();
        }));
    }
}

// Thread-local scheduler storage
thread_local! {
    static SCHEDULER: RefCell<Option<Box<ThreadPoolExecutor>>> = RefCell::new(None);
}

use std::cell::RefCell;

/// Initialize the global scheduler with specified number of threads
pub fn init_scheduler(num_threads: usize) {
    SCHEDULER.with(|s| {
        *s.borrow_mut() = Some(Box::new(ThreadPoolExecutor::new(num_threads)));
    });
}

/// Initialize with default threads (4)
pub fn init_scheduler_default() {
    init_scheduler(DEFAULT_THREADS);
}

/// Execute query through scheduler - returns receiver to get result
/// Returns None if scheduler not initialized
pub fn execute_through_scheduler(
    sql: String,
    table_path: PathBuf,
) -> Option<Receiver<QueryResult>> {
    SCHEDULER.with(|s| s.borrow().as_ref().map(|s| s.submit(sql, table_path)))
}

/// Check if scheduler is initialized
pub fn is_scheduler_initialized() -> bool {
    SCHEDULER.with(|s| s.borrow().is_some())
}

/// Get active query count
pub fn get_active_count() -> Option<usize> {
    SCHEDULER.with(|s| s.borrow().as_ref().map(|s| s.active_count()))
}
