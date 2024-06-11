import os
import threading
import time
from pathlib import Path

import httpx
import numpy as np
import msgpack


class WAL:
    def __init__(self, collection_name, wal_dir, max_entries_per_file=1000, max_files=4):
        self.collection_name = collection_name
        self.mq_path = Path(wal_dir).parent / 'message_queue'
        self.wal_dir = wal_dir
        self.max_entries_per_file = max_entries_per_file
        self.max_files = max_files
        self.current_file_index = 0
        self.current_entry_count = 0
        self.lock = threading.Lock()
        self.setup_wal_files()
        self.commit_condition = threading.Condition(self.lock)
        self.commit_thread = threading.Thread(target=self.commit_log_files)
        self.commit_thread.daemon = True
        self.commit_thread.start()
        self.log_thread = threading.Thread(target=self._log_operation)
        self.log_thread.daemon = True
        self.log_thread.start()

    def setup_wal_files(self):
        if not os.path.exists(self.wal_dir):
            os.makedirs(self.wal_dir)

        self.wal_files = [os.path.join(self.wal_dir, f'wal_{i}.log') for i in range(self.max_files)]
        self.current_file_index = 0
        self.current_entry_count = 0

    def log_operation(self, operation, vector_id, vector, field=None):
        try:
            response = httpx.post('http://localhost:7785/put',
                                  json={"item": {
                                      'operation': operation, 'vector_id': vector_id,
                                      'vector': vector.tolist() if isinstance(vector, np.ndarray) else vector,
                                      'field': field
                                  },
                                      "partition": self.collection_name, "file_path": str(self.mq_path)})
            response.raise_for_status()
            return response.text
        except Exception as e:
            print(f"Failed to log operation: {e}")

    def _log_operation(self):
        while True:
            try:
                log_entry = httpx.post('http://localhost:7785/get', json={"partition": self.collection_name,
                                                                          "file_path": str(self.mq_path)}).json()

                if log_entry:
                    with self.commit_condition:
                        while True:
                            if self.current_entry_count < self.max_entries_per_file:
                                with open(self.wal_files[self.current_file_index], 'ab') as f:
                                    f.write(msgpack.packb(log_entry))
                                    f.flush()
                                    os.fsync(f.fileno())
                                    self.current_entry_count += 1
                                    break
                            else:
                                self.current_file_index = (self.current_file_index + 1) % self.max_files
                                self.current_entry_count = 0
                                if os.path.getsize(self.wal_files[self.current_file_index]) > 0:
                                    print("All log files are full, waiting for commit...")
                                    self.commit_condition.wait()
            except Exception as e:
                print(f"Failed to log operation from queue: {e}")
                time.sleep(1)

    def replay(self):
        with self.lock:
            for file in self.wal_files:
                if not os.path.exists(file):
                    continue
                with open(file, 'rb') as f:
                    unpacker = msgpack.Unpacker(f, raw=False)
                    for log_entry in unpacker:
                        yield log_entry

    def commit_log_files(self):
        while True:
            time.sleep(5)  # Adjust this interval as needed
            with self.commit_condition:
                for file in self.wal_files:
                    if os.path.exists(file) and os.path.getsize(file) > 0:
                        self._commit_and_clear_log(file)
                self.commit_condition.notify_all()

    @staticmethod
    def _commit_and_clear_log(file):
        with open(file, 'rb') as f:
            unpacker = msgpack.Unpacker(f, raw=False)
            for log_entry in unpacker:
                # Implement your commit logic here
                pass
        with open(file, 'wb') as f:
            f.truncate()

    def clear(self):
        with self.lock:
            for file in self.wal_files:
                with open(file, 'wb') as f:
                    f.truncate()
