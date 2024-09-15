import numpy as np
import json
import os
import shutil
from pathlib import Path

from spinesUtils.timer import Timer

from ..core_components.locks import ThreadLock


class WALStorage:
    def __init__(self, collection_name, chunk_size, storage_path, flush_interval=5):
        self.storage_path = Path(storage_path) / "wal"
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.log_dir = self.storage_path / f"log"
        self.read_dir = self.storage_path / f"read"
        self.state_dir = self.storage_path / f"state"

        # rename the folder, the version upgrade has resulted in compatibility modifications
        for _ in [self.storage_path / f"{collection_name}-log", self.storage_path / f"{collection_name}-read",
                  self.storage_path / f"{collection_name}-state"]:
            if _.exists():
                _.rename(self.storage_path / f"{_.name.split('-')[-1].strip()}")

        self.log_dir.mkdir(exist_ok=True)
        self.read_dir.mkdir(exist_ok=True)
        self.state_dir.mkdir(exist_ok=True)

        self.chunk_size = chunk_size
        self.flush_interval = flush_interval
        self.file_id = 0  # File ID to maintain the naming order
        self.lock = ThreadLock()

        # Buffer to store data, indices, and fields
        self.buffer_data = []
        self.buffer_indices = []
        self.buffer_fields = []

        self.timer = Timer()
        self.timer.start()

        # Start periodic flush thread
        self.running = True

    def write_log_data(self, data, indices, fields):
        with self.lock:
            if not isinstance(data, np.ndarray):
                data = np.vstack(data)

            if not isinstance(indices, np.ndarray):
                indices = np.array(indices)

            if not isinstance(fields, list) or not all(isinstance(field, dict) for field in fields):
                raise ValueError("fields should be a list of dictionaries")

            self.buffer_data.append(data)
            self.buffer_indices.append(indices)
            self.buffer_fields.extend(fields)

            # Check if buffer is full
            if sum(len(chunk) for chunk in self.buffer_data) >= self.chunk_size:
                self._flush_buffer_to_disk()

            if self.timer.last_timestamp_diff() >= self.flush_interval:
                self._flush_buffer_to_disk()
                self.timer.middle_point()

    def _flush_buffer_to_disk(self):
        with self.lock:
            # Check if buffer is empty
            if not self.buffer_data:
                return

            # Concatenate buffered data
            try:
                data = np.concatenate(self.buffer_data, axis=0)
                indices = np.concatenate(self.buffer_indices, axis=0)
                fields = self.buffer_fields
            except Exception as e:
                print(f"Error during concatenation: {e}")
                return

            num_rows = data.shape[0]
            start = 0

            while start < num_rows:
                end = min(start + self.chunk_size, num_rows)
                chunk_data = data[start:end]
                chunk_indices = indices[start:end]
                chunk_fields = fields[start:end]

                # Serialize fields using JSON
                chunk_fields_serialized = [json.dumps(field) for field in chunk_fields]

                # Save to NPZ file
                log_file_path = self.log_dir / f"log_{self.file_id}.npz"
                try:
                    np.savez(log_file_path, data=chunk_data, indices=chunk_indices,
                             fields=np.array(chunk_fields_serialized))
                except Exception as e:
                    print(f"Error during NPZ write: {e}")
                    return

                self.file_id += 1
                self._write_state_file(self.file_id - 1)

                start += self.chunk_size

            # Clear the buffer
            self.buffer_data = []
            self.buffer_indices = []
            self.buffer_fields = []

    def _write_state_file(self, file_id):
        state_file_path = self.state_dir / f"state_{file_id}.txt"
        try:
            with open(state_file_path, 'w') as f:
                f.write("COMPLETED")
        except Exception as e:
            print(f"Error during state file write: {e}")

    def _remove_state_file(self, file_id):
        state_file_path = self.state_dir / f"state_{file_id}.txt"
        if state_file_path.exists():
            os.remove(state_file_path)

    @staticmethod
    def read_log_data(log_filepath):
        try:
            with np.load(log_filepath, allow_pickle=True) as npzfile:
                data = npzfile['data']
                indices = npzfile['indices']
                fields_serialized = npzfile['fields']

                # Deserialize fields using JSON
                fields = []
                for i, field in enumerate(fields_serialized):
                    try:
                        fields.append(json.loads(field))
                    except Exception as e:
                        print(f"Error unpacking field at index {i}: {field}. Error: {e}")
                        raise

            return data, indices, fields
        except (IOError, OSError) as e:
            print(f"Error reading file {log_filepath}, it may be corrupted. Error: {e}")
            return None, None, None

    def get_file_iterator(self):
        with self.lock:
            self._flush_buffer_to_disk()

            file_ids = sorted([int(i.stem.split('_')[-1]) for i in self.log_dir.glob('log_*.npz')])
            for file_id in file_ids:
                log_path = self.log_dir / f"log_{file_id}.npz"
                state_file_path = self.state_dir / f"state_{file_id}.txt"
                if state_file_path.exists():
                    read_flag = self.read_dir / f"read_{file_id}.flag"
                    if not read_flag.exists():
                        data, indices, fields = self.read_log_data(log_path)
                        if data is not None and indices is not None and fields is not None:
                            yield data, indices, fields
                            read_flag.touch()  # Mark this file as read
                        else:
                            print(f"Skipping corrupted file: {log_path}")
                    self._remove_state_file(file_id)

    @property
    def file_number(self):
        return len(list(self.log_dir.glob('log_*.npz')))

    def cleanup(self):
        with self.lock:
            try:
                if self.storage_path.exists():
                    shutil.rmtree(self.storage_path)
            except Exception as e:
                print(f"Error during cleanup: {e}")

    def reincarnate(self):
        with self.lock:
            self.cleanup()
            self.storage_path.mkdir(parents=True, exist_ok=True)
            self.log_dir.mkdir(exist_ok=True)
            self.read_dir.mkdir(exist_ok=True)
            self.state_dir.mkdir(exist_ok=True)
            self.file_id = 0

    def stop(self):
        self.running = False
