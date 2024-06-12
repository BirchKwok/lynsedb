import threading
import msgpack
import os
import queue as Queue


class SingletonMeta(type):
    _instances = {}
    _lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
            return cls._instances[cls]


class MessageQueue(metaclass=SingletonMeta):
    def __init__(self, base_dir, save_interval=3, max_unsaved_changes=10, batch_size=100):
        self.partitions = {}
        self.base_dir = base_dir
        self.save_interval = save_interval
        self.max_unsaved_changes = max_unsaved_changes
        self.batch_size = batch_size
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.threads = {}

        # Ensure the base directory exists
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

    def _load(self, partition_name):
        file_path = os.path.join(self.base_dir, f'{partition_name}')
        q = Queue.Queue()
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                data = msgpack.unpack(f)
                for item in data:
                    q.put(item)
        return q

    def _save(self, partition_name):
        file_path = os.path.join(self.base_dir, f'{partition_name}')
        q = self.partitions[partition_name]['message_queue']
        items = list(q.queue)
        with open(file_path, 'wb') as f:
            msgpack.pack(items, f)
        print(f"Partition '{partition_name}' saved to {file_path}")

    def _background_serialize(self, partition_name):
        while not self.stop_event.is_set():
            with self.partitions[partition_name]['condition']:
                self.partitions[partition_name]['condition'].wait(timeout=self.save_interval)
                if self.partitions[partition_name]['unsaved_changes'] > 0:
                    self._save(partition_name)
                    self.partitions[partition_name]['unsaved_changes'] = 0

    def _initialize_partition(self, partition_name):
        if partition_name not in self.partitions:
            self.partitions[partition_name] = {
                'message_queue': self._load(partition_name),
                'unsaved_changes': 0,
                'condition': threading.Condition(self.lock)
            }
            self.threads[partition_name] = threading.Thread(target=self._background_serialize, args=(partition_name,))
            self.threads[partition_name].start()

    def put(self, item, partition_name='default'):
        self._initialize_partition(partition_name)
        with self.partitions[partition_name]['condition']:
            self.partitions[partition_name]['message_queue'].put(item)
            self.partitions[partition_name]['unsaved_changes'] += 1
            self.partitions[partition_name]['condition'].notify_all()

    def get(self, partition_name=None):
        if partition_name:
            self._initialize_partition(partition_name)
            with self.partitions[partition_name]['condition']:
                item = self.partitions[partition_name]['message_queue'].get()
                self.partitions[partition_name]['unsaved_changes'] += 1
                self.partitions[partition_name]['condition'].notify_all()
                return item
        else:
            for pname, pdata in self.partitions.items():
                with pdata['condition']:
                    if not pdata['message_queue'].empty():
                        item = pdata['message_queue'].get()
                        pdata['unsaved_changes'] += 1
                        pdata['condition'].notify_all()
                        return item
            raise Queue.Empty()

    def get_batch(self, partition_name='default', batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        self._initialize_partition(partition_name)
        items = []
        with self.partitions[partition_name]['condition']:
            for _ in range(batch_size):
                if self.partitions[partition_name]['message_queue'].empty():
                    break
                items.append(self.partitions[partition_name]['message_queue'].get())
            self.partitions[partition_name]['unsaved_changes'] += len(items)
            self.partitions[partition_name]['condition'].notify_all()
        return items

    def empty(self, partition_name='default'):
        self._initialize_partition(partition_name)
        with self.lock:
            return self.partitions[partition_name]['message_queue'].empty()

    def stop(self):
        self.stop_event.set()
        for partition_name in self.partitions:
            with self.partitions[partition_name]['condition']:
                self.partitions[partition_name]['condition'].notify_all()
            self.threads[partition_name].join()
            self._save(partition_name)

    def __del__(self):
        if hasattr(self, 'stop_event'):
            self.stop()
