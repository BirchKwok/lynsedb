from pathlib import Path
from typing import Union
from filelock import FileLock


class GlobalWritingLock:
    def __init__(self, lock_file_path: Union[str, Path]):
        self.lock = FileLock(lock_file_path)
        self.lock_file_path = lock_file_path

    def acquire(self, timeout: float = -1):
        """
        获取锁，可以指定超时时间
        :param timeout: 超时时间（秒），默认-1表示永远等待
        :raises Timeout: 当超时时抛出
        """
        try:
            self.lock.acquire(timeout=timeout)
        except Exception as e:
            raise RuntimeError(f"获取锁失败: {str(e)}")

    def release(self):
        """释放锁"""
        try:
            if self.is_locked():
                self.lock.release()
        except Exception as e:
            raise RuntimeError(f"释放锁失败: {str(e)}")

    def is_locked(self) -> bool:
        """检查锁是否被获取"""
        return self.lock.is_locked()

    def __enter__(self):
        """支持with语句的上下文管理器"""
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文管理器时释放锁"""
        self.release()

    def __del__(self):
        """析构函数，安全地关闭锁"""
        try:
            if hasattr(self, 'lock'):
                self.lock.close()
        except Exception:
            pass  # 忽略清理过程中的错误
