# database insert_session

class DatabaseSession:
    def __init__(self, db):
        self.db = db

    def __enter__(self):
        # 可以在这里执行一些初始化操作，如打开数据库连接
        return self.db

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 在这里执行清理操作，如提交更改或关闭数据库连接
        if not self.db._COMMIT_FLAG:
            self.db.commit()

        # 根据需要处理异常
        return False  # 让任何异常正常抛出
