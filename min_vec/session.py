# database insert_session

class DatabaseSession:
    def __init__(self, db):
        self.db = db

    def __enter__(self):
        return self.db

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.db._COMMIT_FLAG:
            self.db.commit()

        return False
