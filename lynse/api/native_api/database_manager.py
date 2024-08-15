import json
import shutil

from spinesUtils.asserts import raise_if


class DatabaseManager:
    def __init__(self, root_path):
        self.root_path = root_path
        if not self.root_path.exists():
            self.root_path.mkdir(parents=True, exist_ok=True)

    def register(self, db_name: str):
        raise_if(TypeError, not isinstance(db_name, str), 'db_name must be a string')

        if not (self.root_path / 'databases.json').exists():
            with open(self.root_path / 'databases.json', 'w') as f:
                json.dump([db_name], f)
        else:
            with open(self.root_path / 'databases.json', 'r') as f:
                databases = json.load(f)
                if db_name in databases:
                    return
                databases.append(db_name)
            with open(self.root_path / 'databases.json', 'w') as f:
                json.dump(databases, f)

    def delete(self, db_name: str):
        raise_if(TypeError, not isinstance(db_name, str), 'db_name must be a string')
        if not (self.root_path / 'databases.json').exists():
            return

        databases = self.update_database()

        if db_name in databases:
            databases.remove(db_name)
            shutil.rmtree(self.root_path / db_name)

        with open(self.root_path / 'databases.json', 'w') as f:
            json.dump(databases, f)

    def list_database(self):
        return self.update_database()

    def update_database(self):
        if not (self.root_path / 'databases.json').exists():
            return []

        self.root_path.mkdir(parents=True, exist_ok=True)

        # if a folder does not contain fingerprint.db, exclude it
        folders = [x.name for x in self.root_path.iterdir()
                   if x.is_dir() and (x / '.fingerprint').exists()]

        with open(self.root_path / 'databases.json', 'r') as f:
            databases = json.load(f)

        for db in folders:
            if db not in databases:
                databases.append(db)

        not_existed_db = []
        for db in databases:
            if db not in folders:
                not_existed_db.append(db)

        for db in not_existed_db:
            databases.remove(db)

        with open(self.root_path / 'databases.json', 'w') as f:
            json.dump(databases, f)

        return databases
