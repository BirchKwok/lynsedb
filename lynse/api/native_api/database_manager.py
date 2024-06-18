import json

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

        with open(self.root_path / 'databases.json', 'w') as f:
            json.dump(databases, f)

    def list_database(self):
        return self.update_database()

    def update_database(self):
        if not (self.root_path / 'databases.json').exists():
            return []

        folders = [f.name for f in (self.root_path / 'databases').iterdir() if f.is_dir()]

        with open(self.root_path / 'databases.json', 'r') as f:
            databases = json.load(f)

        new_databases = []
        for db in databases:
            if db in folders:
                new_databases.append(db)

        with open(self.root_path / 'databases.json', 'w') as f:
            json.dump(new_databases, f)

        return new_databases
