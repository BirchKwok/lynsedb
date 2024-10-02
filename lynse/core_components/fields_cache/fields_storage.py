import sqlite3
import msgpack
from typing import Dict, List, Any, Union, Tuple
from pathlib import Path


class FieldsStorage:
    """
    Fields storage class.
    """
    def __init__(self, filepath=None):
        """
        Initialize the fields storage.

        Parameters:
            filepath: str
                The file path to the storage.
        """
        if filepath is None:
            raise ValueError("You must provide a file path.")

        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(str(self.filepath), check_same_thread=False)
        self.cursor = self.conn.cursor()
        self._initialize_database()

    def _initialize_database(self):
        """
        Initialize the database.
        """
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS records (
                internal_id INTEGER PRIMARY KEY AUTOINCREMENT,
                external_id INTEGER UNIQUE,
                data BLOB
            )
        """)
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_external_id ON records (external_id)")
        self.conn.commit()

    def store(self, data: dict, external_id: int) -> int:
        """
        Store a record in the cache.

        Parameters:
            data: dict
                The record to be stored.
            external_id: int
                The external ID of the record.

        Returns:
            int: The internal ID of the record.
        """
        if not isinstance(data, dict):
            raise ValueError("Only dict-type data is allowed.")

        try:
            packed_data = msgpack.packb(data)
            self.cursor.execute(
                "INSERT INTO records (external_id, data) VALUES (?, ?)",
                (external_id, packed_data)
            )
            self.conn.commit()
            return self.cursor.lastrowid
        except sqlite3.IntegrityError:
            raise ValueError(f"external_id {external_id} already exists.")

    def retrieve_by_external_id(self, external_ids: Union[int, List[int]], include_external_id=True) -> List[dict]:
        """
        Retrieve records by external ID.

        Parameters:
            external_ids: Union[int, List[int]]
                The external ID or list of external IDs.
            include_external_id: bool
                If True, include the external ID in the records.

        Returns:
            List[dict]: List of records.
        """
        if isinstance(external_ids, int):
            external_ids = [external_ids]

        placeholders = ','.join('?' for _ in external_ids)
        query = f"SELECT external_id, data FROM records WHERE external_id IN ({placeholders})"
        self.cursor.execute(query, external_ids)
        results = []
        for row in self.cursor.fetchall():
            data = msgpack.unpackb(row[1])
            if include_external_id:
                data[':id:'] = row[0]
            results.append(data)
        return results

    def retrieve_all(self, include_external_id=True):
        """
        Retrieve all records.

        Parameters:
            include_external_id: bool
                If True, include the external ID in the records.

        Returns:
            List[dict]: List of records.
        """
        self.cursor.execute("SELECT external_id, data FROM records")
        for row in self.cursor.fetchall():
            data = msgpack.unpackb(row[1])
            if include_external_id:
                data[':id:'] = row[0]
            yield row[0], data

    def field_exists(self, field: str) -> bool:
        """
        Check if a field exists.

        Parameters:
            field: str
                The field to check.

        Returns:
            bool: True if the field exists, False otherwise.
        """
        field = field.strip(':')
        self.cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='index_{field}'")
        return self.cursor.fetchone() is not None

    def build_index(self, schema: Dict[str, type]):
        """
        Build an index for a field.

        Parameters:
            schema: Dict[str, type]
                The schema of the fields.
        """
        for field, field_type in schema.items():
            field = field.strip(':')
            if self.field_exists(field):
                continue  # Skip if the index already exists and does not need to be rebuilt

            sqlite_type = {int: 'INTEGER', float: 'REAL', str: 'TEXT'}[field_type]
            self.cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS index_{field} (
                    internal_id INTEGER,
                    value {sqlite_type},
                    FOREIGN KEY (internal_id) REFERENCES records (internal_id)
                )
            """)
            self.cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{field} ON index_{field} (value)")

            # build index for all existing records
            self.cursor.execute("SELECT internal_id, data FROM records")
            for internal_id, packed_data in self.cursor.fetchall():
                data = msgpack.unpackb(packed_data)
                if field in data:
                    self.cursor.execute(f"INSERT INTO index_{field} (internal_id, value) VALUES (?, ?)",
                                        (internal_id, data[field]))

        self.conn.commit()

    def remove_index(self, field_name: str):
        """
        Remove an index for a field.

        Parameters:
            field_name: str
                The name of the field.
        """
        self.cursor.execute(f"DROP TABLE IF EXISTS index_{field_name}")
        self.conn.commit()

    def search(self, field: str, value: Any) -> List[int]:
        """
        Search for records by field and value.

        Parameters:
            field: str
                The field to search.
            value: Any
                The value to search for.

        Returns:
            List[int]: The external IDs of the records.
        """
        # Check if the index exists
        self.cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='index_{field}'")
        if self.cursor.fetchone() is None:
            # If index doesn't exist, fall back to full table scan
            self.cursor.execute("SELECT external_id, data FROM records")
            results = []
            for row in self.cursor.fetchall():
                data = msgpack.unpackb(row[1])
                if field in data and data[field] == value:
                    results.append(row[0])  # Append external_id
            return results
        else:
            # If index exists, use it
            self.cursor.execute(f"""
                SELECT r.external_id
                FROM records r
                JOIN index_{field} i ON r.internal_id = i.internal_id
                WHERE i.value = ?
            """, (value,))
            return [row[0] for row in self.cursor.fetchall()]

    def range_search(self, field: str, start_value: Any, end_value: Any) -> List[int]:
        """
        Range search for records by field and value range.

        Parameters:
            field: str
                The field to search.
            start_value: Any
                The start value of the range.
            end_value: Any
                The end value of the range.

        Returns:
            List[int]: The external IDs of the records.
        """
        self.cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='index_{field}'")
        if self.cursor.fetchone() is None:
            # If index doesn't exist, fall back to full table scan
            self.cursor.execute("SELECT external_id, data FROM records")
            results = []
            for row in self.cursor.fetchall():
                data = msgpack.unpackb(row[1])
                if field in data and start_value <= data[field] <= end_value:
                    results.append(row[0])  # Append external_id
            return results
        else:
            # If index exists, use it
            self.cursor.execute(f"""
                SELECT r.external_id
                FROM records r
                JOIN index_{field} i ON r.internal_id = i.internal_id
                WHERE i.value BETWEEN ? AND ?
            """, (start_value, end_value))
            return [row[0] for row in self.cursor.fetchall()]

    def starts_with(self, field: str, prefix: str) -> List[int]:
        """
        Search for records by field and prefix.

        Parameters:
            field: str
                The field to search.
            prefix: str
                The prefix to search for.

        Returns:
            List[int]: The external IDs of the records.
        """
        self.cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='index_{field}'")
        if self.cursor.fetchone() is None:
            # If index doesn't exist, fall back to full table scan
            self.cursor.execute("SELECT external_id, data FROM records")
            results = []
            for row in self.cursor.fetchall():
                data = msgpack.unpackb(row[1])
                if field in data and isinstance(data[field], str) and data[field].startswith(prefix):
                    results.append(row[0])  # Append external_id
            return results
        else:
            # If index exists, use it
            self.cursor.execute(f"""
                SELECT r.external_id
                FROM records r
                JOIN index_{field} i ON r.internal_id = i.internal_id
                WHERE i.value LIKE ?
            """, (prefix + '%',))
            return [row[0] for row in self.cursor.fetchall()]

    def delete(self):
        """
        Delete the storage.
        """
        self.conn.close()
        if self.filepath.exists():
            self.filepath.unlink()

    def list_fields(self) -> Dict[str, str]:
        """
        List all fields in the cache, including those without indexes.

        Returns:
            Dict[str, str]: The fields of the cache with their types.
        """
        fields = {}

        # First, get all fields that have been indexed
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'index_%'")
        for (table_name,) in self.cursor.fetchall():
            field = table_name[6:]  # Remove 'index_' prefix
            self.cursor.execute(f"PRAGMA table_info({table_name})")
            for column in self.cursor.fetchall():
                if column[1] == 'value':
                    sqlite_type = column[2]
                    python_type = {'INTEGER': 'int', 'REAL': 'float', 'TEXT': 'str'}[sqlite_type]
                    fields[f":{field}:"] = python_type
                    break

        # Then, check all records to find fields without indexes
        self.cursor.execute("SELECT data FROM records LIMIT 1000")  # Limit the number of records checked for performance
        for (packed_data,) in self.cursor.fetchall():
            data = msgpack.unpackb(packed_data)
            for field, value in data.items():
                if f":{field}:" not in fields:
                    if isinstance(value, int):
                        fields[f":{field}:"] = 'int'
                    elif isinstance(value, float):
                        fields[f":{field}:"] = 'float'
                    elif isinstance(value, str):
                        fields[f":{field}:"] = 'str'

        return fields

    def __del__(self):
        """
        Delete the storage.
        """
        if hasattr(self, 'conn'):
            self.conn.close()

    def batch_store(self, data_list: List[Tuple[dict, int]]) -> List[int]:
        """
        Batch store multiple records in the cache.

        Parameters:
            data_list: List[Tuple[dict, int]]
                List of records to be stored. Each element is a tuple containing a data dictionary and its corresponding external ID.

        Returns:
            List[int]: List of internal IDs of the stored records.
        """
        try:
            self.cursor.execute("BEGIN TRANSACTION")

            # Prepare bulk insert data
            bulk_data = [(external_id, msgpack.packb(data)) for data, external_id in data_list]

            # Execute bulk insert
            self.cursor.executemany(
                "INSERT INTO records (external_id, data) VALUES (?, ?)",
                bulk_data
            )

            # Get inserted internal IDs
            self.cursor.execute("SELECT last_insert_rowid()")
            last_id = self.cursor.fetchone()[0]
            internal_ids = list(range(last_id - len(data_list) + 1, last_id + 1))

            self.conn.commit()
            return internal_ids
        except sqlite3.IntegrityError as e:
            self.conn.rollback()
            raise ValueError(f"Batch storage failed: {str(e)}")
        except Exception as e:
            self.conn.rollback()
            raise e
