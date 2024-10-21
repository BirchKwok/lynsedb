import os

import pandas as pd
import pyarrow as pa
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import time
import shutil
import threading
import duckdb
import concurrent.futures
from functools import lru_cache
import hashlib
import msgpack

from ..bitset import BitSet
from .query import QueryBuilder


class LyStorage:
    """
    A storage class for managing data in a directory.

    Usage:
        storage = LyStorage("path/to/data")
        storage.bulk_add([{"name": "John", "age": 30}])
        storage.delete_where("age > 30")
    """

    def __init__(self, data_path: str, schema: Optional[pa.Schema] = None,
                 lazy_load: bool = True, overwrite=False,
                 max_rows_per_region: int = 1_000_000):
        """
        Initialize the LyStorage instance.
        """
        self.data_path = Path(data_path)
        self.schema = schema  # Initially set to the provided schema or None
        self.lazy_load = lazy_load
        self.overwrite = overwrite
        self.max_rows_per_region = max_rows_per_region

        if overwrite:
            self.delete()

        if not self.data_path.exists():
            self.data_path.mkdir(parents=True, exist_ok=True)

        self._lifecycle_path = self.data_path / f"__lifecycle"
        if not self._lifecycle_path.exists():
            self._lifecycle_path.mkdir(parents=True, exist_ok=True)

        # Load schema from disk if schema is None
        if self.schema is None:
            self._load_schema_from_disk()

        self.region_meta_file = self.data_path / "regions_meta.lymp"
        if not self.region_meta_file.exists() or overwrite:
            self.regions_meta = {
                'version_counter': 0,
                'regions': []
            }
            self.regions = []
            self.version_counter = 0
            self._save_regions_meta()
        else:
            self.regions_meta = msgpack.load(open(self.region_meta_file, 'rb'))
            self.regions = self.regions_meta.get('regions', [])
            self.version_counter = self.regions_meta.get('version_counter', 0)

            for region in self.regions:
                region['bitset'] = None

        self.current_region = self.regions[-1] if self.regions else None
        if not self.current_region or self.current_region['row_count'] >= self.max_rows_per_region:
            self._create_new_region()

        self.table = None  # Initialize as None
        self._duckdb_conn = duckdb.connect(database=str(self.data_path / "__virtual.ddb"), read_only=False)
        self._register_table()

        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=min(os.cpu_count(), 32))
        self.lock = threading.RLock()

        if not lazy_load:
            self._load_all_regions()

    def _load_schema_from_disk(self):
        """
        Load the schema from disk if it exists.
        """
        schema_file = self._lifecycle_path / "schema.arrow"
        if schema_file.exists():
            with open(schema_file, 'rb') as f:
                schema_buffer = f.read()
                self.schema = pa.ipc.read_schema(pa.BufferReader(schema_buffer))

    def _create_new_region(self):
        """
        Create a new region.
        """
        new_region = {
            "region_id": len(self.regions) + 1,
            "start_row": self.current_region['start_row'] + self.current_region['row_count'] if self.current_region else 0,
            "row_count": 0,
            "data_files": [],
            "versions": [],
            "bitset": BitSet()
        }
        self.regions.append(new_region)
        self.current_region = new_region
        self._save_regions_meta()

    def _save_regions_meta(self):
        """
        Save the regions metadata without including the BitSet objects.
        """
        # Create a serializable copy of regions without 'bitset'
        serializable_regions = []
        for region in self.regions:
            # Copy all fields except 'bitset'
            serializable_region = {k: v for k, v in region.items() if k != 'bitset'}
            serializable_regions.append(serializable_region)

        # Update the regions_meta dictionary
        self.regions_meta['regions'] = serializable_regions
        self.regions_meta['version_counter'] = self.version_counter

        # Serialize the regions_meta to the meta file
        with open(self.region_meta_file, 'wb') as f:
            msgpack.dump(self.regions_meta, f)

    def _load_all_regions(self):
        """
        Load all regions' data and BitSets.
        """
        tables = []
        for region in self.regions:
            if region['row_count'] == 0 or not region['data_files']:
                continue

            # Ensure 'bitset' is initialized
            if region['bitset'] is None:
                region['bitset'] = BitSet(size=region['row_count'], fill=1)

            for data_file in region['data_files']:
                full_path = self.data_path / data_file
                try:
                    with pa.memory_map(full_path.as_posix(), 'r') as source:
                        table = pa.ipc.open_file(source).read_all()

                        # Apply BitSet to filter active rows
                        bitset = region['bitset']
                        row_indices = list(range(len(table)))
                        active_indices = [i for i in row_indices if bitset.get_bit(i)]

                        if active_indices:
                            active_table = table.take(active_indices)

                            # Check if '__rowid' already exists
                            if '__rowid' not in active_table.schema.names:
                                # Add '__rowid' column
                                global_row_indices = [region['start_row'] + i for i in active_indices]
                                rowid_array = pa.array(global_row_indices, type=pa.int64())
                                active_table = active_table.append_column('__rowid', rowid_array)

                            tables.append(active_table)
                except pa.lib.ArrowInvalid as e:
                    print(f"Error: Unable to read data file {data_file}. Details: {e}")

        if tables:
            try:
                self.table = pa.concat_tables(tables)
            except pa.lib.ArrowInvalid as e:
                print(f"Error: Unable to concatenate tables. Details: {e}")
                self.table = None
        else:
            self.table = None

        self._register_table()

    def _register_table(self):
        """
        Register the table to the DuckDB connection.
        """
        try:
            self._duckdb_conn.unregister("storage")
        except duckdb.InvalidInputException:
            pass

        if self.table is not None and len(self.table.schema.names) > 0:
            self._duckdb_conn.register("storage", self.table)

    def _get_data_file(self, region_id: int, timestamp: int, segment: int) -> str:
        """
        Get the data file path based on region, timestamp, and segment.

        Parameters:
            region_id (int): The ID of the region.
            timestamp (int): The timestamp of the version.
            segment (int): The segment number of the file.

        Returns:
            str: The path to the data file.
        """
        file_hash = hashlib.md5(f"{region_id}_{timestamp}_{segment}".encode()).hexdigest()
        return os.path.join(self.data_path, f"region_{region_id}_{file_hash}.ly")

    def _save_schema_to_disk(self):
        """
        Save the schema to disk for later loading.
        """
        schema_file = self._lifecycle_path / "schema.arrow"
        with open(schema_file, 'wb') as f:
            f.write(self.schema.serialize())

    def bulk_add(self, data: Union[List[Dict[str, Any]], pd.DataFrame, pa.Table], description: Optional[str] = None):
        """
        Bulk add multiple rows of data to the storage.
        """
        if data is None or (isinstance(data, (list, pd.DataFrame, pa.Table)) and len(data) == 0):
            return

        with self.lock:
            self._load_all_regions()

            if isinstance(data, list):
                try:
                    new_data = {k: [d[k] for d in data] for k in data[0].keys()}
                except KeyError as e:
                    print(f"Failed to add data, missing column: {e}")
                    return
                df = pd.DataFrame(new_data)
                if self.schema:
                    new_table = pa.Table.from_pandas(df, schema=self.schema)
                else:
                    new_table = pa.Table.from_pandas(df)
                    # If schema is None, set it from the new_table
                    self.schema = new_table.schema
                    self._save_schema_to_disk()
            elif isinstance(data, pd.DataFrame):
                data = data.copy()
                if self.schema:
                    new_table = pa.Table.from_pandas(data, schema=self.schema)
                else:
                    new_table = pa.Table.from_pandas(data)
                    # If schema is None, set it from the new_table
                    self.schema = new_table.schema
                    self._save_schema_to_disk()
            elif isinstance(data, pa.Table):
                if self.schema is None:
                    self.schema = data.schema
                    self._save_schema_to_disk()
                new_table = data
            else:
                raise ValueError(
                    "Unsupported data type. Please provide a list of dictionaries, pandas DataFrame, or pyarrow Table.")

            # Check if '__rowid' already exists
            if '__rowid' not in new_table.schema.names:
                # Compute global row indices for the new rows
                start_row = self.current_region['start_row'] + self.current_region['row_count']
                new_rowids = list(range(start_row, start_row + len(new_table)))
                rowid_array = pa.array(new_rowids, type=pa.int64())
                new_table = new_table.append_column('__rowid', rowid_array)

            # Update BitSet and region row count, and add data to region
            new_rows = len(new_table)
            rows_to_add = new_rows
            regions_modified = set()

            while rows_to_add > 0:
                remaining_space = self.max_rows_per_region - self.current_region['row_count']
                if remaining_space <= 0:
                    self._create_new_region()
                    remaining_space = self.max_rows_per_region

                rows_in_this_iteration = min(remaining_space, rows_to_add)
                start_index = new_rows - rows_to_add
                data_slice = new_table.slice(start_index, rows_in_this_iteration)

                # Update BitSet
                self.current_region['bitset'].resize(self.current_region['row_count'] + rows_in_this_iteration, fill=1)

                # Add data to current region
                self._add_to_current_region(data_slice)

                rows_to_add -= rows_in_this_iteration
                regions_modified.add(self.current_region['region_id'])

            # Update version information
            for region_id in regions_modified:
                region = next(r for r in self.regions if r['region_id'] == region_id)
                self.version_counter += 1
                self._update_version(description or "Bulk add data", region=region)

    def _add_to_current_region(self, table: pa.Table):
        region = self.current_region
        timestamp = int(time.time())
        segment = len(region['data_files'])
        data_file = self._get_data_file(region['region_id'], timestamp, segment)

        with pa.OSFile(data_file, 'wb') as sink:
            writer = pa.ipc.new_file(sink, table.schema)
            writer.write_table(table)
            writer.close()
        region['data_files'].append(os.path.basename(data_file))
        region['row_count'] += len(table)

        self._save_regions_meta()

    def delete_where(self, condition: Union[str, int, list], description: Optional[str] = None):
        """
        Delete rows matching a SQL condition.

        Parameters:
            condition (str or int or list): The condition to match for deletion.
            description (str): Description of the deletion.
        """
        with self.lock:
            # Ensure table is up-to-date
            self._load_all_regions()

            try:
                if isinstance(condition, str):
                    # Execute query to get rowids
                    query = f"SELECT __rowid FROM storage WHERE {condition}"
                    df = self.execute_sql(query)
                    if df.empty:
                        return
                    rowids = df['__rowid'].tolist()
                else:
                    if isinstance(condition, int):
                        rowids = [condition]

                self.bulk_delete_rows(rowids, description)
            except Exception as e:
                print(f"Failed to execute delete_where: {e}")

    def restore_where(self, condition: Union[str, int, list], description: Optional[str] = None):
        """
        Restore rows matching a SQL condition.

        Parameters:
            condition (str or int or list): The condition to match for deletion.
            description (str): Description of the restoration.
        """
        with self.lock:
            # Ensure table is up-to-date
            self._load_all_regions()

            try:
                if isinstance(condition, str):
                    # Execute query to get __rowid
                    query = f"SELECT __rowid FROM storage WHERE {condition}"
                    df = self.execute_sql(query)
                    if df.empty:
                        return
                    rowids = df['__rowid'].tolist()
                else:
                    if isinstance(condition, int):
                        rowids = [condition]

                self.bulk_restore_rows(rowids, description)
            except Exception as e:
                print(f"Failed to execute restore_where: {e}")

    def delete_row(self, row_index: int, description: Optional[str] = None):
        """
        Delete a specific row.

        Parameters:
            row_index (int): The index of the row to delete.
            description (str): Description of the deletion.
        """
        with self.lock:
            region = self._find_region_by_row(row_index)
            if not region:
                return
            local_index = row_index - region['start_row']
            try:
                region['bitset'].clear_bit(local_index)
                # Update version with global version_counter
                self.version_counter += 1
                self._update_version(description or f"Delete row {row_index}", region=region)  # 传递 region 参数
            except IndexError as e:
                print(f"Delete row failed: {e}")

    def restore_row(self, row_index: int, description: Optional[str] = None):
        """
        Restore a specific row.

        Parameters:
            row_index (int): The index of the row to restore.
            description (str): Description of the restoration.
        """
        with self.lock:
            region = self._find_region_by_row(row_index)
            if not region:
                return
            local_index = row_index - region['start_row']
            try:
                region['bitset'].set_bit(local_index)
                # Update version with global version_counter
                self.version_counter += 1
                self._update_version(description or f"Restore row {row_index}", region=region)  # 传递 region 参数
            except IndexError as e:
                print(f"Restore row failed: {e}")

    def bulk_delete_rows(self, rowids: List[int], description: Optional[str] = None):
        """
        Delete multiple rows by rowids.

        Parameters:
            rowids (list): The rowids of the rows to delete.
            description (str): The description of the deletion.
        """
        with self.lock:
            regions_modified = set()
            for rowid in rowids:
                region = self._find_region_by_row(rowid)
                if region:
                    local_index = rowid - region['start_row']
                    try:
                        region['bitset'].clear_bit(local_index)
                        regions_modified.add(region['region_id'])
                    except IndexError as e:
                        print(f"Failed to delete row {rowid}: {e}")

            # Update version information
            for region_id in regions_modified:
                region = next(r for r in self.regions if r['region_id'] == region_id)
                self.version_counter += 1
                self._update_version(description or "Delete rows", region=region)
            self._update_table()

    def bulk_restore_rows(self, rowids: List[int], description: Optional[str] = None):
        """
        Restore multiple rows by rowids.

        Parameters:
            rowids (list): The rowids of the rows to restore.
            description (str): The description of the restoration.
        """
        with self.lock:
            regions_modified = set()
            for rowid in rowids:
                region = self._find_region_by_row(rowid)
                if region:
                    local_index = rowid - region['start_row']
                    try:
                        region['bitset'].set_bit(local_index)
                        regions_modified.add(region['region_id'])
                    except IndexError as e:
                        print(f"Failed to restore row {rowid}: {e}")

            # Update version information
            for region_id in regions_modified:
                region = next(r for r in self.regions if r['region_id'] == region_id)
                self.version_counter += 1
                self._update_version(description or "Restore rows", region=region)
            self._update_table()

    def _find_region_by_row(self, row_index: int) -> Optional[Dict[str, Any]]:
        """
        Find the region containing the given row index.

        Parameters:
            row_index (int): The global row index.

        Returns:
            Dict[str, Any] | None: The region information or None.
        """
        for region in self.regions:
            if region['start_row'] <= row_index < region['start_row'] + region['row_count']:
                return region
        return None

    def _update_version(self, description: Optional[str] = None, region: Optional[Dict[str, Any]] = None):
        if region is None:
            region = self.current_region

        timestamp = int(time.time())
        bitset_filename = f"region_{region['region_id']}_version_{self.version_counter}.bits"
        bitset_file = (
                self._lifecycle_path.relative_to(self.data_path) / bitset_filename
        ).as_posix()
        new_version = {
            "version": self.version_counter,
            "modify_time": timestamp,
            "description": description or "Data updated",
            "region_id": region['region_id'],
            "bitset_file": bitset_file
        }
        region['versions'].append(new_version)
        bitset_path = self._lifecycle_path / bitset_filename

        region['bitset'].save_to_file(bitset_path)
        self._save_regions_meta()

    def rollback(self, target_version: int):
        """
        Roll back to a specified version without deleting versions beyond the target version.

        Parameters:
            target_version (int): The target version number to roll back to.
        """
        with self.lock:
            # Reload regions from regions_meta
            self.regions_meta = msgpack.load(open(self.region_meta_file, 'rb'))
            serializable_regions = self.regions_meta.get('regions', [])

            # Rebuild regions list and initialize bitset
            self.regions = []
            current_start_row = 0

            for serializable_region in serializable_regions:
                region = serializable_region.copy()
                region['bitset'] = None  # Will be set later
                self.regions.append(region)

            for region in self.regions:
                # Find all versions <= target_version
                valid_versions = [v for v in region['versions'] if v['version'] <= target_version]
                if valid_versions:
                    # Load the corresponding BitSet
                    target_version_info = valid_versions[-1]
                    bitset_file = target_version_info.get('bitset_file')
                    if bitset_file and (self.data_path / bitset_file).exists():
                        bitset_path = self.data_path / bitset_file
                        region['bitset'] = BitSet.load_from_file(bitset_path)
                    else:
                        # If no bitset file, initialize BitSet with all ones
                        region['bitset'] = BitSet(size=region['row_count'], fill=1)
                    # Update versions
                    region['versions'] = valid_versions
                    # Update start_row
                    region['start_row'] = current_start_row
                    current_start_row += region['row_count']
                else:
                    # If no valid versions, set BitSet to all zeros
                    region['bitset'] = BitSet(size=region['row_count'], fill=0)
                    region['versions'] = []
                    # Update start_row
                    region['start_row'] = current_start_row
                    current_start_row += region['row_count']

            # Update version_counter
            self.version_counter = target_version

            # Reload all regions
            self._load_all_regions()

    def restore_to_latest(self):
        """
        Restore to the latest version.
        """
        with self.lock:
            # Reload regions from regions_meta
            self.regions_meta = msgpack.load(open(self.region_meta_file, 'rb'))
            serializable_regions = self.regions_meta.get('regions', [])

            # Rebuild regions list and initialize bitset
            self.regions = []
            current_start_row = 0

            for serializable_region in serializable_regions:
                region = serializable_region.copy()
                region['bitset'] = None  # Will be set later
                self.regions.append(region)

            if not self.regions:
                return

            # Find the highest version number
            latest_version = max(
                [v['version'] for region in self.regions for v in region['versions']], default=0
            )

            # Restore each region to its latest version
            for region in self.regions:
                if region['versions']:
                    latest_version_info = region['versions'][-1]
                    bitset_file = latest_version_info.get('bitset_file')
                    if bitset_file and (self.data_path / bitset_file).exists():
                        bitset_path = self.data_path / bitset_file
                        region['bitset'] = BitSet.load_from_file(bitset_path)
                    else:
                        region['bitset'] = BitSet(size=region['row_count'], fill=1)
                    # Update start_row
                    region['start_row'] = current_start_row
                    current_start_row += region['row_count']
                else:
                    region['bitset'] = BitSet(size=region['row_count'], fill=1)
                    # Update start_row
                    region['start_row'] = current_start_row
                    current_start_row += region['row_count']

            self.version_counter = latest_version
            self._load_all_regions()

    def list_versions(self) -> List[Dict[str, Any]]:
        """
        List all versions across all regions.

        Returns:
            list: The list of versions.
        """
        with self.lock:
            all_versions = []
            for region in self.regions:
                for version in region['versions']:
                    all_versions.append(version)
            # Sort versions by version number
            all_versions.sort(key=lambda x: x['version'])
            return all_versions

    def count_rows(self) -> int:
        """
        Count the number of active rows in the storage.

        Returns:
            int: The number of active rows.
        """
        with self.lock:
            active_count = 0
            for region in self.regions:
                region_active_count = region['bitset'].count()
                active_count += region_active_count
            return active_count

    def _update_table(self):
        """
        Update the DuckDB table registration.
        """
        self._register_table()

    def query(self, query_vector: Optional[List[float]] = None, columns: Optional[List[str]] = None) -> QueryBuilder:
        """
        Create a QueryBuilder object for querying the storage.

        Parameters:
            query_vector (list of float, optional): The query vector for similarity search.
            columns (list of str], optional): The columns to select.

        Returns:
            QueryBuilder: The QueryBuilder object.
        """
        return QueryBuilder(self, query_vector=query_vector, columns=columns)

    def search(self, columns: Optional[List[str]] = None) -> QueryBuilder:
        """
        Search the data from the storage.

        Parameters:
            columns (list): The columns to search.

        Returns:
            SearchIterator: The search iterator.
        """
        return QueryBuilder(self, columns)

    @lru_cache(maxsize=128)
    def cached_execute_sql(self, query: str) -> pd.DataFrame:
        """
        Execute a SQL query and return the result as a pandas DataFrame.

        Parameters:
            query (str): The SQL query to execute.

        Returns:
            pandas.DataFrame: The result of the SQL query.
        """
        return self.execute_sql(query)

    def execute_sql(self, query: str) -> pd.DataFrame:
        """
        Execute a SQL query and return the result as a pandas DataFrame.

        Parameters:
            query (str): The SQL query to execute.

        Returns:
            pandas.DataFrame: The result of the SQL query.
        """
        try:
            # Use the thread pool to execute the query in parallel
            future = self._executor.submit(self._duckdb_conn.execute, query)
            result = future.result().fetchdf()
            return result
        except Exception as e:
            print(f"SQL execution failed, error: {e}")
            return pd.DataFrame()

    def to_arrow(self, include_vector: bool = True, columns: Optional[List[str]] = None) -> pa.Table:
        """
        Convert the storage to a pyarrow Table.

        Parameters:
            include_vector (bool): Whether to include the vector column.
            columns (list): The columns to include.

        Returns:
            pyarrow.Table: The pyarrow Table.
        """
        with self.lock:
            if self.table is None:
                self._load_all_regions()

            table = self.table
            if columns is not None:
                if include_vector and 'vector' not in columns:
                    columns = ['vector'] + columns
                table = table.select(columns)
            else:
                table: pa.Table = table.drop('__rowid')

            return table

    def to_pandas(self, include_vector: bool = True, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Convert the storage to a pandas DataFrame.

        Parameters:
            include_vector (bool): Whether to include the vector column.
            columns (list): The columns to include.

        Returns:
            pandas.DataFrame: The pandas DataFrame.
        """
        with self.lock:
            table = self.to_arrow(include_vector, columns)
            df = table.to_pandas()
            return df

    def delete(self):
        """
        Delete the storage.
        """
        if self.data_path.exists():
            shutil.rmtree(self.data_path)

    def _execute_query(self, query: str):
        """
        Execute a query on the entire storage.

        Parameters:
            query (str): The condition part of the SQL WHERE clause.

        Returns:
            pd.DataFrame: The query result.
        """
        sql_query = f"SELECT * FROM 'storage' WHERE {query}"
        try:
            result = self.cached_execute_sql(sql_query)
            return result
        except Exception as e:
            print(f"Failed to execute query: {e}")
            return pd.DataFrame()
