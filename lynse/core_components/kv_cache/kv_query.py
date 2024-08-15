import operator
from typing import List

from ...core_components.kv_cache import Filter, MatchField, FieldCondition


class KVCacheQuery:
    def __init__(self, storage):
        self.storage = storage

    def _index_query(self, filter_instance, filter_ids=None, return_ids_only=True):
        match_ids = None
        not_match_ids = set()
        non_indexed_must_conditions = []
        non_indexed_must_not_conditions = []
        non_indexed_any_conditions = []
        all_fields_indexed = True  # Flag to check if all fields are indexed

        # Shortcut for MatchID conditions
        if filter_instance.must_fields:
            for condition in filter_instance.must_fields:
                if condition.matcher.name == "MatchID":
                    match_ids = set(condition.matcher.indices)
                    break

        if filter_instance.must_not_fields:
            for condition in filter_instance.must_not_fields:
                if condition.matcher.name == "MatchID":
                    not_match_ids.update(condition.matcher.indices)
                    break

        if match_ids is not None:
            match_ids -= not_match_ids

        # Process must fields using index
        if filter_instance.must_fields:
            for condition in filter_instance.must_fields:
                if condition.key in self.storage.index.indices:
                    current_ids = self._handle_range_search(condition)
                    match_ids = current_ids if match_ids is None else match_ids & current_ids
                else:
                    non_indexed_must_conditions.append(condition)
                    all_fields_indexed = False

        # Process must_not fields using index
        if filter_instance.must_not_fields:
            for condition in filter_instance.must_not_fields:
                if condition.key in self.storage.index.indices:
                    current_not_match_ids = self._handle_range_search(condition)
                    not_match_ids |= current_not_match_ids
                else:
                    non_indexed_must_not_conditions.append(condition)
                    all_fields_indexed = False

        # Exclude must_not matches from match_ids
        if match_ids is not None:
            match_ids -= not_match_ids

        if all_fields_indexed and not match_ids:
            return []

        if not all_fields_indexed:
            if match_ids:
                match_ids = {external_id for external_id, data in zip(match_ids, self.retrieve_ids(list(match_ids)))
                             if self._index_filter(external_id, data, filter_ids, non_indexed_must_conditions,
                                                   non_indexed_must_not_conditions, non_indexed_any_conditions=None)}
            else:
                match_ids = self._scan_all(filter_ids, return_ids_only, non_indexed_must_conditions,
                                           non_indexed_must_not_conditions, non_indexed_any_conditions=None)

            all_fields_indexed = True

        # Process any fields using index and non-indexed fields
        if filter_instance.any_fields:
            any_match_ids = set()
            for condition in filter_instance.any_fields:
                if condition.key in self.storage.index.indices:
                    any_match_ids |= self._handle_range_search(condition)
                else:
                    non_indexed_any_conditions.append(condition)
                    all_fields_indexed = False

            if not all_fields_indexed:
                any_match_ids |= {external_id for external_id, data in
                                  zip(match_ids, self.retrieve_ids(list(match_ids)))
                                  if self._index_filter(external_id, data, filter_ids, non_indexed_must_conditions=None,
                                                        non_indexed_must_not_conditions=None,
                                                        non_indexed_any_conditions=non_indexed_any_conditions)}

        else:
            any_match_ids = match_ids

        if all_fields_indexed:
            match_ids &= any_match_ids if match_ids else any_match_ids

        if not match_ids and all_fields_indexed:
            return []

        matched = []
        if match_ids:
            for external_id, data in zip(match_ids, self.retrieve_ids(list(match_ids))):
                if return_ids_only:
                    matched.append(external_id)
                else:
                    matched.append(data)
            return matched

        return self._normal_query(filter_instance, filter_ids, return_ids_only)

    def _handle_range_search(self, condition):
        if condition.matcher.name == 'MatchRange':
            start = condition.matcher.start
            end = condition.matcher.end
            inclusive = condition.matcher.inclusive
            if inclusive is True:
                return self.storage.index.range_search(condition.key, start, end)
            elif inclusive == "left":
                all_ids = self.storage.index.range_search(condition.key, start, end)
                end_ids = self.storage.index.search(condition.key, end)
                return all_ids - end_ids
            elif inclusive == "right":
                all_ids = self.storage.index.range_search(condition.key, start, end)
                start_ids = self.storage.index.search(condition.key, start)
                return all_ids - start_ids
            else:
                return self.storage.index.range_search(condition.key, start, end)
        elif condition.matcher.name == 'MatchField':
            comparator = getattr(condition.matcher, 'comparator', None)
            if comparator in [operator.le, operator.lt]:
                return self.storage.index.range_search(condition.key, None, condition.matcher.value)
            elif comparator in [operator.ge, operator.gt]:
                return self.storage.index.range_search(condition.key, condition.matcher.value, None)

    @staticmethod
    def _index_filter(external_id, data, filter_ids, non_indexed_must_conditions,
                      non_indexed_must_not_conditions, non_indexed_any_conditions):
        if non_indexed_must_not_conditions:
            if any(condition.evaluate(data, external_id) for condition in non_indexed_must_not_conditions):
                return False

        if non_indexed_must_conditions:
            if not all(condition.evaluate(data, external_id) for condition in non_indexed_must_conditions):
                return False

        if non_indexed_any_conditions:
            if not any(condition.evaluate(data, external_id) for condition in non_indexed_any_conditions):
                return False

        return filter_ids is None or external_id in filter_ids

    @staticmethod
    def _matches_filter(external_id, data, filter_instance, filter_ids):
        must_pass = all(condition.evaluate(data, external_id) for condition in filter_instance.must_fields
                        if condition.matcher.name != 'MatchID') if filter_instance.must_fields else True

        must_not_pass = any(condition.evaluate(data, external_id) for condition in filter_instance.must_not_fields
                            if condition.matcher.name != 'MatchID') if filter_instance.must_not_fields else False

        if not must_pass or must_not_pass:
            return False

        any_pass = any(condition.evaluate(data, external_id) for condition in filter_instance.any_fields) \
            if filter_instance.any_fields else True

        if not any_pass:
            return False

        return filter_ids is None or external_id in filter_ids

    def _scan_all(self, filter_ids, return_ids_only=True, non_indexed_must_conditions=None,
                  non_indexed_must_not_conditions=None, non_indexed_any_conditions=None):
        matched = []

        for external_id, data in self.storage.retrieve_all():
            if self._index_filter(external_id, data, filter_ids,
                                  non_indexed_must_conditions, non_indexed_must_not_conditions,
                                  non_indexed_any_conditions):
                if return_ids_only:
                    matched.append(external_id)
                else:
                    matched.append((external_id, data))

        return matched

    def _normal_query(self, filter_instance, filter_ids=None, return_ids_only=True):
        matched = []
        match_ids = set()
        not_match_ids = set()

        # if has must fields, filter by match id
        if filter_instance.must_fields:
            for condition in filter_instance.must_fields:
                if condition.matcher.name == "MatchID":
                    match_ids.update(condition.matcher.indices)

        # if no must fields, filter by must_not fields
        if filter_instance.must_not_fields:
            for condition in filter_instance.must_not_fields:
                if condition.matcher.name == "MatchID":
                    not_match_ids.update(condition.matcher.indices)

        match_ids -= not_match_ids

        if match_ids:
            for external_id, data in zip(match_ids, self.retrieve_ids(list(match_ids))):
                if self._matches_filter(external_id, data, filter_instance, filter_ids):
                    if return_ids_only:
                        matched.append(external_id)
                    else:
                        matched.append(data)
        else:
            for external_id, data in self.storage.retrieve_all():
                if self._matches_filter(external_id, data, filter_instance, filter_ids):
                    if return_ids_only:
                        matched.append(external_id)
                    else:
                        matched.append(data)

        return matched

    def query(self, filter_instance, filter_ids=None, return_ids_only=True):
        """
        Query the cache.

        Parameters:
            filter_instance: Filter or dict
                The filter object or the specify data to filter.
            filter_ids: List[int]
                The list of external IDs to filter.
            return_ids_only: bool
                If True, only the external IDs will be returned.

        Returns:
            List[dict]: The records that match the filter.
        """
        if isinstance(filter_instance, dict):
            if ('must_fields' not in filter_instance or
                    'must_not_fields' not in filter_instance or
                    'any_fields' not in filter_instance):
                final_filter = []
                for key, value in filter_instance.items():
                    final_filter.append(FieldCondition(key=key, matcher=MatchField(value, comparator=operator.eq)))
                    filter_instance = Filter(must=final_filter)
            else:
                filter_instance = Filter().load_dict(filter_instance)

        if self.storage.index.indices:
            return self._index_query(filter_instance, filter_ids, return_ids_only)
        return self._normal_query(filter_instance, filter_ids, return_ids_only)

    def retrieve(self, external_id, include_external_id=True):
        """
        Retrieve a record from the cache.

        Parameters:
            external_id: int
                The external ID of the record.
            include_external_id: bool
                If True, the external ID will be included in the record.

        Returns:
            dict: The record.
        """
        res = self.storage.retrieve_by_external_id(external_id, include_external_id=include_external_id)
        if res:
            return res[0]
        return None

    def retrieve_ids(self, external_ids: List[int], include_external_id=True):
        """
        Retrieve records from the cache.

        Parameters:
            external_ids: List[int]
                The external IDs of the records.
            include_external_id: bool
                If True, the external ID will be included in the record.

        Returns:
            List[dict]: The records.
        """
        results = self.storage.retrieve_by_external_id(external_ids, include_external_id=include_external_id)
        return results if results else None
