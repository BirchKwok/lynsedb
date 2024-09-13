import operator
from typing import List

from . import ExpressionParser
from ...core_components.fields_cache.filter import Filter, MatchField, FieldCondition


class FieldsQuery:
    def __init__(self, storage):
        self.storage = storage

    def filter_non_id_conditions(self, condition_list):
        return [condition for condition in condition_list
                if condition.matcher.name != "MatchID" and condition.key != ":id:"]

    def filter_range_conditions(self, condition_list):
        return [condition for condition in condition_list
                if condition.matcher.name == "MatchRange" or
                (condition.matcher.name == "MatchField" and condition.matcher.comparator in [operator.le, operator.lt,
                                                                                             operator.ge, operator.gt])]

    def filter_equal_conditions(self, condition_list):
        return [condition for condition in condition_list
                if
                condition.matcher.name == "MatchField" and condition.matcher.comparator in [operator.eq, operator.ne]]

    def retrieve_all_ids(self):
        return self.storage.retrieve_all_ids()

    def is_filter_empty(self, filter_instance):
        return not filter_instance.must_fields and \
            not filter_instance.must_not_fields and \
            not filter_instance.any_fields

    def _process_conditions(self, range_conditions=None, equal_conditions=None, is_any_fields=False):
        """
        Process the conditions.
        """
        if not range_conditions and not equal_conditions:
            return set()

        match_ids = set()
        non_indexed_conditions = []

        # handle range conditions
        for condition in range_conditions:
            if condition.matcher.name == 'MatchRange':
                start = condition.matcher.start
                end = condition.matcher.end
                inclusive = condition.matcher.inclusive

                end = end if inclusive != "left" and inclusive is not False else end - 1
                start = start if inclusive != "right" and inclusive is not False else start + 1

                if condition.key in self.storage.index.indices:
                    match_ids |= self.storage.index.range_search(condition.key, start, end)
                else:
                    non_indexed_conditions.append(condition)
            elif condition.matcher.name == 'MatchField':
                comparator = getattr(condition.matcher, 'comparator', None)
                if comparator is None:
                    raise ValueError("The comparator must be provided for MatchField conditions")

                start_value = None
                end_value = None
                if comparator == operator.lt:
                    end_value = condition.matcher.value - 1
                elif comparator == operator.le:
                    end_value = condition.matcher.value
                elif comparator == operator.gt:
                    start_value = condition.matcher.value + 1
                else:
                    start_value = condition.matcher.value

                if condition.key in self.storage.index.indices:
                    match_ids |= self.storage.index.range_search(condition.key, start_value, end_value)
                else:
                    non_indexed_conditions.append(condition)

        if non_indexed_conditions:
            match_ids |= self._match_filter(non_indexed_conditions, is_any_fields)

        non_indexed_conditions = []
        # handle equal conditions
        for condition in equal_conditions:
            if condition.matcher.name == 'MatchField':
                comparator = getattr(condition.matcher, 'comparator', None)
                if comparator is None:
                    raise ValueError("The comparator must be provided for MatchField conditions")

                if condition.key in self.storage.index.indices:
                    match_ids |= self.storage.index.search(condition.key, condition.matcher.value)
                else:
                    non_indexed_conditions.append(condition)

        if non_indexed_conditions:
            match_ids |= self._match_filter(non_indexed_conditions, is_any_fields)

        return match_ids

    def _match_filter(self, conditions, is_any_fields=False):
        """
        Match the filter.
        """
        match_ids = set()
        compare_op = all if not is_any_fields else any
        for external_id, data in self.storage.retrieve_all():
            if compare_op(condition.evaluate(data, external_id) for condition in conditions):
                match_ids.add(external_id)

        return match_ids

    def _pre_match_ids(self, condition_list, is_any_fields=False):
        """
        Pre-match the IDs.
        """
        pre_match_ids = set()
        for condition in condition_list:
            if condition.matcher.name == "MatchID":
                if pre_match_ids:
                    pre_match_ids = pre_match_ids & set(condition.matcher.indices) \
                        if not is_any_fields else pre_match_ids | set(condition.matcher.indices)
                else:
                    pre_match_ids = set(condition.matcher.indices)
            elif condition.key == ":id:":
                if pre_match_ids:
                    pre_match_ids = pre_match_ids & set(self.storage.id_filter.filter_ids(condition)) \
                        if not is_any_fields else pre_match_ids | set(self.storage.id_filter.filter_ids(condition))
                else:
                    pre_match_ids = set(self.storage.id_filter.filter_ids(condition))
        return pre_match_ids

    def _query(self, filter_instance, filter_ids=None, return_ids_only=True):
        """
        Query the fields cache using normal method.
        """
        if self.is_filter_empty(filter_instance):
            match_ids = self.storage.id_filter.retrieve_all_ids(return_set=True)
        else:
            match_ids = set()
            not_match_ids = set()
            any_match_ids = set()

            must_fields = filter_instance.must_fields
            must_not_fields = filter_instance.must_not_fields
            any_fields = filter_instance.any_fields

            # if has must fields, filter by match id
            if must_fields:
                match_ids.update(self._pre_match_ids(must_fields))
            else:
                match_ids = self.storage.id_filter.retrieve_all_ids(return_set=True)

            # if no must fields, filter by must_not fields
            if must_not_fields:
                not_match_ids.update(self._pre_match_ids(must_not_fields))

            if any_fields:
                any_match_ids.update(self._pre_match_ids(any_fields, is_any_fields=True))

            must_fields = self.filter_non_id_conditions(must_fields)
            must_not_fields = self.filter_non_id_conditions(must_not_fields)
            any_fields = self.filter_non_id_conditions(any_fields)

            if must_fields:
                # process the remaining conditions
                match_ids |= self._process_conditions(
                    self.filter_range_conditions(must_fields),
                    self.filter_equal_conditions(must_fields)
                )
            if must_not_fields:
                not_match_ids |= self._process_conditions(
                    self.filter_range_conditions(must_not_fields),
                    self.filter_equal_conditions(must_not_fields)
                )
            if any_fields:
                any_match_ids |= self._process_conditions(
                    self.filter_range_conditions(any_fields),
                    self.filter_equal_conditions(any_fields),
                    is_any_fields=True
                )

            if not_match_ids:
                match_ids -= not_match_ids

            if any_match_ids:
                # match_ids is never empty
                match_ids &= any_match_ids

            if not match_ids:
                return []

        if return_ids_only:
            return list(match_ids) if not filter_ids else list(match_ids - set(filter_ids))
        else:
            return self.retrieve_ids(list(match_ids), include_external_id=True)

    def query(self, query_filter, filter_ids=None, return_ids_only=True):
        """
        Query the fields cache.

        Parameters:
            query_filter: Filter or dict or FieldExpression (string)
                The filter object or the specify data to filter.
            filter_ids: List[int]
                The list of external IDs to filter.
            return_ids_only: bool
                If True, only the external IDs will be returned.

        Returns:
            List[dict]: The records. If not return_ids_only, the records will be returned.
            List[int]: The external IDs. If return_ids_only, the external IDs will be returned.
        """
        if isinstance(query_filter, dict):
            if ('must_fields' not in query_filter or
                    'must_not_fields' not in query_filter or
                    'any_fields' not in query_filter):
                final_filter = []
                for key, value in query_filter.items():
                    final_filter.append(FieldCondition(key=key, matcher=MatchField(value, comparator=operator.eq)))
                    query_filter = Filter(must=final_filter)
            else:
                query_filter = Filter().load_dict(query_filter)
        elif isinstance(query_filter, str):
            query_filter = ExpressionParser(query_filter).to_filter()
        elif not isinstance(query_filter, Filter):
            raise ValueError("Invalid query filter")

        return self._query(query_filter, filter_ids, return_ids_only)

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
