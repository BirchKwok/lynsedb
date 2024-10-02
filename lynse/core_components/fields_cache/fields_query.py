import operator
from typing import List, Union

from ...core_components.fields_cache.filter import Filter, MatchField, FieldCondition
from ...core_components.fields_cache.expression_parse import ExpressionParser


class FieldsQuery:
    """
    The FieldsQuery class is used to query data in the fields_cache.
    """
    def __init__(self, storage):
        """
        Initialize the FieldsQuery class.

        Parameters:
            storage: Storage
                The storage object.
        """
        self.storage = storage

    def filter_non_id_conditions(self, condition_list):
        """
        Filter out non-id conditions.

        Parameters:
            condition_list: List[FieldCondition]
                The list of conditions.

        Returns:
            List[FieldCondition]: The filtered list of conditions.
        """
        return [condition for condition in condition_list
                if condition.matcher.name != "MatchID" and condition.key != ":id:"]

    def filter_range_conditions(self, condition_list):
        """
        Filter out range conditions.

        Parameters:
            condition_list: List[FieldCondition]
                The list of conditions.

        Returns:
            List[FieldCondition]: The filtered list of conditions.
        """
        return [condition for condition in condition_list
                if condition.matcher.name == "MatchRange" or
                (condition.matcher.name == "MatchField" and condition.matcher.comparator in [operator.le, operator.lt,
                                                                                             operator.ge, operator.gt])]

    def filter_equal_conditions(self, condition_list):
        """
        Filter out equal conditions.

        Parameters:
            condition_list: List[FieldCondition]
                The list of conditions.

        Returns:
            List[FieldCondition]: The filtered list of conditions.
        """
        return [condition for condition in condition_list
                if
                condition.matcher.name == "MatchField" and condition.matcher.comparator in [operator.eq, operator.ne]]

    def retrieve_all_ids(self):
        """
        Retrieve all IDs from the storage.

        Returns:
            List[int]: The list of IDs.
        """
        return [row[0] for row in self.storage.cursor.execute("SELECT external_id FROM records")]

    def is_filter_empty(self, filter_instance):
        """
        Check if the filter is empty.

        Parameters:
            filter_instance: Filter
                The filter object.

        Returns:
            bool: True if the filter is empty, False otherwise.
        """
        return not filter_instance.must_fields and \
            not filter_instance.must_not_fields and \
            not filter_instance.any_fields

    def _process_conditions(self, range_conditions=None, equal_conditions=None, is_any_fields=False):
        """
        Process the conditions.

        Parameters:
            range_conditions: List[FieldCondition]
                The list of range conditions.
            equal_conditions: List[FieldCondition]
                The list of equal conditions.
            is_any_fields: bool
                If True, process any fields.

        Returns:
            set[int]: The set of IDs.
        """
        if not range_conditions and not equal_conditions:
            return set()

        match_ids = set()

        # handle range conditions
        for condition in range_conditions:
            if condition.matcher.name == 'MatchRange':
                start = condition.matcher.start
                end = condition.matcher.end
                inclusive = condition.matcher.inclusive

                end = end if inclusive != "left" and inclusive is not False else end - 1
                start = start if inclusive != "right" and inclusive is not False else start + 1

                match_ids |= set(self.storage.range_search(condition.key, start, end))
            elif condition.matcher.name == 'MatchField':
                comparator = getattr(condition.matcher, 'comparator', None)
                if comparator is None:
                    raise ValueError("MatchField condition must provide comparator")

                values = condition.matcher.value
                if not isinstance(values, list):
                    values = [values]

                if condition.matcher.all_comparators:
                    # all conditions must be satisfied, take intersection
                    temp_ids = None
                    for value in values:
                        start_value = None
                        end_value = None
                        if comparator == operator.lt:
                            end_value = value - 1
                        elif comparator == operator.le:
                            end_value = value
                        elif comparator == operator.gt:
                            start_value = value + 1
                        else:
                            start_value = value

                        current_ids = set(self.storage.range_search(condition.key, start_value, end_value))
                        if temp_ids is None:
                            temp_ids = current_ids
                        else:
                            temp_ids &= current_ids  # take intersection

                    if temp_ids:
                        match_ids |= temp_ids
                else:
                    # any condition satisfied, take union
                    for value in values:
                        start_value = None
                        end_value = None
                        if comparator == operator.lt:
                            end_value = value - 1
                        elif comparator == operator.le:
                            end_value = value
                        elif comparator == operator.gt:
                            start_value = value + 1
                        else:
                            start_value = value

                        match_ids |= set(self.storage.range_search(condition.key, start_value, end_value))

        # handle equal conditions
        for condition in equal_conditions:
            if condition.matcher.name == 'MatchField':
                comparator = getattr(condition.matcher, 'comparator', None)
                if comparator is None:
                    raise ValueError("MatchField condition must provide comparator")

                values = condition.matcher.value
                if not isinstance(values, list):
                    values = [values]

                if condition.matcher.all_comparators:
                    temp_ids = None
                    for value in values:
                        current_ids = set(self.storage.search(condition.key, value))
                        if temp_ids is None:
                            temp_ids = current_ids
                        else:
                            temp_ids &= current_ids  # take intersection

                    if temp_ids:
                        match_ids |= temp_ids
                else:
                    for value in values:
                        match_ids |= set(self.storage.search(condition.key, value))

        return match_ids

    def _match_filter(self, conditions, is_any_fields=False):
        """
        Match the filter.

        Parameters:
            conditions: List[FieldCondition]
                The list of conditions.
            is_any_fields: bool
                If True, match any fields.

        Returns:
            set[int]: The set of IDs.
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

        Parameters:
            condition_list: List[FieldCondition]
                The list of conditions.
            is_any_fields: bool
                If True, match any fields.

        Returns:
            set[int]: The set of IDs.
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
                    pre_match_ids = pre_match_ids & set(self.storage.search(":id:", condition.matcher.value)) \
                        if not is_any_fields else pre_match_ids | set(self.storage.search(":id:", condition.matcher.value))
                else:
                    pre_match_ids = set(self.storage.search(":id:", condition.matcher.value))
        return pre_match_ids

    def _query(self, filter_instance, filter_ids=None, return_ids_only=True):
        """
        Query the fields cache.

        Parameters:
            filter_instance: Filter
                The filter object.
            filter_ids: List[int]
                List of external IDs to filter.
            return_ids_only: bool
                If True, only return external IDs.

        Returns:
            List[dict]: Records. If not return_ids_only, returns records.
            List[int]: External IDs. If return_ids_only, returns external IDs.
        """
        if self.is_filter_empty(filter_instance):
            match_ids = set(self.retrieve_all_ids())
        else:
            match_ids = set()
            not_match_ids = set()
            any_match_ids = set()

            must_fields = filter_instance.must_fields
            must_not_fields = filter_instance.must_not_fields
            any_fields = filter_instance.any_fields

            # handle must_fields and must_not_fields
            if must_fields:
                match_ids.update(self._pre_match_ids(must_fields))
            else:
                match_ids = set(self.retrieve_all_ids())

            if must_not_fields:
                not_match_ids.update(self._pre_match_ids(must_not_fields))

            # filter must_fields and must_not_fields
            must_fields = self.filter_non_id_conditions(must_fields)
            must_not_fields = self.filter_non_id_conditions(must_not_fields)

            if must_fields:
                match_ids |= self._process_conditions(
                    self.filter_range_conditions(must_fields),
                    self.filter_equal_conditions(must_fields)
                )
            if must_not_fields:
                not_match_ids |= self._process_conditions(
                    self.filter_range_conditions(must_not_fields),
                    self.filter_equal_conditions(must_not_fields)
                )

            match_ids -= not_match_ids  # apply must_not_fields filter

            # handle any_fields
            if any_fields:
                any_match_ids.update(self._pre_match_ids(any_fields, is_any_fields=True))
                any_fields = self.filter_non_id_conditions(any_fields)
                any_match_ids |= self._process_conditions(
                    self.filter_range_conditions(any_fields),
                    self.filter_equal_conditions(any_fields),
                    is_any_fields=True
                )
                match_ids &= any_match_ids  # take intersection with any_fields

            if not match_ids:
                return []

        if return_ids_only:
            return list(match_ids) if not filter_ids else list(match_ids - set(filter_ids))
        else:
            return self.storage.retrieve_by_external_id(list(match_ids), include_external_id=True)

    def query(self, query_filter: Union[Filter, dict, str], filter_ids: List[int] = None, return_ids_only: bool = True):
        """
        Query the fields cache.

        Parameters:
            query_filter: Filter or dict or FieldExpression (string)
                Filter object or specified filter data.
            filter_ids: List[int]
                List of external IDs to filter.
            return_ids_only: bool
                If True, only return external IDs.

        Returns:
            List[dict]: Records. If not return_ids_only, returns records.
            List[int]: External IDs. If return_ids_only, returns external IDs.
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

    def retrieve(self, external_id: int, include_external_id: bool = True) -> dict:
        """
        Retrieve a record from the cache.

        Parameters:
            external_id: int
                The external ID of the record.
            include_external_id: bool
                If True, include the external ID in the record.

        Returns:
            dict: The record.
        """
        res = self.storage.retrieve_by_external_id(external_id, include_external_id=include_external_id)
        if res:
            return res[0]
        return None

    def retrieve_ids(self, external_ids: List[int], include_external_id: bool = True) -> List[dict]:
        """
        Retrieve records from the cache.

        Parameters:
            external_ids: List[int]
                List of external IDs of the records.
            include_external_id: bool
                If True, include the external ID in the records.

        Returns:
            List[dict]: List of records.
        """
        results = self.storage.retrieve_by_external_id(external_ids, include_external_id=include_external_id)
        return results if results else None
