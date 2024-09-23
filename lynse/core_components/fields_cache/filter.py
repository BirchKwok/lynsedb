import operator

import msgpack
from spinesUtils.asserts import raise_if


class MatchRange:
    name: str = 'MatchRange'

    def __init__(self, start, end, inclusive=True):
        """
        Initialize a MatchRange instance with a start and end value.

        Parameters:
            start (any): The start value of the range.
            end (any): The end value of the range.
            inclusive (bool or str): Whether the range is inclusive of the start and end values.
                                     Can be True, False, "left", or "right".
        """
        self.start = start
        self.end = end
        self.inclusive = inclusive

    def match(self, data_value):
        """
        Check if the data_value falls within the range.

        Parameters:
            data_value (any): The value to compare against the range.

        Returns:
            bool: True if data_value is within the range, otherwise False.
        """
        if self.inclusive is True:
            return self.start <= data_value <= self.end
        elif self.inclusive == "left":
            return self.start <= data_value < self.end
        elif self.inclusive == "right":
            return self.start < data_value <= self.end
        else:
            return self.start < data_value < self.end


class MatchField:
    name: str = 'MatchField'

    def __init__(self, value, comparator=operator.eq, all_comparators=False, not_in=False):
        """
        Initialize a MatchField instance with a specific value and comparator.

        Parameters:
            value (list or tuple or any object that implements comparison functions such as __ eq__): The value to compare the data attribute with.
            comparator: The comparator function to apply to the data attribute.
            all_comparators: Whether to apply the comparator to all values in the list or tuple.
                If True, all values in the list or tuple must satisfy the comparison condition.
                If False, at least one value in the list or tuple must satisfy the comparison condition.
            not_in: Whether to negate the comparison condition.
        """
        self.value = value
        self.comparator = comparator
        self.all_comparators = all_comparators
        self.not_in = not_in
        raise_if(ValueError, all_comparators is True and not_in is True,
                 "all_comparators and not_in cannot be True at the same time.")

    def match(self, data_value):
        """
        Apply the comparator to compare the data attribute with the specified value.

        Parameters:
            data_value: The value of the data attribute to compare.

        Returns:
            bool: True if the data attribute matches the specified value, otherwise False.
        """
        if isinstance(self.value, (list, tuple)):
            if not self.not_in:
                if self.all_comparators:
                    return all([self.comparator(data_value, value) for value in self.value])
                else:
                    return data_value in self.value
            else:
                return data_value not in self.value

        return self.comparator(data_value, self.value)


class MatchID:
    name: str = 'MatchID'

    def __init__(self, ids: list):
        """
        Initialize an MatchID instance.

        Parameters:
            ids (list): The indices to filter the numpy array.
        """
        raise_if(TypeError, not isinstance(ids, list), "Invalid ids type.")

        self.indices = ids

    def match(self, external_id):
        """
        Filter the numpy array according to the specified indices.

        Parameters:
            external_id (int): The external id to filter.

        Returns:
            bool: True if the external id is in the specified indices, otherwise False.
        """
        return external_id in self.indices


class FieldCondition:
    def __init__(self, key=None, matcher=None):
        """
        Initialize a FieldCondition instance with a specific key and matcher.

        Parameters:
            key: The key of the data attribute to compare.
                If using MatchID, this should be the special key ':id:'.
            matcher: The MatchField, MatchID, or MatchRange instance to compare the data attribute with.
        """
        self.matcher = matcher
        if matcher is not None:
            self.key = ':id:' if isinstance(matcher, MatchID) else key
        else:
            self.key = key

        raise_if(TypeError, not isinstance(self.matcher, (MatchField, MatchID, MatchRange)), "Invalid matcher type.")

    def evaluate(self, data: dict, external_id: int):
        """
        Evaluate the condition against a given dictionary.

        Parameters:
            data: A dictionary with the key specified in the condition.
            external_id: The external id to filter.

        Returns:
            bool: True if the data attribute matches the specified value, otherwise False.
        """
        if self.key == ':id:':
            attribute_value = external_id
        else:
            attribute_value = data.get(self.key)

        if attribute_value is not None:
            return self.matcher.match(attribute_value)
        return False


class Filter:
    __slots__ = ['must_fields', 'any_fields', 'must_not_fields']

    def __init__(self, must=None, any=None, must_not=None):
        """
        Initialize a Filter instance with must and any conditions.

        The filter result must satisfy both must_fields and any_fields, but not must_not_fields.

        Parameters:
            must: A list of conditions that must be satisfied.
            any: A list of conditions where at least one must be satisfied.
            must_not: A list of conditions that must not be satisfied.
        """
        self.must_fields = []
        self.any_fields = []
        self.must_not_fields = []

        if must:
            for condition in must:
                if isinstance(condition, FieldCondition):
                    self.must_fields.append(condition)
                else:
                    raise ValueError("Invalid condition type.")

        if any:
            for condition in any:
                if isinstance(condition, FieldCondition):
                    self.any_fields.append(condition)
                else:
                    raise ValueError("Invalid condition type.")

        if must_not:
            for condition in must_not:
                if isinstance(condition, FieldCondition):
                    if isinstance(getattr(condition, "matcher"), MatchField):
                        if condition.matcher.not_in:
                            condition.matcher.not_in = False  # Needs to reverse the not_in flag in must_not conditions
                            condition.matcher.all_comparators = False
                    self.must_not_fields.append(condition)
                else:
                    raise ValueError("Invalid condition type.")

        self.must_fields = self._to_unique_list(self.must_fields)
        self.any_fields = self._to_unique_list(self.any_fields)
        self.must_not_fields = self._to_unique_list(self.must_not_fields)

    def to_dict(self):
        """
        Convert the filter to a dictionary.

        Returns:
            dict: A dictionary representation of the filter.
        """
        return {
            'must_fields': [
                {
                    'ids': condition.matcher.indices
                } if isinstance(condition.matcher, MatchID) else {
                    'key': condition.key,
                    'matcher': {
                        'value': condition.matcher.value,
                        'comparator': condition.matcher.comparator.__name__,
                        'all_comparators': condition.matcher.all_comparators,
                        'not_in': condition.matcher.not_in
                    }
                } if isinstance(condition.matcher, MatchField) else {
                    'key': condition.key,
                    'matcher': {
                        'start': condition.matcher.start,
                        'end': condition.matcher.end,
                        'inclusive': condition.matcher.inclusive
                    }
                } for condition in self.must_fields] if self.must_fields else [],
            'any_fields': [
                {
                    'ids': condition.matcher.indices
                } if isinstance(condition.matcher, MatchID) else {
                    'key': condition.key,
                    'matcher': {
                        'value': condition.matcher.value,
                        'comparator': condition.matcher.comparator.__name__,
                        'all_comparators': condition.matcher.all_comparators,
                        'not_in': condition.matcher.not_in
                    }
                } if isinstance(condition.matcher, MatchField) else {
                    'key': condition.key,
                    'matcher': {
                        'start': condition.matcher.start,
                        'end': condition.matcher.end,
                        'inclusive': condition.matcher.inclusive
                    }
                } for condition in self.any_fields] if self.any_fields else [],
            'must_not_fields': [
                {
                    'ids': condition.matcher.indices
                } if isinstance(condition.matcher, MatchID) else {
                    'key': condition.key,
                    'matcher': {
                        'value': condition.matcher.value,
                        'comparator': condition.matcher.comparator.__name__,
                        'all_comparators': condition.matcher.all_comparators,
                        'not_in': condition.matcher.not_in
                    }
                } if isinstance(condition.matcher, MatchField) else {
                    'key': condition.key,
                    'matcher': {
                        'start': condition.matcher.start,
                        'end': condition.matcher.end,
                        'inclusive': condition.matcher.inclusive
                    }
                } for condition in self.must_not_fields] if self.must_not_fields else []
        }

    def load_dict(self, data):
        """
        Load the filter from a dictionary.

        Parameters:
            data: The dictionary to load the filter from.
        """
        self.must_fields = []
        self.any_fields = []
        self.must_not_fields = []

        for condition in data.get('must_fields', []):
            if 'ids' in condition:
                self.must_fields.append(FieldCondition(':id:', MatchID(condition['ids'])))
            elif 'value' in condition['matcher']:
                self.must_fields.append(FieldCondition(condition['key'],
                                                       MatchField(
                                                           condition['matcher']['value'],
                                                           getattr(operator, condition['matcher']['comparator']),
                                                           condition['matcher']['all_comparators'],
                                                           condition['matcher']['not_in']
                                                       )))
            else:
                self.must_fields.append(FieldCondition(condition['key'],
                                                       MatchRange(
                                                           condition['matcher']['start'],
                                                           condition['matcher']['end'],
                                                           condition['matcher']['inclusive'])))

        for condition in data.get('any_fields', []):
            if 'ids' in condition:
                self.any_fields.append(FieldCondition(':id:', MatchID(condition['ids'])))
            elif 'value' in condition['matcher']:
                self.any_fields.append(FieldCondition(condition['key'],
                                                      MatchField(
                                                          condition['matcher']['value'],
                                                          getattr(operator, condition['matcher']['comparator']),
                                                          condition['matcher']['all_comparators'],
                                                          condition['matcher']['not_in']
                                                      )))
            else:
                self.any_fields.append(FieldCondition(condition['key'],
                                                      MatchRange(
                                                          condition['matcher']['start'],
                                                          condition['matcher']['end'],
                                                          condition['matcher']['inclusive'])))

        for condition in data.get('must_not_fields', []):
            if 'ids' in condition:
                self.must_not_fields.append(FieldCondition(':id:', MatchID(condition['ids'])))
            elif 'value' in condition['matcher']:
                self.must_not_fields.append(FieldCondition(
                    condition['key'], MatchField(
                        condition['matcher']['value'],
                        getattr(operator, condition['matcher']['comparator']),
                        condition['matcher']['all_comparators'],
                        condition['matcher']['not_in']
                    )))
            else:
                self.must_not_fields.append(FieldCondition(
                    condition['key'], MatchRange(
                        condition['matcher']['start'],
                        condition['matcher']['end'],
                        condition['matcher']['inclusive'])))

        self.must_fields = self._to_unique_list(self.must_fields)
        self.any_fields = self._to_unique_list(self.any_fields)
        self.must_not_fields = self._to_unique_list(self.must_not_fields)

        return self

    @staticmethod
    def _to_unique_list(conditions):
        """
        Remove duplicate conditions from the list.

        Parameters:
            conditions: The list of conditions to remove duplicates from.

        Returns:
            list: A list of conditions with duplicates removed.
        """
        unique_conditions = []
        for condition in conditions:
            if not any(Filter._conditions_equal(condition, unique_cond) for unique_cond in unique_conditions):
                unique_conditions.append(condition)
        return unique_conditions

    @staticmethod
    def _conditions_equal(cond1, cond2):
        """
        Compare two FieldCondition objects for equality.

        Parameters:
            cond1: The first FieldCondition object.
            cond2: The second FieldCondition object.

        Returns:
            bool: True if the conditions are equal, False otherwise.
        """
        if not isinstance(cond1, FieldCondition) or not isinstance(cond2, FieldCondition):
            return False
        if cond1.key != cond2.key:
            return False
        if type(cond1.matcher) != type(cond2.matcher):
            return False
        if isinstance(cond1.matcher, (MatchField, MatchID, MatchRange)):
            return cond1.matcher.__dict__ == cond2.matcher.__dict__
        return cond1.matcher == cond2.matcher

    def __hash__(self):
        return hash(msgpack.packb(self.to_dict()))
