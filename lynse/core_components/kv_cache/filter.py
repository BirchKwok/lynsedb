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

    def __init__(self, value, comparator=operator.eq, all_comparators=False):
        """
        Initialize a MatchField instance with a specific value and comparator.

        Parameters:
            value (list or tuple or any object that implements comparison functions such as __ eq__): The value to compare the data attribute with.
            comparator: The comparator function to apply to the data attribute.
            all_comparators: Whether to apply the comparator to all values in the list or tuple.
                If True, all values in the list or tuple must satisfy the comparison condition.
                If False, at least one value in the list or tuple must satisfy the comparison condition.
        """
        self.value = value
        self.comparator = comparator
        self.all_comparators = all_comparators

    def match(self, data_value):
        """
        Apply the comparator to compare the data attribute with the specified value.

        Parameters:
            data_value: The value of the data attribute to compare.

        Returns:
            bool: True if the data attribute matches the specified value, otherwise False.
        """
        if isinstance(self.value, (list, tuple)):
            if self.all_comparators:
                return all([self.comparator(data_value, value) for value in self.value])
            else:
                return any([self.comparator(data_value, value) for value in self.value])

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
                If using MatchID, this should be the special key ':match_id:'.
            matcher: The MatchField, MatchID, or MatchRange instance to compare the data attribute with.
        """
        self.matcher = matcher
        if matcher is not None:
            self.key = ':match_id:' if isinstance(matcher, MatchID) else key
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
        if self.key == ':match_id:':
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

        Parameters:
            must: A list of conditions that must be satisfied.
            any: A list of conditions where at least one must be satisfied.
            must_not: A list of conditions that must not be satisfied.
        """
        self.must_fields = []
        self.any_fields = []
        self.must_not_fields = []

        must_ids = []

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
                    self.must_not_fields.append(condition)
                else:
                    raise ValueError("Invalid condition type.")

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
                        'comparator': condition.matcher.comparator.__name__
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
                        'comparator': condition.matcher.comparator.__name__
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
                        'comparator': condition.matcher.comparator.__name__
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
                self.must_fields.append(FieldCondition(':match_id:', MatchID(condition['ids'])))
            elif 'value' in condition['matcher']:
                self.must_fields.append(FieldCondition(condition['key'],
                                                       MatchField(
                                                           condition['matcher']['value'],
                                                           getattr(operator, condition['matcher']['comparator']))))
            else:
                self.must_fields.append(FieldCondition(condition['key'],
                                                       MatchRange(
                                                           condition['matcher']['start'],
                                                           condition['matcher']['end'],
                                                           condition['matcher']['inclusive'])))

        for condition in data.get('any_fields', []):
            if 'ids' in condition:
                self.any_fields.append(FieldCondition(':match_id:', MatchID(condition['ids'])))
            elif 'value' in condition['matcher']:
                self.any_fields.append(FieldCondition(condition['key'],
                                                      MatchField(
                                                          condition['matcher']['value'],
                                                          getattr(operator, condition['matcher']['comparator']))))
            else:
                self.any_fields.append(FieldCondition(condition['key'],
                                                      MatchRange(
                                                          condition['matcher']['start'],
                                                          condition['matcher']['end'],
                                                          condition['matcher']['inclusive'])))

        for condition in data.get('must_not_fields', []):
            if 'ids' in condition:
                self.must_not_fields.append(FieldCondition(':match_id:', MatchID(condition['ids'])))
            elif 'value' in condition['matcher']:
                self.must_not_fields.append(FieldCondition(
                    condition['key'], MatchField(
                        condition['matcher']['value'],
                        getattr(operator, condition['matcher']['comparator']))))
            else:
                self.must_not_fields.append(FieldCondition(
                    condition['key'], MatchRange(
                        condition['matcher']['start'],
                        condition['matcher']['end'],
                        condition['matcher']['inclusive'])))
        return self

    def __hash__(self):
        return hash(msgpack.packb(self.to_dict()))
