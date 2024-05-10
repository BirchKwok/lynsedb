import operator
from typing import Union

import numpy as np
from spinesUtils.asserts import raise_if


class MatchField:
    def __init__(self, value, comparator=operator.eq, all_comparators=False):
        """
        Initialize a MatchField instance with a specific value and comparator.
            .. versionadded:: 0.3.0

        Parameters:
            value (list or tuple or any object that implements comparison functions such as __ eq__): The value to compare the data attribute with.
            comparator: The comparator function to apply to the data attribute.
            all_comparators: Whether to apply the comparator to all values in the list or tuple.
                If True, all values in the list or tuple must satisfy the comparison condition.
                If False, at least one value in the list or tuple must satisfy the comparison condition.
                    .. versionadded:: 0.3.5
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


class FieldCondition:
    def __init__(self, key, matcher):
        """
        Initialize a FieldCondition instance with a specific key and matcher.
            .. versionadded:: 0.3.0

        Parameters:
            key: The key of the data attribute to compare.
            matcher: The MatchField instance to compare the data attribute with.
        """
        self.key = key
        self.matcher = matcher

    def evaluate(self, data):
        """
        Evaluate the condition against a given dictionary.

        Parameters:
            data: A dictionary containing the data attributes to compare.

        Returns:
            bool: True if the data attribute matches the specified value, otherwise False.
        """
        attribute_value = data.get(self.key)
        if attribute_value is not None:
            return self.matcher.match(attribute_value)
        return False


class MatchID:
    def __init__(self, ids: Union[list, np.ndarray]):
        """
        Initialize an MatchID instance.
            .. versionadded:: 0.3.0

        Parameters:
            ids (list or np.ndarray): The indices to filter the numpy array.
        """
        self.indices = ids

    def match(self, array):
        """
        Filter the numpy array according to the specified indices.

        Parameters:
            array (np.ndarray): The numpy array to filter.

        Returns:
            np.ndarray: A numpy array filtered according to the specified indices.
        """
        if isinstance(array, list):
            array = np.array(array)

        return np.isin(array, self.indices, assume_unique=True)


class IDCondition:
    def __init__(self, matcher):
        """
        Initialize an IDCondition instance.
            .. versionadded:: 0.3.0

        Parameters:
            matcher: The MatchID instance to filter the numpy array.
        """
        self.matcher = matcher

    def evaluate(self, array):
        """
        Evaluate the condition against a given numpy array.
        :param array: A numpy array that needs to be filtered.
        :return: A numpy array filtered according to the specified indices.
        """
        return self.matcher.match(array)


class Filter:
    __slots__ = ['must_fields', 'any_fields', 'must_not_fields', 'must_ids', 'any_ids', 'must_not_ids']

    def __init__(self, must=None, any=None, must_not=None):
        """
        Initialize a Filter instance with must and any conditions.
            .. versionadded:: 0.3.0

        Parameters:
            must: A list of conditions that must be satisfied.
            any: A list of conditions where at least one must be satisfied.
            must_not: A list of conditions that must not be satisfied.
                .. versionadded:: 0.3.3
        """
        self.must_fields = []
        self.any_fields = []
        self.must_not_fields = []

        self.must_ids = []
        self.any_ids = []
        self.must_not_ids = []

        if must:
            for condition in must:
                if isinstance(condition, FieldCondition):
                    self.must_fields.append(condition)
                elif isinstance(condition, IDCondition):
                    self.must_ids.append(condition)
                else:
                    raise_if(ValueError, True, "Invalid condition type.")

        if any:
            for condition in any:
                if isinstance(condition, FieldCondition):
                    self.any_fields.append(condition)
                elif isinstance(condition, IDCondition):
                    self.any_ids.append(condition)
                else:
                    raise_if(ValueError, True, "Invalid condition type.")

        if must_not:
            for condition in must_not:
                if isinstance(condition, FieldCondition):
                    self.must_not_fields.append(condition)
                elif isinstance(condition, IDCondition):
                    self.must_not_ids.append(condition)
                else:
                    raise_if(ValueError, True, "Invalid condition type.")

    def to_dict(self):
        """
        Convert the filter to a dictionary.

        Returns:
            dict: A dictionary representation of the filter.
        """
        return {
            'must_fields': [{
                'key': condition.key,
                'matcher': {
                    'value': condition.matcher.value,
                    'comparator': condition.matcher.comparator.__name__
                }
            } for condition in self.must_fields] if self.must_fields else [],
            'any_fields': [{
                'key': condition.key,
                'matcher': {
                    'value': condition.matcher.value,
                    'comparator': condition.matcher.comparator.__name__
                }
            } for condition in self.any_fields] if self.any_fields else [],
            'must_not_fields': [{
                'key': condition.key,
                'matcher': {
                    'value': condition.matcher.value,
                    'comparator': condition.matcher.comparator.__name__
                }
            } for condition in self.must_not_fields] if self.must_not_fields else [],
            'must_ids': [{
                'matcher': {
                    'ids': condition.matcher.indices.tolist() if isinstance(condition.matcher.indices, np.ndarray)
                    else condition.matcher.indices
                }
            } for condition in self.must_ids] if self.must_ids else [],
            'any_ids': [{
                'matcher': {
                    'ids': condition.matcher.indices.tolist() if isinstance(condition.matcher.indices, np.ndarray)
                    else condition.matcher.indices
                }
            } for condition in self.any_ids] if self.any_ids else [],
            'must_not_ids': [{
                'matcher': {
                    'ids': condition.matcher.indices.tolist() if isinstance(condition.matcher.indices, np.ndarray)
                    else condition.matcher.indices
                }
            } for condition in self.must_not_ids] if self.must_not_ids else []
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

        self.must_ids = []
        self.any_ids = []
        self.must_not_ids = []

        for condition in data.get('must_fields', []):
            self.must_fields.append(
                FieldCondition(
                    condition['key'],
                    MatchField(condition['matcher']['value'], getattr(operator, condition['matcher']['comparator']))
                )
            )

        for condition in data.get('any_fields', []):
            self.any_fields.append(
                FieldCondition(
                    condition['key'],
                    MatchField(condition['matcher']['value'], getattr(operator, condition['matcher']['comparator']))
                )
            )

        for condition in data.get('must_not_fields', []):
            self.must_not_fields.append(
                FieldCondition(
                    condition['key'],
                    MatchField(condition['matcher']['value'], getattr(operator, condition['matcher']['comparator']))
                )
            )

        for condition in data.get('must_ids', []):
            self.must_ids.append(IDCondition(MatchID(np.array(condition['matcher']['ids']))))
        for condition in data.get('any_ids', []):
            self.any_ids.append(IDCondition(MatchID(np.array(condition['matcher']['ids']))))
        for condition in data.get('must_not_ids', []):
            self.must_not_ids.append(IDCondition(MatchID(np.array(condition['matcher']['ids']))))

        return self
