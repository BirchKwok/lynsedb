import operator
from typing import Union

import numpy as np


class MatchField:
    def __init__(self, value, comparator=operator.eq):
        """
        Initialize a MatchField instance with a specific value and comparator.
            .. versionadded:: 0.3.0

        Parameters:
            value: The value to compare the data attribute with.
            comparator: The comparator function to apply to the data attribute.
        """
        self.value = value
        self.comparator = comparator

    def match(self, data_value):
        """
        Apply the comparator to compare the data attribute with the specified value.

        Parameters:
            data_value: The value of the data attribute to compare.

        Returns:
            bool: True if the data attribute matches the specified value, otherwise False.
        """
        return self.comparator(data_value, self.value)

    def __hash__(self):
        return hash(self.value) + hash(self.comparator)


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

    def __hash__(self):
        return hash(self.key) + hash(self.matcher)


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
        return array[np.isin(array, self.indices, assume_unique=True)]

    def __hash__(self):
        if isinstance(self.indices, list):
            return hash(tuple(self.indices))
        else:
            return hash(self.indices.tobytes())


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

    def __hash__(self):
        return hash(self.matcher)


class Filter:
    __slots__ = ['must', 'any']

    def __init__(self, must=None, any=None):
        """
        Initialize a Filter instance with must and any conditions.
            .. versionadded:: 0.3.0

        Parameters:
            must: A list of conditions that must be satisfied.
            any: A list of conditions where at least one must be satisfied.
        """
        self.must = must if must is not None else []
        self.any = any if any is not None else []

    def apply(self, data):
        """
        Apply the filter to the given data.

        Parameters:
            data: The data to apply the filter to.

        Returns:
            bool: True if the data satisfies the filter conditions, otherwise False.
        """
        if not self.must and not self.any:
            must_pass = True
            any_pass = False
        else:
            must_pass = all(condition.evaluate(data) for condition in self.must) if self.must else True
            any_pass = any(condition.evaluate(data) for condition in self.any) if self.any else False

        return must_pass and (any_pass or not self.any)

    def to_dict(self):
        """
        Convert the filter to a dictionary.

        Returns:
            dict: A dictionary representation of the filter.
        """
        return {
            'must': [{'field': condition.key, 'operator': condition.matcher.comparator.__name__, 'value': condition.matcher.value}
                     if isinstance(condition, FieldCondition) else {'ids': condition.matcher.indices}
                     for condition in self.must] if self.must else None,
            'any': [{'field': condition.key, 'operator': condition.matcher.comparator.__name__, 'value': condition.matcher.value}
                    if isinstance(condition, FieldCondition) else {'ids': condition.matcher.indices}
                    for condition in self.any] if self.any else None
        }

    def __hash__(self):
        return hash((tuple(self.must), tuple(self.any)))
