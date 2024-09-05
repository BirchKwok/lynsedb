from abc import ABC, abstractmethod

from spinesUtils.asserts import raise_if

from ..core_components.fast_sort import FastSort


class BaseIndex(ABC):
    """Defines the interface for an index."""

    def __init__(self):
        self.distance = None

    def _register_distance(self, distance):
        """Registers the distance function."""
        self.distance = distance

    @abstractmethod
    def encode(self, *args, **kwargs):
        """Encodes the data."""
        pass

    @abstractmethod
    def fit_transform(self, *args, **kwargs):
        """Fits the model and transforms the data."""
        pass

    def check_and_encode(self, original_vec=None, encoded_vec=None, original_data=None,
                         encoded_data=None):
        """Searches for nearest neighbors."""
        raise_if(ValueError, original_data is None and encoded_data is None,
                 "Either original_data or encoded_data must be provided.")
        raise_if(ValueError, original_vec is None and encoded_vec is None,
                 "Either original_vec or encoded_vec must be provided.")

        if encoded_data is None:
            encoded_data = self.encode(original_data)
        if encoded_vec is None:
            encoded_vec = self.encode(original_vec)

        return encoded_vec, encoded_data

    @abstractmethod
    def save(self, filepath):
        """Saves the model to a file."""
        pass

    @abstractmethod
    def load(self, filepath):
        """Loads the model from a file."""
        pass
