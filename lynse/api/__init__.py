import os
from ..logger import logger

from ..configs.config import config

# Re-export the centralized logger for backward compatibility
Logger = type(logger)
