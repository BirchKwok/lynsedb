import os
from spinesUtils.logging import Logger

from ..configs.config import config

# Create a logger for the LynseDB
logger = Logger(
    fp=os.environ.get('LYNSE_LOG_PATH') or config.LYNSE_LOG_PATH,
    name='LynseDB',
    truncate_file=os.environ.get('LYNSE_TRUNCATE_LOG_FILE') or config.LYNSE_TRUNCATE_LOG,
    with_time=os.environ.get('LYNSE_LOG_WITH_TIME') or config.LYNSE_LOG_WITH_TIME,
    level=os.environ.get('LYNSE_LOG_LEVEL') or config.LYNSE_LOG_LEVEL
)
