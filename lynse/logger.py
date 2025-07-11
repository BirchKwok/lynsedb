import os
from ultralog import UltraLog

from .configs.config import config

# Centralized UltraLog instance for the whole project
# Instantiate UltraLog then apply unified format via set_format
logger = UltraLog(
    fp=os.environ.get('LYNSE_LOG_PATH') or config.LYNSE_LOG_PATH,
    name='LynseDB',
    truncate_file=os.environ.get('LYNSE_TRUNCATE_LOG_FILE') or config.LYNSE_TRUNCATE_LOG,
    with_time=os.environ.get('LYNSE_LOG_WITH_TIME') or config.LYNSE_LOG_WITH_TIME,
    level=os.environ.get('LYNSE_LOG_LEVEL') or config.LYNSE_LOG_LEVEL
)

# Apply concise format (no module/line)
logger.set_format("%(asctime)s | %(levelname)-8s | %(message)s")
