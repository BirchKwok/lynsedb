from spinesUtils.logging import Logger

from lynse.configs.config import config


logger = Logger(
    fp=config.LYNSE_LOG_PATH,
    name='LynseDB',
    truncate_file=config.LYNSE_TRUNCATE_LOG,
    with_time=config.LYNSE_LOG_WITH_TIME,
    level=config.LYNSE_LOG_LEVEL
)
