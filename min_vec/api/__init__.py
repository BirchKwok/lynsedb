from spinesUtils.logging import Logger

from min_vec.configs.config import config


logger = Logger(
    fp=config.MVDB_LOG_PATH,
    name='MinVectorDB',
    truncate_file=config.MVDB_TRUNCATE_LOG,
    with_time=config.MVDB_LOG_WITH_TIME,
    level=config.MVDB_LOG_LEVEL
)
