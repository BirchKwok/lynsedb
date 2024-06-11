from spinesUtils.logging import Logger

from cvg.configs.config import config


logger = Logger(
    fp=config.CVG_LOG_PATH,
    name='Convergence',
    truncate_file=config.CVG_TRUNCATE_LOG,
    with_time=config.CVG_LOG_WITH_TIME,
    level=config.CVG_LOG_LEVEL
)
