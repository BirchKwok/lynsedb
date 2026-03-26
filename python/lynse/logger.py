import os
import logging

from .configs.config import config


def _create_logger() -> logging.Logger:
    """Create a centralized stdlib logger for the whole project."""
    name = 'LynseDB'
    _logger = logging.getLogger(name)

    # Avoid adding duplicate handlers on re-import
    if _logger.handlers:
        return _logger

    level_str = os.environ.get('LYNSE_LOG_LEVEL') or config.LYNSE_LOG_LEVEL or 'INFO'
    _logger.setLevel(getattr(logging, level_str.upper(), logging.INFO))

    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s")

    # File handler (optional)
    fp = os.environ.get('LYNSE_LOG_PATH') or config.LYNSE_LOG_PATH
    if fp:
        os.makedirs(os.path.dirname(fp) if os.path.dirname(fp) else '.', exist_ok=True)
        truncate = os.environ.get('LYNSE_TRUNCATE_LOG_FILE') or config.LYNSE_TRUNCATE_LOG
        mode = 'w' if truncate else 'a'
        fh = logging.FileHandler(fp, mode=mode, encoding='utf-8')
        fh.setFormatter(fmt)
        _logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    _logger.addHandler(ch)

    return _logger


logger = _create_logger()
