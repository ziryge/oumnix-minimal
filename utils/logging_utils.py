import logging
import os
from logging.handlers import RotatingFileHandler

_LEVELS = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
}

def get_logger(name: str = "oumnix") -> logging.Logger:
    level_name = os.environ.get("OUMNIX_LOG_LEVEL", "INFO").upper()
    level = _LEVELS.get(level_name, logging.INFO)
    if not logging.getLogger().handlers:
        logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logfile = os.environ.get("OUMNIX_LOG_FILE")
    if logfile and not any(isinstance(h, RotatingFileHandler) for h in logger.handlers):
        try:
            fh = RotatingFileHandler(logfile, maxBytes=5*1024*1024, backupCount=3)
            fh.setLevel(level)
            fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
            logger.addHandler(fh)
        except Exception:
            pass
    return logger
