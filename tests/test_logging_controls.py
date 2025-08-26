import os
from utils.logging_utils import get_logger

def test_logging_level_env():
    os.environ["OUMNIX_LOG_LEVEL"] = "DEBUG"
    logger = get_logger("test")
    assert logger.isEnabledFor(10)  # DEBUG
    os.environ["OUMNIX_LOG_LEVEL"] = "INFO"
    logger = get_logger("test")
    assert logger.isEnabledFor(20)  # INFO
