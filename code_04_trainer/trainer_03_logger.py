import sys
from loguru import logger


class TrainerLogger:
    def __init__(self, log_path: str| None):
        logger_format = ""
        logger_format += "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green>"
        logger_format += "<level>{message}</level>"
        logger.remove(handler_id=None)
        logger.add(sink=sys.stdout, format=logger_format, level="INFO")
        if log_path is not None:
            logger.add(sink=log_path, format=logger_format)

    @staticmethod
    def info(message: str):
        logger.info(message)

    @staticmethod
    def debug(message: str):
        logger.debug(message)

    @staticmethod
    def warning(message: str):
        logger.warning(message)

    @staticmethod
    def error(message: str):
        logger.error(message)

    @staticmethod
    def remove_all_loggers():
        logger.remove(handler_id=None)
