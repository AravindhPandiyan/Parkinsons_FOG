"""
Custom Logging Configuration
---------------------------

This module defines a custom logging configuration with a `MaxSizeRotatingFileHandler` class that facilitates log rotation based on file size. The configuration establishes loggers and handlers to manage different log levels and file rotation.

Classes:

- `MaxSizeRotatingFileHandler`: A custom file handler class that supports log rotation based on the size of log files.

    Methods:

    - `__init__(self, filename: str, maxBytes: int, delay: int = 0)`: Initializes the handler.

    - `shouldRollover(self, _)`: Checks if the log file should be rotated based on size.

    - `doRollover(self)`: Performs log file rotation by closing the current stream and opening a new file.

Logger Configuration:

- `logger`: The main logger instance with the name of the current module.

- Logging level is set to `DEBUG`.

Log Formatting:

- Log format includes timestamp, log level, thread name, source filename, function name, and message.

Log Rotation:

- Log files are rotated when their size exceeds the specified `max_log_size`.

- The rotated files are stored in the "logs" directory.

- Maximum log file size is set to 1 MB.

"""
import json
import logging
import os
import time

with open("config/logging.json") as file:
    config = json.load(file)


class MaxSizeRotatingFileHandler(logging.FileHandler):
    """
    MaxSizeRotatingFileHandler is a custom file handler that rotates log files based on their size.
    """

    def __init__(self, filename: str, maxBytes: int, delay: int = 0):
        """
        Initializes the MaxSizeRotatingFileHandler.

        Params:
            `filename`: The path to the log file.

            `maxBytes`: The maximum size of each log file in bytes.

            `delay`: A delay in seconds for file opening.
        """
        super().__init__(filename, "a", delay=delay)
        self.maxBytes = maxBytes
        self.current_size = os.path.getsize(filename) if os.path.exists(filename) else 0
        self.creation_time = time.strftime(config["dt_format"], time.localtime())

    def shouldRollover(self, _):
        """
        shouldRollover checks if the log file should be rotated based on size.

        Params:
            `_`: Placeholder for the log record (not used).

        Returns:
            True if the log file should be rotated, False otherwise.
        """
        return self.current_size >= self.maxBytes

    def doRollover(self):
        """
        doRollover performs log file rotation by closing the current stream and opening a new file.
        """
        self.stream.close()
        self.stream = None
        self.baseFilename = self.creation_time
        self.stream = self._open()


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

log_format = config["l_format"]
formatter = logging.Formatter(log_format)

log_path = config["path"]
max_log_size = config["bytes"]
log_file = os.path.join(log_path, "log.log")
file_handler = MaxSizeRotatingFileHandler(log_file, maxBytes=max_log_size)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
