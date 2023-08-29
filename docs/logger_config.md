Module logger_config
====================
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

- Maximum log file size is set to 10 MB.

Classes
-------

`MaxSizeRotatingFileHandler(filename: str, maxBytes: int, delay: int = 0)`
:   MaxSizeRotatingFileHandler a custom file handler that rotates log files based on their size.
    
    Initializes the MaxSizeRotatingFileHandler.
    
    Args:
        filename: The path to the log file.
        maxBytes: The maximum size of each log file in bytes.
        delay: A delay in seconds for file opening.

    ### Ancestors (in MRO)

    * logging.FileHandler
    * logging.StreamHandler
    * logging.Handler
    * logging.Filterer

    ### Methods

    `doRollover(self)`
    :   doRollover performs log file rotation by closing the current stream and opening a new file.

    `shouldRollover(self, _)`
    :   shouldRollover checks if the log file should be rotated based on size.
        
        Args:
            `_`: Placeholder for the log record (not used).
        
        Returns:
            True if the log file should be rotated, False otherwise.