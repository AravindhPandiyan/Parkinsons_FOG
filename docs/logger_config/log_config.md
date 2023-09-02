Module logger_config.log_config
===============================
This script demonstrates setting up a logging configuration that outputs log messages to a file. The logs are formatted
with a custom format that includes timestamp, log level, thread name, filename, function name, and log message. The log
file names are dynamically generated based on the current date in the format 'YYYY-MM-DD.log'.

Usage:

- The script configures a logger to write log messages to a file named after the current date.

- Log messages of varying severity levels (debug, info, warning, error, and critical) are provided as examples.

Note:

- To include this logging functionality in your module or application, you can import the `logger` object and use its
  methods to log messages at different levels.

Example output:

2023-08-28 10:15:30,123 DEBUG root MainThread example.py <module> : This is a debug message.

2023-08-28 10:15:30,124 INFO root MainThread example.py <module> : This is an info message.

2023-08-28 10:15:30,124 WARNING root MainThread example.py <module> : This is a warning message.

2023-08-28 10:15:30,124 ERROR root MainThread example.py <module> : This is an error message.

2023-08-28 10:15:30,124 CRITICAL root MainThread example.py <module> : This is a critical message.