# Copyright (c) Alibaba, Inc. and its affiliates.
"""
Unified logging utilities for Sirchmunk
Provides flexible logging with optional callbacks and fallback to loguru
Supports both synchronous and asynchronous logging
"""
import asyncio
from typing import Any, Awaitable, Callable, Optional, Union

from loguru import logger as default_logger


# Type alias for log callback function (can be sync or async)
# Signature: (level: str, message: str, end: str, flush: bool) -> None or Awaitable[None]
LogCallback = Optional[Callable[[str, str, str, bool], Union[None, Awaitable[None]]]]


async def log_with_callback_async(
    level: str,
    message: str,
    log_callback: LogCallback = None,
    flush: bool = False,
    end: str = "\n",
) -> None:
    """
    Send log message through callback if available, otherwise use loguru logger.
    
    This is a universal logging utility that supports both synchronous and
    asynchronous callback functions, with automatic fallback to loguru.
    
    Args:
        level: Log level (e.g., "info", "debug", "error", "warning", "success")
        message: Message content to log
        log_callback: Optional callback function (sync or async) that takes (level, message).
                     If None, uses loguru's default_logger.
        flush: If True, force immediate output and use raw mode (no timestamp/level prefix).
               Useful for progress indicators. Equivalent to logger.opt(raw=True).
        end: String appended after the message (default: "\n")
    
    Examples:
        # Using default loguru logger (with prefix)
        await log_with_callback("info", "Processing started")
        # Output: 2026-01-16 10:30:00.123 | INFO | Processing started
        
        # Progress indicator without prefix (flush=True removes formatting)
        await log_with_callback("info", "Processing...", flush=True, end="")
        await log_with_callback("info", " Done!", flush=True, end="\n")
        # Output: Processing... Done!
        
        # Using custom async callback
        async def my_callback(level: str, msg: str):
            await websocket.send_text(f"[{level}] {msg}")
        await log_with_callback("debug", "Custom log", log_callback=my_callback)
    """
    if log_callback is not None:
        # Pass original message, end, and flush to callback
        # Let the callback handle message formatting
        if asyncio.iscoroutinefunction(log_callback):
            await log_callback(level, message, end, flush)
        else:
            # Call sync callback directly with all parameters
            log_callback(level, message, end, flush)
        
        # If flush is requested and callback is async, yield control to allow immediate processing
        if flush and asyncio.iscoroutinefunction(log_callback):
            await asyncio.sleep(0)
    else:
        # Fallback to loguru logger (process message locally)
        full_message = message + end if end else message
        if flush:
            # Use raw mode (no prefix) for flush=True
            default_logger.opt(raw=True).log(level.upper(), full_message)
        else:
            # Normal formatted output with prefix
            getattr(default_logger, level.lower())(full_message.rstrip("\n"))


def log_with_callback(
    level: str,
    message: str,
    log_callback: LogCallback = None,
    flush: bool = False,
    end: str = "\n",
) -> None:
    """
    Synchronous version of log_with_callback.
    
    Args:
        level: Log level (e.g., "info", "debug", "error", "warning", "success")
        message: Message content to log
        log_callback: Optional callback function (must be sync)
        flush: If True, force immediate output and use raw mode (no timestamp/level prefix).
               Useful for progress indicators. Equivalent to logger.opt(raw=True).
        end: String appended after the message (default: "\n")
    
    Examples:
        # Normal logging (with prefix)
        log_with_callback("info", "Processing started")
        # Output: 2026-01-16 10:30:00.123 | INFO | Processing started
        
        # Progress indicator without prefix (flush=True removes formatting)
        log_with_callback("info", "Loading", flush=True, end="")
        log_with_callback("info", "...", flush=True, end="")
        log_with_callback("info", " Done!", flush=True)
        # Output: Loading... Done!
    """
    if log_callback is not None:
        # Pass original message, end, and flush to callback
        # Let the callback handle message formatting
        if not asyncio.iscoroutinefunction(log_callback):
            log_callback(level, message, end, flush)
        else:
            # If async callback provided in sync mode, schedule safely.
            # Avoid asyncio.run() when already inside a running event loop.
            try:
                running_loop = asyncio.get_running_loop()
            except RuntimeError:
                asyncio.run(log_callback(level, message, end, flush))
            else:
                running_loop.create_task(log_callback(level, message, end, flush))
    else:
        # Fallback to loguru logger (process message locally)
        full_message = message + end if end else message
        if flush:
            # Use raw mode (no prefix) for flush=True
            default_logger.opt(raw=True).log(level.upper(), full_message)
        else:
            # Normal formatted output with prefix
            getattr(default_logger, level.lower())(full_message.rstrip("\n"))


def create_logger(log_callback: LogCallback = None, enable_async: bool = True) -> Union["AsyncLogger", "SyncLogger"]:
    """
    Create a logger instance with a bound log_callback.
    
    This factory function creates a logger with logger-style methods (info, warning, etc.)
    pre-configured with a specific callback, compatible with loguru logger usage.
    
    Args:
        log_callback: Optional callback function to bind
        enable_async: If True, create AsyncLogger; if False, create SyncLogger
        
    Returns:
        AsyncLogger or SyncLogger instance depending on enable_async parameter
        
    Example:
        # Create async logger (default)
        async def my_callback(level: str, msg: str):
            print(f"[{level}] {msg}")
        
        logger = create_logger(log_callback=my_callback, enable_async=True)
        await logger.info("Starting process")  # Async usage
        
        # Create sync logger
        def sync_callback(level: str, msg: str):
            print(f"[{level}] {msg}")
        
        logger = create_logger(log_callback=sync_callback, enable_async=False)
        logger.info("Starting process")  # Sync usage (no await)
        
        # Without callback (uses default loguru)
        async_logger = create_logger(enable_async=True)
        await async_logger.info("Async with loguru")
        
        sync_logger = create_logger(enable_async=False)
        sync_logger.info("Sync with loguru")
    """
    if enable_async:
        return AsyncLogger(log_callback=log_callback)
    else:
        return SyncLogger(log_callback=log_callback)


class SyncLogger:
    """
    Synchronous logger class with optional callback support.
    
    Provides a synchronous interface for logging. Use this when you need
    synchronous logging or when working in non-async contexts.
    
    Supports print-like flush and end parameters for advanced output control.
    When flush=True, uses raw mode (no timestamp/level prefix) for clean output.
    
    Example:
        # With custom sync callback
        def my_callback(level: str, msg: str):
            print(f"[{level}] {msg}", end="")
        
        logger = SyncLogger(log_callback=my_callback)
        logger.info("Starting process")
        logger.error("Failed to connect")
        
        # Progress indicator (flush=True removes prefix for clean output)
        logger.info("Processing", flush=True, end="")
        logger.info("...", flush=True, end="")
        logger.info(" Done!", flush=True)
        # Output: Processing... Done!
        
        # Without callback (uses loguru with normal formatting)
        logger = SyncLogger()
        logger.info("Using default logger")
        # Output: 2026-01-16 10:30:00.123 | INFO | Using default logger
    """
    
    def __init__(self, log_callback: LogCallback = None):
        """
        Initialize sync logger with optional callback.
        
        Args:
            log_callback: Optional callback function (preferably sync)
        """
        self.log_callback = log_callback
    
    def log(self, level: str, message: str, flush: bool = False, end: str = "\n"):
        """Log a message at the specified level (synchronous)"""
        log_with_callback(level, message, log_callback=self.log_callback, flush=flush, end=end)
    
    def debug(self, message: str, flush: bool = False, end: str = "\n"):
        """Log a debug message (synchronous)"""
        self.log("debug", message, flush=flush, end=end)
    
    def info(self, message: str, flush: bool = False, end: str = "\n"):
        """Log an info message (synchronous)"""
        self.log("info", message, flush=flush, end=end)
    
    def warning(self, message: str, flush: bool = False, end: str = "\n"):
        """Log a warning message (synchronous)"""
        self.log("warning", message, flush=flush, end=end)
    
    def error(self, message: str, flush: bool = False, end: str = "\n"):
        """Log an error message (synchronous)"""
        self.log("error", message, flush=flush, end=end)
    
    def success(self, message: str, flush: bool = False, end: str = "\n"):
        """Log a success message (synchronous)"""
        self.log("success", message, flush=flush, end=end)
    
    def critical(self, message: str, flush: bool = False, end: str = "\n"):
        """Log a critical message (synchronous)"""
        self.log("critical", message, flush=flush, end=end)


class AsyncLogger:
    """
    Async logger class with optional callback support.
    
    Provides a class-based interface for logging with instance-level
    callback configuration. Useful for classes that need persistent
    logging configuration.
    
    Supports print-like flush and end parameters for advanced output control.
    When flush=True, uses raw mode (no timestamp/level prefix) for clean output.
    
    Example:
        # With custom callback
        async def my_callback(level: str, msg: str):
            await websocket.send(f"{level}: {msg}")
        
        logger = AsyncLogger(log_callback=my_callback)
        await logger.info("Starting process")
        await logger.error("Failed to connect")
        
        # Progress indicator (flush=True removes prefix for clean output)
        await logger.info("Processing", flush=True, end="")
        await logger.info("...", flush=True, end="")
        await logger.info(" Done!", flush=True)
        # Output: Processing... Done!
        
        # Without callback (uses loguru with normal formatting)
        logger = AsyncLogger()
        await logger.info("Using default logger")
        # Output: 2026-01-16 10:30:00.123 | INFO | Using default logger
    """
    
    def __init__(self, log_callback: LogCallback = None):
        """
        Initialize async logger with optional callback.
        
        Args:
            log_callback: Optional callback function (sync or async)
        """
        self.log_callback = log_callback
    
    async def log(self, level: str, message: str, flush: bool = False, end: str = "\n"):
        """
        Log a message at the specified level.
        
        Args:
            level: Log level
            message: Message to log
            flush: If True, force immediate output
            end: String appended after message (default: "\n")
        """
        await log_with_callback_async(level, message, log_callback=self.log_callback, flush=flush, end=end)
    
    async def debug(self, message: str, flush: bool = False, end: str = "\n"):
        """
        Log a debug message.
        
        Args:
            message: Message to log
            flush: If True, force immediate output
            end: String appended after message (default: "\n")
        """
        await self.log("debug", message, flush=flush, end=end)
    
    async def info(self, message: str, flush: bool = False, end: str = "\n"):
        """
        Log an info message.
        
        Args:
            message: Message to log
            flush: If True, force immediate output
            end: String appended after message (default: "\n")
        """
        await self.log("info", message, flush=flush, end=end)
    
    async def warning(self, message: str, flush: bool = False, end: str = "\n"):
        """
        Log a warning message.
        
        Args:
            message: Message to log
            flush: If True, force immediate output
            end: String appended after message (default: "\n")
        """
        await self.log("warning", message, flush=flush, end=end)
    
    async def error(self, message: str, flush: bool = False, end: str = "\n"):
        """
        Log an error message.
        
        Args:
            message: Message to log
            flush: If True, force immediate output
            end: String appended after message (default: "\n")
        """
        await self.log("error", message, flush=flush, end=end)
    
    async def success(self, message: str, flush: bool = False, end: str = "\n"):
        """
        Log a success message.
        
        Args:
            message: Message to log
            flush: If True, force immediate output
            end: String appended after message (default: "\n")
        """
        await self.log("success", message, flush=flush, end=end)
    
    async def critical(self, message: str, flush: bool = False, end: str = "\n"):
        """
        Log a critical message.
        
        Args:
            message: Message to log
            flush: If True, force immediate output
            end: String appended after message (default: "\n")
        """
        await self.log("critical", message, flush=flush, end=end)
