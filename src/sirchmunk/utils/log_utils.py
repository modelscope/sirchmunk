# Copyright (c) Alibaba, Inc. and its affiliates.
"""
Unified logging utilities for Sirchmunk
Provides flexible logging with optional callbacks and fallback to loguru
"""
import asyncio
from typing import Any, Awaitable, Callable, Optional, Union

from loguru import logger as default_logger


# Type alias for log callback function (can be sync or async)
LogCallback = Optional[Callable[[str, str], Union[None, Awaitable[None]]]]


async def log_with_callback(
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
        flush: If True, force immediate output (useful for progress indicators)
        end: String appended after the message (default: "\n")
    
    Examples:
        # Using default loguru logger
        await log_with_callback("info", "Processing started")
        
        # Progress indicator without newline
        await log_with_callback("info", "Processing...", flush=True, end="")
        await log_with_callback("info", " Done!", flush=True, end="\n")
        
        # Using custom async callback
        async def my_callback(level: str, msg: str):
            await websocket.send_text(f"[{level}] {msg}")
        await log_with_callback("debug", "Custom log", log_callback=my_callback)
    """
    # Append end character to message
    full_message = message + end if end else message
    
    if log_callback is not None:
        # Check if callback is async
        if asyncio.iscoroutinefunction(log_callback):
            await log_callback(level, full_message)
        else:
            # Call sync callback directly
            log_callback(level, full_message)
        
        # If flush is requested and callback is async, yield control to allow immediate processing
        if flush and asyncio.iscoroutinefunction(log_callback):
            await asyncio.sleep(0)
    else:
        # Fallback to loguru logger
        # For loguru, we just log the message (loguru handles its own output)
        getattr(default_logger, level.lower())(full_message.rstrip("\n"))


def create_logger(log_callback: LogCallback = None) -> "AsyncLogger":
    """
    Create an AsyncLogger instance with a bound log_callback.
    
    This factory function creates a logger with logger-style methods (info, warning, etc.)
    pre-configured with a specific callback, compatible with loguru logger usage.
    
    Args:
        log_callback: Optional callback function to bind
        
    Returns:
        An AsyncLogger instance that can be used like: await logger.info("message")
        
    Example:
        # Create a custom logger
        async def my_callback(level: str, msg: str):
            print(f"[{level}] {msg}")
        
        logger = create_logger(log_callback=my_callback)
        
        # Use the logger (same style as loguru)
        await logger.info("Starting process")
        await logger.error("Failed to load file")
        await logger.warning("Low memory")
        
        # Without callback (uses default loguru)
        logger = create_logger()
        await logger.info("Using default logger")
    """
    return AsyncLogger(log_callback=log_callback)


class AsyncLogger:
    """
    Async logger class with optional callback support.
    
    Provides a class-based interface for logging with instance-level
    callback configuration. Useful for classes that need persistent
    logging configuration.
    
    Supports print-like flush and end parameters for advanced output control.
    
    Example:
        # With custom callback
        async def my_callback(level: str, msg: str):
            await websocket.send(f"{level}: {msg}")
        
        logger = AsyncLogger(log_callback=my_callback)
        await logger.info("Starting process")
        await logger.error("Failed to connect")
        
        # Progress indicator
        await logger.info("Processing", flush=True, end="")
        await logger.info("...", flush=True, end="")
        await logger.info(" Done!", flush=True)
        
        # Without callback (uses loguru)
        logger = AsyncLogger()
        await logger.info("Using default logger")
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
        await log_with_callback(level, message, log_callback=self.log_callback, flush=flush, end=end)
    
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
