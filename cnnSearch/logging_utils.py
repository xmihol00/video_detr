from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
import logging.config
import os
from pathlib import Path
import threading
import time
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class LoggingConfig:
    logLevel: str = "INFO"
    logFormat: str = "text"
    logFilePath: Optional[str] = None
    enableConsole: bool = True


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        for key, value in record.__dict__.items():
            if key.startswith("_"):
                continue
            if key in {
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "exc_info",
                "exc_text",
                "stack_info",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "message",
            }:
                continue
            payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False)


class EventLogger:
    def __init__(self, baseLogger: logging.Logger) -> None:
        self._baseLogger = baseLogger
        self._onceKeys: set[str] = set()
        self._everyNCounters: Dict[str, int] = {}
        self._intervalTimes: Dict[str, float] = {}
        self._lock = threading.Lock()

    def debug(self, message: str, **kwargs: object) -> None:
        self._baseLogger.debug(message, extra=kwargs)

    def info(self, message: str, **kwargs: object) -> None:
        self._baseLogger.info(message, extra=kwargs)

    def warning(self, message: str, **kwargs: object) -> None:
        self._baseLogger.warning(message, extra=kwargs)

    def error(self, message: str, **kwargs: object) -> None:
        self._baseLogger.error(message, extra=kwargs)

    def exception(self, message: str, **kwargs: object) -> None:
        self._baseLogger.exception(message, extra=kwargs)

    def logOnce(self, key: str, message: str, level: int = logging.INFO, **kwargs: object) -> None:
        with self._lock:
            if key in self._onceKeys:
                return
            self._onceKeys.add(key)
        self._baseLogger.log(level, message, extra=kwargs)

    def logEveryN(self, key: str, everyN: int, message: str, level: int = logging.INFO, **kwargs: object) -> None:
        if everyN <= 0:
            self._baseLogger.log(level, message, extra=kwargs)
            return

        with self._lock:
            current = self._everyNCounters.get(key, 0) + 1
            self._everyNCounters[key] = current
            if current % everyN != 0:
                return

        self._baseLogger.log(level, message, extra=kwargs)

    def logInterval(self, key: str, intervalSeconds: float, message: str, level: int = logging.INFO, **kwargs: object) -> None:
        if intervalSeconds <= 0:
            self._baseLogger.log(level, message, extra=kwargs)
            return

        now = time.monotonic()
        with self._lock:
            lastTime = self._intervalTimes.get(key)
            if lastTime is not None and now - lastTime < intervalSeconds:
                return
            self._intervalTimes[key] = now

        self._baseLogger.log(level, message, extra=kwargs)


_CONFIGURED = False
_CONFIG_LOCK = threading.Lock()


def _resolveLevel(logLevel: str) -> str:
    return logLevel.upper().strip()


def configureLogging(config: LoggingConfig) -> None:
    global _CONFIGURED

    with _CONFIG_LOCK:
        handlers: Dict[str, Dict[str, Any]] = {}
        rootHandlers: list[str] = []

        formatterName = "jsonFormatter" if config.logFormat.lower() == "json" else "textFormatter"

        if config.enableConsole:
            handlers["console"] = {
                "class": "logging.StreamHandler",
                "level": _resolveLevel(config.logLevel),
                "formatter": formatterName,
                "stream": "ext://sys.stdout",
            }
            rootHandlers.append("console")

        if config.logFilePath:
            targetPath = Path(config.logFilePath)
            targetPath.parent.mkdir(parents=True, exist_ok=True)
            handlers["file"] = {
                "class": "logging.FileHandler",
                "level": _resolveLevel(config.logLevel),
                "formatter": formatterName,
                "filename": str(targetPath),
                "encoding": "utf-8",
            }
            rootHandlers.append("file")

        if not rootHandlers:
            handlers["console"] = {
                "class": "logging.StreamHandler",
                "level": _resolveLevel(config.logLevel),
                "formatter": formatterName,
                "stream": "ext://sys.stdout",
            }
            rootHandlers.append("console")

        logging.config.dictConfig(
            {
                "version": 1,
                "disable_existing_loggers": False,
                "formatters": {
                    "textFormatter": {
                        "format": "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                        "datefmt": "%Y-%m-%d %H:%M:%S",
                    },
                    "jsonFormatter": {
                        "()": "cnnSearch.logging_utils.JsonLogFormatter",
                    },
                },
                "handlers": handlers,
                "root": {
                    "level": _resolveLevel(config.logLevel),
                    "handlers": rootHandlers,
                },
            }
        )

        _CONFIGURED = True


def configureLoggingFromEnvironment(defaultLevel: str = "INFO") -> None:
    envLevel = os.environ.get("CNNSEARCH_LOG_LEVEL", defaultLevel)
    envFormat = os.environ.get("CNNSEARCH_LOG_FORMAT", "text")
    envFile = os.environ.get("CNNSEARCH_LOG_FILE")

    configureLogging(
        LoggingConfig(
            logLevel=envLevel,
            logFormat=envFormat,
            logFilePath=envFile,
            enableConsole=True,
        )
    )


def getEventLogger(name: str) -> EventLogger:
    if not _CONFIGURED:
        configureLoggingFromEnvironment()
    return EventLogger(logging.getLogger(name))
