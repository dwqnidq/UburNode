"""loguru 初始化：控制台（开发）+ 滚动文件（生产采集）。

enqueue=True：异步写盘，避免阻塞请求线程。
"""

import sys
from pathlib import Path

from loguru import logger

from app.core.config import Settings

FILE_LOG_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
    "{name}:{function}:{line} | {message}"
)

CONSOLE_LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level>"
)


def setup_logging(settings: Settings) -> Path:
    """初始化双输出；返回日志文件路径供启动日志引用。"""
    log_dir = settings.log_dir_path
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / settings.log_file_name

    logger.remove()

    level = settings.log_level.upper()

    logger.add(sys.stderr, level=level, format=CONSOLE_LOG_FORMAT)

    logger.add(
        str(log_file),
        level=level,
        format=FILE_LOG_FORMAT,
        rotation=settings.log_rotation,
        retention=settings.log_retention,
        encoding="utf-8",
        enqueue=True,
    )

    logger.info("日志已初始化，级别={}，文件={}", level, log_file)
    return log_file
