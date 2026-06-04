from pathlib import Path

from app.core.config import Settings
from app.core.logging import setup_logging


def test_setup_logging_creates_log_file(tmp_path: Path) -> None:
    settings = Settings(
        log_dir=str(tmp_path),
        log_file_name="test.log",
        log_level="INFO",
    )
    log_file = setup_logging(settings)
    assert log_file.exists()
    assert log_file.name == "test.log"
