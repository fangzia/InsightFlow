from insightflow import __version__
from insightflow.core.config import get_settings


def test_version() -> None:
    assert __version__ == "0.1.0"


def test_settings_paths() -> None:
    settings = get_settings()
    assert settings.spider2_dbt_dir.name == "spider2-dbt"
