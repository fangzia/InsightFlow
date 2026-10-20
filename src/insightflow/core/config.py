"""InsightFlow 全局配置中心。

所有配置项从带 ``INSIGHTFLOW_`` 前缀的环境变量和/或仓库根目录下的
``.env`` 文件加载。
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """项目全局配置。"""

    model_config = SettingsConfigDict(
        env_prefix="INSIGHTFLOW_",
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM 提供方（通过 .env 填写） ---
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    # Agent 循环与 baseline 评测默认使用的模型。
    default_model: str = "gpt-4o"

    # --- Spider 2.0 集成 ---
    # 官方仓库以 vendored 方式放在 third_party/（已 gitignore），
    # 供评测 harness、官方基线与本项目适配器共用。
    spider2_dir: Path = REPO_ROOT / "third_party" / "Spider2"
    spider2_dbt_dir: Path = spider2_dir / "spider2-dbt"


@lru_cache
def get_settings() -> Settings:
    return Settings()
