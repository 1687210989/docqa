from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目根目录（app/ 的上一级，即 docqa/）
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    # env_file 必须用绝对路径：写 ".env" 会跟着「当前工作目录」变，
    # 从其它目录启动时 .env 会静默加载失败、回退到默认值，且不报任何错。
    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", extra="ignore")

    # 默认值与 .env 保持一致，万一 .env 丢了也还是同一个库，不会静默切换
    database_url: str = "sqlite:///.docqa.db"

    # ===== 大模型配置（第二阶段）=====

    openai_api_key: str = ""
    openai_base_url: str = "https://api.deepseek.com"
    model_name: str = "deepseek"

    def require_llm_config(self) -> None:
        """调用大模型前先自检，避免拿到空 key 时报出莫名其妙的错误。"""
        if not self.openai_api_key:
            raise RuntimeError("未配置 OPENAI_API_KEY，请先在 .env 中填写")

settings = Settings()