import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目根目录（app/ 的上一级，即 docqa/）
BASE_DIR = Path(__file__).resolve().parent.parent

# 【P0 修复】离线开关内置到代码里，不再依赖运维记得设环境变量。
# 必须在任何 huggingface 相关库被 import 之前定下来：否则每次 embed 都会去
# huggingface.co 校验 etag，无网/有代理的环境会卡 20 秒以上甚至直接挂死。
# 万一确实要联网拉模型，显式设 EMBEDDING_OFFLINE=0 即可恢复。
if os.getenv("EMBEDDING_OFFLINE", "1") != "0":
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


class Settings(BaseSettings):
    # env_file 必须用绝对路径：写 ".env" 会跟着「当前工作目录」变，
    # 从其它目录启动时 .env 会静默加载失败、回退到默认值，且不报任何错。
    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", extra="ignore")

    # 【P0 修复】数据库与向量库都用绝对路径。写相对路径会跟着「当前工作目录」跑，
    # 换个目录启动（systemd / Docker）就会新建空库，看起来像数据凭空消失。
    database_url: str = f"sqlite:///{(BASE_DIR / '.docqa.db').as_posix()}"
    chroma_dir: str = str(BASE_DIR / "chroma_db")

    # ===== 大模型配置（第二阶段）=====

    openai_api_key: str = ""
    openai_base_url: str = "https://api.deepseek.com"
    model_name: str = "deepseek-flash"   # 原默认 "deepseek" 不是有效模型名，会被上游 400
    llm_timeout: float = 60.0            # 单次调用超时，避免请求无限挂住打满线程池

    # ===== 向量/嵌入配置 =====
    embedding_provider: str = "local"                  # "local" 或 "openai"
    embedding_model: str = "text-embedding-3-small"    # 仅 openai 模式使用

    def require_llm_config(self) -> None:
        """调用大模型前先自检，避免拿到空 key 时报出莫名其妙的错误。"""
        if not self.openai_api_key:
            raise RuntimeError("未配置 OPENAI_API_KEY，请先在 .env 中填写")

settings = Settings()