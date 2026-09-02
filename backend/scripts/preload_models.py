"""Pre-download local embedding and reranker models during deployment."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import settings
from app.core.embed import get_embeddings


def main() -> None:
    if settings.embedding_provider == "local":
        get_embeddings()
        print(f"embedding ready: {settings.embedding_model}")
    if settings.enable_reranker:
        from app.core.rerank import get_reranker
        get_reranker()
        if settings.reranker_provider == "siliconflow":
            print(f"SiliconFlow reranker client ready: {settings.reranker_model}")
        else:
            print(f"reranker ready: {settings.reranker_model}")


if __name__ == "__main__":
    main()
