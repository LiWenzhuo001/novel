"""Atomically rebuild a single indexed knowledge file."""
import argparse
import asyncio
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.context import reset_current_user, set_current_user
from app.services.kb_service import reindex_file


async def run(file_id: str, user_id: str) -> None:
    token = set_current_user(user_id)
    try:
        result = await reindex_file(file_id, user_id=user_id)
    finally:
        reset_current_user(token)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file_id")
    parser.add_argument("--user", default="default")
    args = parser.parse_args()
    asyncio.run(run(args.file_id, args.user))
