import os
import redis
import json
from typing import Dict, Any, Optional

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

redis_client = redis.from_url(REDIS_URL, decode_responses=True)

def get_conv_state(chat_id: int) -> Optional[Dict[str, Any]]:
    """從 Redis 根據 chat_id 取得會談狀態"""
    try:
        state_json = redis_client.get(str(chat_id))
        if state_json:
            return json.loads(state_json)
    except Exception as e:
        print(f"Error getting state from Redis for chat_id {chat_id}: {e}")
    return None

def set_conv_state(chat_id: int, state: Dict[str, Any]):
    """將會談狀態的 Python 字典存入 Redis"""
    try:
        state_json = json.dumps(state, ensure_ascii=False)
        # 86400 秒 = 24 小時 604800 秒 = 7 天
        redis_client.set(str(chat_id), state_json, ex=604800)
    except Exception as e:
        print(f"Error setting state to Redis for chat_id {chat_id}: {e}")


def delete_conv_state(chat_id: int):
    """從 Redis 刪除會談狀態"""
    try:
        redis_client.delete(str(chat_id))
    except Exception as e:
        print(f"Error deleting state from Redis for chat_id {chat_id}: {e}")
