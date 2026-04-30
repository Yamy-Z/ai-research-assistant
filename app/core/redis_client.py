from typing import Any

import redis
from app.core.config import get_settings
from app.utils.logger import setup_logger

settings = get_settings()
logger = setup_logger(__name__)


class RedisClient:
    """Redis client singleton."""
    
    _instance = None
    client: Any
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.client = redis.from_url(
                settings.redis_url,
                decode_responses=True
            )
            logger.info("Redis client initialized")
        return cls._instance
    
    def get(self, key: str):
        """Get value from Redis."""
        return self.client.get(key)
    
    def set(self, key: str, value: str, ex: int = 3600):
        """Set value in Redis with expiration."""
        return self.client.set(key, value, ex=ex)
    
    def delete(self, key: str):
        """Delete key from Redis."""
        return self.client.delete(key)
    
    def exists(self, key: str) -> bool:
        """Check if key exists."""
        return bool(self.client.exists(key))


def get_redis() -> RedisClient:
    """Get Redis client dependency."""
    return RedisClient()
