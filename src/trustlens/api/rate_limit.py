"""Simple in-memory rate limiter (per-IP per-minute)."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import time as now_time


@dataclass
class RateLimiter:
    limit_per_min: int
    time_fn: callable = now_time
    buckets: dict[str, tuple[int, int]] = field(default_factory=dict)

    def allow(self, key: str) -> bool:
        """Return True if request is allowed for this minute bucket."""
        t = int(self.time_fn())
        minute = t // 60
        count, bucket = self.buckets.get(key, (0, minute))
        if bucket != minute:
            count = 0
            bucket = minute
        if count >= self.limit_per_min:
            self.buckets[key] = (count, bucket)
            return False
        self.buckets[key] = (count + 1, bucket)
        return True
