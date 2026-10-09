"""Integration test for the Redis channel layer.

The test runs only when a Redis server is reachable at REDIS_HOST:REDIS_PORT.
Otherwise it is skipped, so a plain ``pytest`` run does not need Redis.
"""

import asyncio
import socket

import pytest
from channels.layers import get_channel_layer
from channels_redis.core import RedisChannelLayer
from django.conf import settings


def _redis_is_reachable() -> bool:
    """Return True if a TCP connection to the configured Redis server succeeds."""
    try:
        with socket.create_connection(
            (settings.REDIS_HOST, settings.REDIS_PORT), timeout=0.5
        ):
            return True
    except OSError:
        return False


pytestmark = [
    pytest.mark.redis,
    pytest.mark.skipif(
        not _redis_is_reachable(),
        reason="no Redis server reachable at REDIS_HOST:REDIS_PORT",
    ),
]


async def test_group_message_round_trip_over_redis():
    layer = get_channel_layer()
    assert isinstance(layer, RedisChannelLayer)

    channel = await layer.new_channel()
    group = "phase1_redis_check"
    await layer.group_add(group, channel)
    try:
        await layer.group_send(group, {"type": "ping.message", "value": 42})
        message = await asyncio.wait_for(layer.receive(channel), timeout=2)
    finally:
        await layer.group_discard(group, channel)

    assert message == {"type": "ping.message", "value": 42}
