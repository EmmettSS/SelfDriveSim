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


async def test_control_stream_and_final_stop_reach_renderer_over_redis():
    """Exercise both consumers at the phone's 50 ms cadence on a real layer."""
    from channels.routing import URLRouter
    from channels.testing import WebsocketCommunicator

    from simulation.routing import websocket_urlpatterns

    router = URLRouter(websocket_urlpatterns)
    render = WebsocketCommunicator(router, "/ws/render/")
    control = WebsocketCommunicator(router, "/ws/control/")
    assert (await render.connect())[0]
    assert (await control.connect())[0]
    try:
        for tick in range(20):
            command = {
                "steering": round((tick - 10) / 20, 3),
                "throttle": 0.5,
                "brake": 0.0,
                "turn_signal": tick % 3,
            }
            await control.send_json_to(command)
            assert await render.receive_json_from(timeout=2) == {
                "type": "control_update", **command,
            }
            await asyncio.sleep(0.05)
        stop = {"steering": 0, "throttle": 0, "brake": 1, "turn_signal": 0}
        await control.send_json_to(stop)
        assert await render.receive_json_from(timeout=2) == {
            "type": "control_update", **stop,
        }
        assert await control.receive_nothing(timeout=0.1)
    finally:
        await control.disconnect()
        await render.disconnect()
