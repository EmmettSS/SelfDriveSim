"""WebSocket tests for ControlConsumer and RenderConsumer.

The communicators go through ``URLRouter(websocket_urlpatterns)``, so the
paths declared in ``simulation.routing`` are exercised as well.
"""

from channels.routing import URLRouter
from channels.testing import WebsocketCommunicator

from simulation.routing import websocket_urlpatterns

# Seconds to wait for a message that is expected to arrive.
RECEIVE_TIMEOUT = 1

ROUTER = URLRouter(websocket_urlpatterns)


def make_communicator(path: str) -> WebsocketCommunicator:
    """Create a WebSocket test client for ``path``."""
    return WebsocketCommunicator(ROUTER, path)


async def test_control_consumer_accepts_connection():
    control = make_communicator("/ws/control/")

    connected, _ = await control.connect()

    assert connected
    await control.disconnect()


async def test_command_from_control_reaches_render():
    render = make_communicator("/ws/render/")
    control = make_communicator("/ws/control/")
    assert (await render.connect())[0]
    assert (await control.connect())[0]

    command = {"steering": 0.25, "throttle": 0.5, "brake": 0.0, "turn_signal": 1}
    await control.send_json_to(command)

    # The renderer gets the command with a type field it can dispatch on.
    assert await render.receive_json_from(timeout=RECEIVE_TIMEOUT) == {
        "type": "control_update",
        **command,
    }
    # The sender must not receive its own command back.
    assert await control.receive_nothing(timeout=0.1)

    await control.disconnect()
    await render.disconnect()


async def test_out_of_range_steering_is_clipped_to_one():
    render = make_communicator("/ws/render/")
    control = make_communicator("/ws/control/")
    await render.connect()
    await control.connect()

    await control.send_json_to(
        {"steering": 5.0, "throttle": 0.0, "brake": 0.0, "turn_signal": 0}
    )

    message = await render.receive_json_from(timeout=RECEIVE_TIMEOUT)
    assert message["steering"] == 1.0

    await control.disconnect()
    await render.disconnect()


async def test_malformed_frames_get_an_error_and_are_not_broadcast():
    render = make_communicator("/ws/render/")
    control = make_communicator("/ws/control/")
    await render.connect()
    await control.connect()

    await control.send_to(text_data="not json")
    reply = await control.receive_json_from(timeout=RECEIVE_TIMEOUT)
    assert reply["type"] == "error"

    # Only steering is present, so the other three fields are missing.
    await control.send_json_to({"steering": 0.1})
    reply = await control.receive_json_from(timeout=RECEIVE_TIMEOUT)
    assert reply["type"] == "error"
    assert "missing fields" in reply["message"]

    assert await render.receive_nothing(timeout=0.1)

    await control.disconnect()
    await render.disconnect()


async def test_renderer_speed_reaches_the_phone_and_is_not_echoed():
    render = make_communicator("/ws/render/")
    control = make_communicator("/ws/control/")
    assert (await render.connect())[0]
    assert (await control.connect())[0]

    await render.send_json_to({"type": "telemetry", "speed_kmh": 47.6})
    assert await control.receive_json_from(timeout=RECEIVE_TIMEOUT) == {
        "type": "telemetry",
        "speed_kmh": 47.6,
    }
    assert await render.receive_nothing(timeout=0.1)

    await render.send_json_to({"type": "telemetry", "speed_kmh": -12})
    assert await control.receive_json_from(timeout=RECEIVE_TIMEOUT) == {
        "type": "telemetry",
        "speed_kmh": 0.0,
    }

    await render.send_json_to({"speed_kmh": 10})
    reply = await render.receive_json_from(timeout=RECEIVE_TIMEOUT)
    assert reply["type"] == "error"
    assert await control.receive_nothing(timeout=0.1)

    await control.disconnect()
    await render.disconnect()
