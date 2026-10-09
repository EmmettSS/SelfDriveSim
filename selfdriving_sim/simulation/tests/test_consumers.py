"""Consumer tests using an in-memory channel layer (no Redis required)."""
import json

import pytest
from channels.testing import WebsocketCommunicator

from selfdriving_sim.asgi import application

IN_MEMORY_LAYER = {"default": {"BACKEND": "channels.layers.InMemoryChannelLayer"}}


@pytest.fixture(autouse=True)
def in_memory_layer(settings):
    settings.CHANNEL_LAYERS = IN_MEMORY_LAYER


async def test_control_connects():
    comm = WebsocketCommunicator(application, "/ws/control/")
    connected, _ = await comm.connect()
    assert connected
    await comm.disconnect()


async def test_control_to_render():
    render = WebsocketCommunicator(application, "/ws/render/")
    control = WebsocketCommunicator(application, "/ws/control/")
    assert (await render.connect())[0]
    assert (await control.connect())[0]

    await control.send_to(text_data=json.dumps(
        {"steering": -0.3, "throttle": 0.5, "brake": 0.0, "turn_signal": 1}
    ))
    msg = json.loads(await render.receive_from(timeout=2))
    assert msg == {"type": "control", "steering": -0.3, "throttle": 0.5,
                   "brake": 0.0, "turn_signal": 1}

    await control.disconnect()
    await render.disconnect()


async def test_steering_is_clipped():
    render = WebsocketCommunicator(application, "/ws/render/")
    control = WebsocketCommunicator(application, "/ws/control/")
    await render.connect()
    await control.connect()

    await control.send_to(text_data=json.dumps(
        {"steering": 5.0, "throttle": 0.2, "brake": 0.0, "turn_signal": 0}
    ))
    msg = json.loads(await render.receive_from(timeout=2))
    assert msg["steering"] == 1.0

    await control.disconnect()
    await render.disconnect()
