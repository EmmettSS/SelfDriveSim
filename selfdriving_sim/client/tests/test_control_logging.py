"""Phone commands must be visible in the Django console without logging images."""

from unittest.mock import patch

from channels.routing import URLRouter
from channels.testing import WebsocketCommunicator

from simulation.routing import websocket_urlpatterns


async def test_valid_control_logs_only_sanitized_numeric_fields():
    control = WebsocketCommunicator(URLRouter(websocket_urlpatterns), "/ws/control/")
    await control.connect()
    try:
        with patch("simulation.consumers.logger.info") as log:
            await control.send_json_to({
                "steering": 5, "throttle": 0.7, "brake": 0, "turn_signal": 1,
                "image": "not part of the protocol",
            })
            # Waiting for an invalid frame's reply also drains the prior receive.
            await control.send_json_to({"steering": 0})
            assert (await control.receive_json_from())["type"] == "error"
            log.assert_called_once_with(
                "Control /ws/control/ steering=%.3f throttle=%.3f brake=%.3f turn_signal=%d",
                1.0, 0.7, 0.0, 1,
            )
    finally:
        await control.disconnect()


async def test_rejected_frames_are_not_logged_as_accepted_commands():
    control = WebsocketCommunicator(URLRouter(websocket_urlpatterns), "/ws/control/")
    await control.connect()
    try:
        with patch("simulation.consumers.logger.info") as log:
            await control.send_to(text_data="not JSON")
            assert (await control.receive_json_from())["type"] == "error"
            await control.send_json_to({
                "steering": True, "throttle": 0.5, "brake": 0, "turn_signal": 0,
            })
            assert (await control.receive_json_from())["type"] == "error"
            log.assert_not_called()
    finally:
        await control.disconnect()
