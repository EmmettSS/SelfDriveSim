"""WebSocket consumers bridging the mobile controller and the 3D renderer."""
import json
import logging

from channels.generic.websocket import AsyncWebsocketConsumer

logger = logging.getLogger(__name__)

RENDER_GROUP = "simulation_render"
TURN_SIGNAL_VALUES = {0, 1, 2}  # off / left / right


def _clip(value, low, high):
    return max(low, min(high, value))


def validate_command(data):
    """Validate and clamp a raw control payload.

    Returns a normalised dict, or raises ValueError on malformed input.
    """
    if not isinstance(data, dict):
        raise ValueError("payload must be a JSON object")

    try:
        steering = float(data.get("steering", 0.0))
        throttle = float(data.get("throttle", 0.0))
        brake = float(data.get("brake", 0.0))
        turn_signal = int(data.get("turn_signal", 0))
    except (TypeError, ValueError) as exc:
        raise ValueError("control values must be numeric") from exc

    for v in (steering, throttle, brake):
        if v != v or v in (float("inf"), float("-inf")):
            raise ValueError("control values must be finite")

    if turn_signal not in TURN_SIGNAL_VALUES:
        turn_signal = 0

    return {
        "steering": _clip(steering, -1.0, 1.0),
        "throttle": _clip(throttle, 0.0, 1.0),
        "brake": _clip(brake, 0.0, 1.0),
        "turn_signal": turn_signal,
    }


class ControlConsumer(AsyncWebsocketConsumer):
    """Receives driving commands from the phone and forwards them to renderers."""

    async def connect(self):
        await self.channel_layer.group_add(RENDER_GROUP, self.channel_name)
        await self.accept()

    async def disconnect(self, code):
        await self.channel_layer.group_discard(RENDER_GROUP, self.channel_name)

    async def receive(self, text_data=None, bytes_data=None):
        try:
            command = validate_command(json.loads(text_data or ""))
        except (json.JSONDecodeError, ValueError) as exc:
            await self.send(text_data=json.dumps({"type": "error", "detail": str(exc)}))
            return

        await self.channel_layer.group_send(
            RENDER_GROUP, {"type": "control.update", "command": command}
        )

    async def control_update(self, event):
        # Controllers share the group but do not need echoes of commands.
        pass


class RenderConsumer(AsyncWebsocketConsumer):
    """Pushes validated commands to the Three.js page."""

    async def connect(self):
        await self.channel_layer.group_add(RENDER_GROUP, self.channel_name)
        await self.accept()

    async def disconnect(self, code):
        await self.channel_layer.group_discard(RENDER_GROUP, self.channel_name)

    async def control_update(self, event):
        await self.send(text_data=json.dumps({"type": "control", **event["command"]}))
