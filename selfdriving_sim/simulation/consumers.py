"""WebSocket consumers for the closed-loop driving simulator.

Message flow::

    phone --/ws/control/--> ControlConsumer
          --group_send(simulation_render, control.update)-->
    browser (Three.js) <--/ws/render/-- RenderConsumer

    browser --/ws/render/--> RenderConsumer
          --group_send(simulation_render, telemetry.update)-->
    phone <--/ws/control/-- ControlConsumer

Control frames are validated and clipped on the server before they are
broadcast, so the renderer can rely on the values it receives. Telemetry
is a speed readout only; it does not apply driving commands.
"""

import json
import logging
import math
from typing import Any

from channels.generic.websocket import AsyncJsonWebsocketConsumer

logger = logging.getLogger(__name__)

# Channel group shared by the control and render consumers.
RENDER_GROUP = "simulation_render"

# Inclusive valid ranges for each control field.
STEERING_RANGE = (-1.0, 1.0)
PEDAL_RANGE = (0.0, 1.0)  # throttle and brake
TURN_SIGNAL_RANGE = (0, 2)  # 0 = off, 1 = left, 2 = right

COMMAND_FIELDS = ("steering", "throttle", "brake", "turn_signal")

# Inclusive km/h range reported by the Three.js scene to the phone HUD.
SPEED_KMH_RANGE = (0.0, 400.0)


class CommandValidationError(ValueError):
    """Raised when a control payload cannot be turned into a valid command."""


def _clip(value: float, bounds: tuple[float, float]) -> float:
    """Clamp ``value`` into the inclusive range given by ``bounds``."""
    low, high = bounds
    return min(max(value, low), high)


def _read_number(payload: dict[str, Any], field: str) -> float:
    """Return ``payload[field]`` as a finite float, or raise a validation error."""
    value = payload[field]
    # bool is a subclass of int in Python, so it has to be rejected explicitly.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CommandValidationError(f"'{field}' must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise CommandValidationError(f"'{field}' must be a finite number")
    return number


def sanitize_command(payload: Any) -> dict[str, float | int]:
    """Validate a control payload and clip each field into its valid range.

    Args:
        payload: The decoded JSON value received from the phone.

    Returns:
        A dict with ``steering``, ``throttle`` and ``brake`` as floats, and
        ``turn_signal`` as an int that is 0, 1 or 2.

    Raises:
        CommandValidationError: If the payload is not an object, a field is
            missing, or a field is not a finite number.
    """
    if not isinstance(payload, dict):
        raise CommandValidationError("payload must be a JSON object")

    missing = [field for field in COMMAND_FIELDS if field not in payload]
    if missing:
        raise CommandValidationError(f"missing fields: {', '.join(missing)}")

    # The turn signal is a discrete class, so the clipped value is rounded to the nearest index.
    turn_signal = _clip(_read_number(payload, "turn_signal"), TURN_SIGNAL_RANGE)
    return {
        "steering": _clip(_read_number(payload, "steering"), STEERING_RANGE),
        "throttle": _clip(_read_number(payload, "throttle"), PEDAL_RANGE),
        "brake": _clip(_read_number(payload, "brake"), PEDAL_RANGE),
        "turn_signal": int(round(turn_signal)),
    }


def sanitize_telemetry(payload: Any) -> dict[str, float]:
    """Validate a renderer telemetry frame and clip speed into a finite range."""
    if not isinstance(payload, dict):
        raise CommandValidationError("payload must be a JSON object")
    if payload.get("type") != "telemetry":
        raise CommandValidationError("telemetry frames must set type to 'telemetry'")
    speed = _clip(_read_number(payload, "speed_kmh"), SPEED_KMH_RANGE)
    return {"speed_kmh": speed}


class ControlConsumer(AsyncJsonWebsocketConsumer):
    """Receives driving commands from the phone and broadcasts them to renderers."""

    async def connect(self):
        # Control clients also join the render group, as the phase 1 design requires.
        # This lets renderer status messages reach the phone in later phases.
        await self.channel_layer.group_add(RENDER_GROUP, self.channel_name)
        await self.accept()

    async def disconnect(self, code):
        await self.channel_layer.group_discard(RENDER_GROUP, self.channel_name)

    async def receive(self, text_data=None, bytes_data=None, **kwargs):
        """Decode the frame here so bad input gets an error reply and keeps the socket open."""
        if text_data is None:
            await self._reject("expected a JSON text frame")
            return
        try:
            content = json.loads(text_data)
        except json.JSONDecodeError:
            await self._reject("frame is not valid JSON")
            return
        await self.receive_json(content)

    async def receive_json(self, content, **kwargs):
        try:
            command = sanitize_command(content)
        except CommandValidationError as exc:
            await self._reject(str(exc))
            return

        # Log the validated controls only, never image data or an untrusted payload.
        logger.info(
            "Control /ws/control/ steering=%.3f throttle=%.3f brake=%.3f turn_signal=%d",
            command["steering"],
            command["throttle"],
            command["brake"],
            command["turn_signal"],
        )
        await self.channel_layer.group_send(
            RENDER_GROUP,
            {"type": "control.update", "command": command},
        )

    async def control_update(self, event):
        """Drop ``control.update`` events.

        This consumer is a group member, so it also receives commands sent by
        control clients, including its own. Forwarding them would echo the
        phone's commands back to it, so they are ignored here.
        """

    async def telemetry_update(self, event):
        """Forward the scene speed to the phone. Commands are never included."""
        await self.send_json({"type": "telemetry", "speed_kmh": event["speed_kmh"]})

    async def _reject(self, message):
        """Tell the sender why a frame was refused. Nothing is broadcast."""
        logger.debug("Rejected control frame: %s", message)
        await self.send_json({"type": "error", "message": message})


class RenderConsumer(AsyncJsonWebsocketConsumer):
    """Pushes validated driving commands to the Three.js page in the browser."""

    async def connect(self):
        await self.channel_layer.group_add(RENDER_GROUP, self.channel_name)
        await self.accept()

    async def disconnect(self, code):
        await self.channel_layer.group_discard(RENDER_GROUP, self.channel_name)

    async def control_update(self, event):
        """Send one validated command to the browser.

        The frame carries ``"type": "control_update"`` so the renderer can tell
        commands apart from other messages it may receive later.
        """
        await self.send_json({"type": "control_update", **event["command"]})

    async def receive(self, text_data=None, bytes_data=None, **kwargs):
        if text_data is None:
            await self._reject("expected a JSON text frame")
            return
        try:
            content = json.loads(text_data)
        except json.JSONDecodeError:
            await self._reject("frame is not valid JSON")
            return
        await self.receive_json(content)

    async def receive_json(self, content, **kwargs):
        try:
            telemetry = sanitize_telemetry(content)
        except CommandValidationError as exc:
            await self._reject(str(exc))
            return
        await self.channel_layer.group_send(
            RENDER_GROUP,
            {"type": "telemetry.update", "speed_kmh": telemetry["speed_kmh"]},
        )

    async def telemetry_update(self, event):
        """Drop telemetry echoes so the scene does not receive its own speed."""

    async def _reject(self, message):
        logger.debug("Rejected render frame: %s", message)
        await self.send_json({"type": "error", "message": message})
