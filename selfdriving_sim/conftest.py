"""Shared pytest fixtures for the SelfDriveSim server."""

import pytest


@pytest.fixture(autouse=True)
def in_memory_channel_layer(request, settings):
    """Run each test on the in-memory channel layer, so no Redis server is needed.

    Tests marked with ``@pytest.mark.redis`` keep the Redis layer from settings.
    """
    if request.node.get_closest_marker("redis") is None:
        settings.CHANNEL_LAYERS = {
            "default": {"BACKEND": "channels.layers.InMemoryChannelLayer"},
        }
