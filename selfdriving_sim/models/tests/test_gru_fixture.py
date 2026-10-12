"""The committed GRU reference fixture must match what Keras computes today.

The Node test client/tests/js/reset-after-gru.test.mjs replays the fixture through
the browser layer. This test regenerates it with Keras and fails if the committed
file is stale, for example after a Keras upgrade changes GRU behaviour.
"""

import json

import numpy as np

from models.tests import make_gru_fixture


def test_committed_fixture_matches_keras():
    committed = json.loads(make_gru_fixture.FIXTURE_PATH.read_text(encoding="utf-8"))
    regenerated = make_gru_fixture.build_fixture()

    assert committed["cases"].keys() == regenerated["cases"].keys()
    for name, case in regenerated["cases"].items():
        stored = committed["cases"][name]
        assert stored["units"] == case["units"]
        assert stored["resetAfter"] == case["resetAfter"]
        np.testing.assert_allclose(stored["expected"], case["expected"], atol=1e-6, err_msg=name)
        for key, values in case["weights"].items():
            np.testing.assert_allclose(stored["weights"][key], values, atol=1e-6, err_msg=f"{name}/{key}")
