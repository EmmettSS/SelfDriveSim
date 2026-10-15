import unittest

import numpy as np
import tf_keras as keras

from models.controller_model import INPUT_SHAPE, build_controller_model, compile_controller_model
from models.train_controller import _label_summary, evaluate_predictions


class ControllerArchitectureTests(unittest.TestCase):
    def test_four_control_contract_is_small_and_uses_portable_layers(self):
        model = build_controller_model()
        compile_controller_model(model)

        self.assertEqual(model.input_shape, (None, *INPUT_SHAPE))
        self.assertEqual(model.output_names, ["steering", "throttle", "brake", "turn_signal"])
        self.assertEqual([tuple(output.shape) for output in model.outputs], [
            (None, 1), (None, 1), (None, 1), (None, 3),
        ])
        self.assertLessEqual(model.count_params(), 100_000)
        self.assertFalse(any(isinstance(layer, keras.layers.GRU) for layer in model.layers))

        outputs = model(np.zeros((2, *INPUT_SHAPE), dtype=np.float32), training=False)
        steering, throttle, brake, turn_signal = [value.numpy() for value in outputs]
        self.assertTrue(np.all(np.abs(steering) <= 1))
        self.assertTrue(np.all((throttle >= 0) & (throttle <= 1)))
        self.assertTrue(np.all((brake >= 0) & (brake <= 1)))
        self.assertTrue(np.allclose(turn_signal.sum(axis=1), 1.0, atol=1e-6))

    def test_validation_metrics_report_turn_classes_and_brake_recall(self):
        labels = {
            "steering": np.asarray([[0.0], [0.5], [-0.5], [0.0]], dtype=np.float32),
            "throttle": np.asarray([[0.4], [0.4], [0.0], [0.2]], dtype=np.float32),
            "brake": np.asarray([[0.0], [0.0], [0.8], [0.0]], dtype=np.float32),
            "turn_signal": np.asarray([0, 1, 2, 0], dtype=np.int32),
        }
        predictions = [
            np.asarray([[0.1], [0.4], [-0.4], [0.0]], dtype=np.float32),
            np.asarray([[0.3], [0.5], [0.1], [0.2]], dtype=np.float32),
            np.asarray([[0.0], [0.1], [0.7], [0.0]], dtype=np.float32),
            np.asarray([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7], [0.8, 0.1, 0.1]], dtype=np.float32),
        ]

        metrics = evaluate_predictions(labels, predictions)

        self.assertAlmostEqual(metrics["steering"]["mae"], 0.075, places=6)
        self.assertEqual(metrics["brake"]["positive_support"], 1)
        self.assertEqual(metrics["brake"]["positive_recall"], 1.0)
        self.assertEqual(metrics["brake"]["threshold_sweep"]["0.10"]["recall"], 1.0)
        self.assertEqual(metrics["turn_signal"]["macro_f1"], 1.0)
        summary = _label_summary(labels)
        self.assertEqual(summary["samples"], 4)
        self.assertEqual(summary["turn_signal_counts"], {"off": 2, "left": 1, "right": 1})
        self.assertEqual(metrics["turn_signal"]["confusion_matrix_true_rows_predicted_columns"], [
            [2, 0, 0], [0, 1, 0], [0, 0, 1],
        ])


if __name__ == "__main__":
    unittest.main()
