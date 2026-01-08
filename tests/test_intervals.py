import numpy as np

from astree_eta.intervals import build_residual_interval, train_quantile_models


def test_residual_interval_predicts_offsets():
    base_pred = np.array([1.0, 2.0, 3.0])
    y = np.array([1.5, 2.5, 2.5])
    model = build_residual_interval(base_pred, y, (0.1, 0.9))
    outputs = model.predict(base_pred, (0.1, 0.9))
    assert set(outputs.keys()) == {0.1, 0.9}
    assert outputs[0.1].shape == base_pred.shape


def test_train_quantile_models_falls_back_on_small_sample():
    X = np.ones((5, 2))
    y = np.ones(5)
    result = train_quantile_models(X, y, min_samples=10)
    assert result["type"] == "residual"
