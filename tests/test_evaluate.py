"""Tests for evaluation metrics."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluate.metrics import (
    accuracy,
    brier_score,
    calibration_curve,
    evaluate_forecast,
    log_loss,
)


class TestBrierScore:
    def test_perfect_prediction(self):
        y = np.array([1, 0, 1, 0])
        p = np.array([1.0, 0.0, 1.0, 0.0])
        assert brier_score(y, p) == pytest.approx(0.0)

    def test_worst_prediction(self):
        y = np.array([1, 1, 0, 0])
        p = np.array([0.0, 0.0, 1.0, 1.0])
        assert brier_score(y, p) == pytest.approx(1.0)

    def test_uniform_prediction(self):
        y = np.array([1, 0, 1, 0])
        p = np.array([0.5, 0.5, 0.5, 0.5])
        assert brier_score(y, p) == pytest.approx(0.25)

    def test_returns_float(self):
        assert isinstance(brier_score(np.array([1, 0]), np.array([0.7, 0.3])), float)


class TestLogLoss:
    def test_small_for_good_predictions(self):
        y = np.array([1, 0, 1, 0])
        p = np.array([0.9, 0.1, 0.9, 0.1])
        assert log_loss(y, p) < 0.15

    def test_large_for_bad_predictions(self):
        y = np.array([1, 1, 0, 0])
        p = np.array([0.1, 0.1, 0.9, 0.9])
        assert log_loss(y, p) > 2.0

    def test_returns_float(self):
        assert isinstance(log_loss(np.array([1, 0]), np.array([0.7, 0.3])), float)


class TestAccuracy:
    def test_perfect(self):
        y = np.array([1, 0, 1, 0])
        p = np.array([0.9, 0.1, 0.8, 0.2])
        assert accuracy(y, p) == pytest.approx(1.0)

    def test_all_wrong(self):
        y = np.array([1, 1, 0, 0])
        p = np.array([0.1, 0.2, 0.8, 0.9])
        assert accuracy(y, p) == pytest.approx(0.0)

    def test_threshold_respected(self):
        y = np.array([1])
        p = np.array([0.49])
        assert accuracy(y, p, threshold=0.5) == pytest.approx(0.0)
        assert accuracy(y, p, threshold=0.4) == pytest.approx(1.0)


class TestCalibrationCurve:
    def test_returns_dataframe(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 100)
        p = rng.uniform(0, 1, 100)
        df = calibration_curve(y, p, n_bins=5)
        assert set(df.columns) == {
            "bin_center",
            "mean_predicted",
            "fraction_positive",
            "count",
        }

    def test_no_empty_bins_in_result(self):
        y = np.array([1, 0] * 50)
        p = np.linspace(0.01, 0.99, 100)
        df = calibration_curve(y, p, n_bins=10)
        assert (df["count"] > 0).all()


class TestEvaluateForecast:
    def test_keys(self):
        y = np.array([1, 0, 1, 0])
        p = np.array([0.7, 0.3, 0.8, 0.2])
        result = evaluate_forecast(y, p)
        assert set(result.keys()) == {"brier_score", "log_loss", "accuracy"}

    def test_all_floats(self):
        y = np.array([1, 0, 1])
        p = np.array([0.6, 0.4, 0.7])
        result = evaluate_forecast(y, p)
        for v in result.values():
            assert isinstance(v, float)
