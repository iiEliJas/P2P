#
# Unit tests for src/utils/metrics.py
#

from __future__ import annotations

import numpy as np
import pytest

from src.utils.metrics import evaluate_all, mae, mape, rmse, smape



class TestMAE:
    def test_perfect_forecast(self):
        assert mae([1, 2, 3], [1, 2, 3]) == pytest.approx(0.0)

    def test_known_value(self):
        assert mae([10, 20, 30], [12, 18, 33]) == pytest.approx(7 / 3)

    def test_symmetric(self):
        assert mae([5, 10], [10, 5]) == pytest.approx(5.0)

    def test_single_element(self):
        assert mae([100], [90]) == pytest.approx(10.0)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            mae([1, 2], [1])

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            mae([], [])



class TestRMSE:
    def test_perfect_forecast(self):
        assert rmse([5, 5, 5], [5, 5, 5]) == pytest.approx(0.0)

    def test_known_value(self):
        assert rmse([0, 0, 0], [1, 2, 3]) == pytest.approx(np.sqrt(14 / 3))

    def test_rmse_ge_mae(self):
        a = [10, 20, 30, 40]
        p = [12, 15, 35, 38]
        assert rmse(a, p) >= mae(a, p)



class TestSMAPE:
    def test_perfect_forecast(self):
        assert smape([1, 2, 3], [1, 2, 3]) == pytest.approx(0.0)

    def test_both_zero_contributes_zero(self):
        # (0, 0) pair should result in 0 error
        assert smape([0, 10], [0, 10]) == pytest.approx(0.0)

    def test_bounded_200(self):
        result = smape([100, 0], [0, 100])
        assert 0 <= result <= 200

    def test_known_value(self):
        result = smape([100], [200])
        assert result == pytest.approx(200 / 3, rel=1e-4)

    def test_zero_actual_nonzero_pred(self):
        result = smape([0], [50])
        assert result == pytest.approx(200.0)



class TestMAPE:
    def test_zero_actuals_returns_nan(self):
        assert np.isnan(mape([0, 0], [1, 2]))

    def test_skips_zero_entries(self):
        result = mape([0, 10], [5, 12])
        assert result == pytest.approx(20.0)



class TestEvaluateAll:
    def test_returns_all_keys(self):
        result = evaluate_all([10, 20], [12, 18])
        assert set(result.keys()) == {"mae", "rmse", "smape", "mape"}

    def test_values_are_floats(self):
        result = evaluate_all([10, 20, 30], [11, 19, 31])
        for v in result.values():
            assert isinstance(v, float)
