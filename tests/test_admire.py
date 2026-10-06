import numpy
import pytest

import adadmire
import adadmire.main


class _Stop(Exception):
    pass


def test_admire_forwards_fitting_options_to_loo_cv_cor(monkeypatch):
    """admire() must pass oIterations, oTol and t on to loo_cv_cor()."""
    captured = {}

    def fake_loo_cv_cor(X, D, levels, lambda_seq, **kwargs):
        captured.update(kwargs)
        raise _Stop  # skip the (slow) remaining computation

    monkeypatch.setattr(adadmire.main, "loo_cv_cor", fake_loo_cv_cor)
    X = numpy.zeros((3, 2))
    D = numpy.zeros((3, 2))
    levels = numpy.array([2])
    with pytest.raises(_Stop):
        adadmire.admire(X, D, levels, numpy.array([0.1]), oIterations=7, oTol=1e-3, t=0.2)
    assert captured == {"oIterations": 7, "oTol": 1e-3, "t": 0.2}


def test_version_is_exported():
    assert isinstance(adadmire.__version__, str)
