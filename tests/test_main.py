import adadmire
import numpy as np


def test_loo_cv_cor():
    # generate data

    X = np.array([[1, 1], [1, 1]])
    D = np.array([[1, 0], [1, 0]])
    levels = np.array([2])
    lam = np.array([0.5])
    p_e = np.zeros([2, 2]) + 3.7856652196880396e-05
    B_e = np.zeros([2, 2]) + 20.51558176857432
    x_e = np.zeros([2, 2]) + 0.09034876630988453
    d_e = np.array([0.9997587101440837, 0.00024128985591637892])
    d_e = np.vstack((d_e, d_e))
    prob, B, lam_o,  x, d = adadmire.loo_cv_cor(
        X, D, levels, lam, oIterations=200)
    assert (prob == p_e).all()
    assert (B == B_e).all()
    assert (lam_o == lam).all()
    assert (x == x_e).all()
    assert (d == d_e).all()


def test_get_threshold_continuous():
    X = np.array([[1, 1], [1, 1]])
    X_hat = np.array([[1, 2], [1, 1]])
    dev = np.zeros([2, 2]) + 0.1
    X_cor, threshold, n, pos = adadmire.get_threshold_continuous(X, X_hat, dev)
    assert (X_cor == X).all()
    assert threshold == 1.5162394040780434
    assert n == 0
    assert pos.size == 0


def test_get_threshold_discrete():
    D = np.array([[1, 0], [0, 1], [1, 0]])
    levels = np.array([2])
    D_hat = np.array([[0.95, 0.05], [0.05, 0.95], [0.95, 0.05]])
    n, threshold, pos = adadmire.get_threshold_discrete(D, levels, D_hat)
    assert threshold == 0.49295914126251644
    assert n == 0
    assert pos.size == 0


def test_place_anomalies_continuous():
    X = np.array([[1.1, 2.0, 4.1], [1.0, 3.5, 3.2], [1.5, 2.2, 4.2]])
    n_ano = 2
    epsilon = np.array([2.0])
    pos_e = np.array([[1., 1.], [2., 0.]])
    sim = np.copy(X)
    sim[1, 1] = -0.12842922430271653
    sim[2, 0] = 0.5570093853992585
    pos_e = np.array([[1, 1], [2, 0]])
    X_ano, pos = adadmire.place_anomalies_continuous(
        X, n_ano, epsilon, positive=False)
    assert (X_ano == sim).all()
    assert (pos == pos_e).all()


def test_impute():
    X = np.array([[np.nan, 1.5, 2.1], [0.5, 2.4, 1.2], [0.8, 0.7, 0.5]])
    D = np.array([[1, 0, 0, 1], [np.nan, np.nan, 1, 0], [0, 1, 0, 1]])
    levels = np.array([2, 2])
    lam = np.array([0.05, 0.04])
    X_e = np.copy(X)
    X_e[0, 0] = -0.3044102747658219
    D_e = np.copy(D)
    D_e[1, 0:2] = np.array([1, 0])
    X_imp, D_imp, lam_o = adadmire.impute(
        X, D, levels, lam, oIterations=200, oTol=1e-6)
    assert lam_o == 0.05
    assert (X_imp == X_e).all()
    assert (D_imp == D_e).all()
