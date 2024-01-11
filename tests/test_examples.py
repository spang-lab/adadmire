import os
import adadmire
import numpy
import pathlib
import pytest


cwd = os.getcwd()
repo_dir = pathlib.Path(__file__).parent.parent


def example1():
    """Copied from docs/source/usage.md"""
    X = numpy.load('data/Feist_et_al/scaled_data_raw.npy')  # continuous data
    D = numpy.load('data/Feist_et_al/pheno.npy')  # discrete data
    levels = numpy.load('data/Feist_et_al/levels.npy')  # levels of discrete variables
    lam = adadmire.penalty(X, D, min=-2.25, max=-1.5, step=0.25)
    return adadmire.admire(X, D, levels, lam)


def example2():
    """Copied from docs/source/usage.md"""
    X = numpy.load('data/Higuera_et_al/scaled_data_raw.npy')  # continuous data
    D = numpy.load('data/Higuera_et_al/pheno.npy')  # discrete data
    levels = numpy.load('data/Higuera_et_al/levels.npy')  # levels of discrete variables
    X_ano, pos = adadmire.place_anomalies_continuous(X, n_ano=1360, epsilon=numpy.array([0.6, 0.8, 1.0, 1.2, 1.4]))
    lam = adadmire.penalty(X, D, min=-2.25, max=-1.5, step=0.25)
    return adadmire.admire(X_ano[2], D, levels, lam)


def example3():
    """Copied from docs/source/usage.md"""
    X = numpy.load('data/Higuera_et_al/data_na_scaled.npy')
    D = numpy.load('data/Higuera_et_al/pheno_na.npy')
    levels = numpy.load('data/Higuera_et_al/levels.npy')  # levels of discrete variables
    lam_zero = numpy.sqrt(numpy.log(X.shape[1] + D.shape[1]/2)/X.shape[0])
    lam_seq = numpy.array([-1.75, -2.0, -2.25])
    lam = [pow(2, x) for x in lam_seq]
    lam = numpy.array(lam)
    lam = lam_zero * lam
    return adadmire.impute(X, D, levels, lam)


@pytest.mark.slow
def test_example1():
    os.chdir(repo_dir)
    try:
        example1()
    finally:
        os.chdir(cwd)


@pytest.mark.slow
def test_example2():
    os.chdir(repo_dir)
    try:
        example2()
    finally:
        os.chdir(cwd)


@pytest.mark.slow
def test_example3():
    os.chdir(repo_dir)
    try:
        example3()
    finally:
        os.chdir(cwd)
