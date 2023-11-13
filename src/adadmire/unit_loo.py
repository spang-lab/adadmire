import adadmire
import random
import numpy as np
def test_loo():
    # generate data
    X = np.array([[1,1],[1,1]])
    D = np.array([[1,0],[1,0]])
    levels = np.array([2])
    lam = np.array([0.5])
    p_e = np.zeros([2,2]) + 3.7856652196880396e-05
    B_e = np.zeros([2,2]) + 20.51558176857432
    x_e = np.zeros([2,2]) + 0.09034876630988453
    d_e = np.array([0.9997587101440837, 0.00024128985591637892])
    d_e = np.vstack((d_e,d_e))
    prob, B, lam_o,  x, d = adadmire.loo_cv_cor(X,D,levels,lam, oIterations= 200)
    assert (prob == p_e).all() 
    assert (B == B_e).all() 
    assert (lam_o == lam).all() 
    assert (x == x_e).all() 
    assert (d == d_e).all() 
