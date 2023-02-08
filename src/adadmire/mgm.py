# MIT License
# Copyright (c) 2019 Michael Altenbuchinger and Helena Zacharias
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import adadmire.apgpy as apg

def grad_neglogli(B, Rho, Phi, alphap, alphaq, X, D, levels):
    n = X.shape[0]
    p = B.shape[0]
    q = levels.shape[0]

    levelSum = [0]
    levelSum.extend(levels)
    levelSum = np.cumsum(levelSum)
    for r in range(0, q):
        Phi[int(levelSum[r]):int(levelSum[r]+levels[r]),
            int(levelSum[r]):int(levelSum[r]+levels[r])] = 0

    Bd = np.diag(B)
    B = B - np.diag(Bd)
    B = np.triu(B)
    B = B + np.transpose(B)
    DRho = np.dot(D, Rho)
    DRho = np.dot(DRho, np.diag(np.divide(np.repeat(1., p), Bd)))
    XB = np.dot(X, B)
    XB = np.dot(XB, np.diag(np.divide(np.repeat(1., p), Bd)))
    consts = np.tile(alphap, (n, 1))
    Xt = np.transpose(X)
    res = consts + DRho + XB + X
    gradBd = np.repeat(0., p)

    for s in range(0, p):
        gradBd[s] = - n/(2.*Bd[s]) - .5*np.dot(res[:, s], res[:, s]
                                              ) + np.dot(res[:, s], XB[:, s] +
                                                         DRho[:, s])

    gradB = - np.dot(Xt, res)
    gradB = gradB - np.diag(np.diag(gradB))
    gradB = np.transpose(np.tril(gradB)) + np.triu(gradB)
    gradalphap = - np.dot(np.diag(Bd), np.sum(res, axis=0)[:, np.newaxis])
    gradalphap = gradalphap[:, 0]

    gradRho = - np.dot(np.transpose(D), res)

    Xt = np.transpose(X)
    RhoX = np.dot(Rho, Xt)
    Phi = Phi - np.diag(np.diag(Phi))
    Phi = np.triu(Phi)
    Phi = Phi + np.transpose(Phi)
    Phirr = np.transpose(np.tile(alphaq, (n, 1)))
    PhiD = np.dot(Phi, np.transpose(D))
    discprod = np.transpose(RhoX+Phirr+PhiD)

    for r in range(0, q):
        disctemp = discprod[:, int(levelSum[r]):int(levelSum[r]+levels[r])]
        denominator = np.logaddexp.reduce(disctemp, axis=1)
        disctemp = disctemp - denominator[:, np.newaxis]
        disctemp = np.exp(disctemp)
        temp = disctemp - D[:, int(levelSum[r]):int(levelSum[r]+levels[r])]
        discprod[:, int(levelSum[r]):int(levelSum[r]+levels[r])] = temp

    gradalphaq = np.sum(discprod, axis=0)
    gradw = np.dot(Xt, discprod)
    gradRho = gradRho+np.transpose(gradw)
    gradPhi = np.dot(np.transpose(D), discprod)

    for r in range(0, q):
        gradPhi[int(levelSum[r]):int(levelSum[r]+levels[r]),
                int(levelSum[r]):int(levelSum[r]+levels[r])] = 0

    gradPhi = np.transpose(np.tril(gradPhi))+np.triu(gradPhi)
    gradB.flat[::p+1] = gradBd
    gradB = gradB/n
    gradRho = gradRho/n
    gradPhi = gradPhi/n
    gradalphap = gradalphap/n
    gradalphaq = gradalphaq/n
    return gradB, gradRho, gradPhi, gradalphap, gradalphaq


def neglogli(B, Rho, Phi, alphap, alphaq, X, D, levels):
    n = X.shape[0]
    p = B.shape[0]
    q = levels.shape[0]
    Bd = np.diag(B)
    B = B - np.diag(Bd)
    B = np.triu(B)
    B = B + np.transpose(B)
    DRho = np.dot(D, Rho)
    DRho = np.dot(DRho, np.diag(np.divide(np.repeat(1, p), Bd)))
    XB = np.dot(X, B)
    XB = np.dot(XB, np.diag(np.divide(np.repeat(1, p), Bd)))
    Xt = np.transpose(X)
    RhoX = np.dot(Rho, Xt)
    Phi = Phi - np.diag(np.diag(Phi))
    Phi = np.triu(Phi)
    Phi = Phi + np.transpose(Phi)
    Phirr = np.transpose(np.tile(alphaq, (n, 1)))
    PhiD = np.dot(Phi, np.transpose(D))
    levelSum = [0]
    levelSum.extend(levels)
    levelSum = np.cumsum(levelSum)
    consts = np.tile(alphap, (n, 1))
    PLcont1 = -n/2.*np.sum(np.log(-Bd)) #here, the constant term is neglected
    PLcont2 = consts + DRho + XB + X
    PLcont2 = np.dot(PLcont2, np.diag(np.sqrt(-Bd)))
    PLcont2 = np.multiply(PLcont2, PLcont2)
    PLcont2 = 0.5*np.sum(np.sum(PLcont2, axis=0))
    temp = RhoX+Phirr+PhiD
    PLdisc = 0
    for r in range(0, q):
        temp2 = temp[int(levelSum[r]):int(levelSum[r]+levels[r]), :]
        denominator = np.sum(
            np.exp(np.dot(np.identity(int(levels[r])), temp2)), axis=0)
        numerator = np.sum(np.multiply(D[:, int(levelSum[r]):int(
            levelSum[r]+levels[r])], np.transpose(temp2)), axis=1)
        PLdisc = PLdisc-numerator+np.log(denominator)
    PLdisc = np.sum(PLdisc)
    return((PLcont1+PLcont2+PLdisc)/n)


def neglogli_plain(B_Rho_Phi_alphap_alphaq, X, D, levels, p, q):
    x2 = Inv_B_Rho_Phi_alphap_alphaq(B_Rho_Phi_alphap_alphaq, p, q)
    x1 = neglogli(x2[0], x2[1], x2[2], x2[3], x2[4], X, D, levels)
    return(x1)


def B_Rho_Phi_alphap_alphaq(B, Rho, Phi, alphap, alphaq):
    p = B.shape[0]
    q = Phi.shape[0]
    sizes = np.cumsum([p*p, p*q, q*q, p, q])
    x = np.repeat(0., sizes[4])
    x[0:sizes[0]] = np.reshape(B, (1, p*p))[0, :]
    x[sizes[0]:sizes[1]] = np.reshape(Rho, (1, p*q))[0, :]
    x[sizes[1]:sizes[2]] = np.reshape(Phi, (1, q*q))[0, :]
    x[sizes[2]:sizes[3]] = alphap
    x[sizes[3]:sizes[4]] = alphaq
    return(x)


def Inv_B_Rho_Phi_alphap_alphaq(x, p, q):
    sizes = np.cumsum([p*p, p*q, q*q, p, q])
    B = np.reshape(x[0:sizes[0]], (p, p))
    Rho = np.reshape(x[sizes[0]:sizes[1]], (q, p))
    Phi = np.reshape(x[sizes[1]:sizes[2]], (q, q))
    alphap = x[sizes[2]:sizes[3]]
    alphaq = x[sizes[3]:sizes[4]]
    return(B, Rho, Phi, alphap, alphaq)


def grad_neglogli_plain(B_Rho_Phi_alphap_alphaq, X, D, levels, p, q):
    x2 = Inv_B_Rho_Phi_alphap_alphaq(B_Rho_Phi_alphap_alphaq, p, q)
    x1 = grad_neglogli(x2[0], x2[1], x2[2], x2[3], x2[4], X, D, levels)
    return(x1[0], x1[1], x1[2], x1[3], x1[4])


def make_starting_parameters(X, D, levels):
    p = X.shape[1]
    q = D.shape[1]
    B = -np.diag(np.repeat(1, p))
    Rho = np.zeros((q, p))
    Phi = np.zeros((q, q))
    alphap = np.zeros(p)
    alphaq = np.zeros(q)
    return(B, Rho, Phi, alphap, alphaq, p, q)


def grad_f_temp(x, X, D, levels, p, q):
    x2 = grad_neglogli_plain(x, X, D, levels, p, q)
    x3 = B_Rho_Phi_alphap_alphaq(x2[0], x2[1], x2[2], x2[3], x2[4])
    return(x3)


def prox_enet(x, l_l1, l_l2, t, pen, p0, tol0):
    prox_l1 = np.sign(x) * np.maximum(abs(x) - t * l_l1 * pen, 0)
    return prox_l1 / (1. + t * l_l2 * pen)


def make_penalty_factors(X, D, levels):
    levelSum = [0]
    levelSum.extend(levels)
    levelSum = np.cumsum(levelSum)
    p = X.shape[1]
    qL = len(levels)
    q = np.sum(levels)
    sds = np.reshape(np.std(X, axis=0), (p, 1))
    ps = np.reshape(np.mean(D, axis=0), (q, 1))
    B = np.ones((p, p)) - np.diag(np.repeat(1, p))
    B = np.multiply(B, np.dot(sds, np.transpose(sds)))
    Rho = np.ones((q, p))
    Phi = np.ones((q, q))
    alphap = np.zeros(p)
    alphaq = np.zeros(q)
    for r in range(0, qL):
        Phi[int(levelSum[r]), :] = np.repeat(10, Phi.shape[1])
        Phi[:, int(levelSum[r])] = np.repeat(10, Phi.shape[1])
        ps_t = ps[int(levelSum[r]):int(levelSum[r]+levels[r])]
        ps[int(levelSum[r]):int(levelSum[r]+levels[r])
           ] = np.sqrt(np.sum(ps_t*(1-ps_t)))
        Rho[int(levelSum[r]), :] = np.repeat(10, Rho.shape[1])
    Rho = np.multiply(Rho, np.dot(ps, np.transpose(sds)))
    Phi = np.multiply(Phi, np.dot(ps, np.transpose(ps)))
    return(B, Rho, Phi, alphap, alphaq, p, q)


def Fit_MGM(X, D, levels, lambda_seq, iterations, eps=1e-6):
    p = X.shape[1]
    q = D.shape[1]
    start = make_starting_parameters(X, D, levels)
    start = B_Rho_Phi_alphap_alphaq(
        start[0], start[1], start[2], start[3], start[4])
    penalty = make_penalty_factors(X, D, levels)
    penalty = B_Rho_Phi_alphap_alphaq(
        penalty[0], penalty[1], penalty[2], penalty[3], penalty[4])
    x_start = np.zeros(penalty.shape[0])

    def grad_f(x): return grad_f_temp(x + x_start, X, D, levels, p, q)

    def fun(x): return neglogli_plain(x + x_start, X, D, levels, p, q)
    x_list = [None]*lambda_seq.shape[0]
    xtemp = start
    for i in range(0, lambda_seq.shape[0]):
        l_l1 = lambda_seq[i]
        print("lambda =", l_l1)

        def prox_g(x, l): return prox_enet(
            x, l_l1, 0, t=l, pen=penalty, p0=p, tol0=eps)
        x_fista = apg.solve(grad_f, prox_g, xtemp, eps=eps,
                            max_iters=iterations, gen_plots=False,
                            debug=False)
        x_list[i] = Inv_B_Rho_Phi_alphap_alphaq(x_fista, p, q)
        xtemp = x_fista
    loss_list = []
    k = 0
    for j in range(0, len(x_list)):
        k = k+1
        loss_vec = neglogli(
            x_list[j][0], x_list[j][1], x_list[j][2], x_list[j][3],
            x_list[j][4], X, D, levels)
        loss_list = np.append(loss_list, loss_vec)
    return([x_list, loss_list])
