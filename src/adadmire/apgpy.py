# Copyright (c) 2012-2013, Brendan O'Donoghue (bodonoghue85@gmail.com)
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# The views and conclusions contained in the software and documentation are those
# of the authors and should not be interpreted as representing official policies,
# either expressed or implied, of the FreeBSD Project, the Air Force Research
# Laboratory, or the U.S. Government.


# Note from Tobias Schmidt: this file contains source code from the package
# apgpy (https://github.com/bodono/apgpy). Package apgpy is not available via
# PyPI (https://pypi.org) and therefore cannot be resolved automatically by
# pip. We include its source code directly within this package to make
# installation via pip possible. This is explicictly allowed by the license of
# apgpy (see above).
from __future__ import print_function
import numpy as np
from functools import partial

class IWrapper:
    def dot(self, other):
        raise NotImplementedError("Implement in subclass")

    def __add__(self, other):
        raise NotImplementedError("Implement in subclass")

    def __sub__(self, other):
        raise NotImplementedError("Implement in subclass")

    def __mul__(self, scalar):
        raise NotImplementedError("Implement in subclass")

    def copy(self):
        raise NotImplementedError("Implement in subclass")

    def norm(self):
        raise NotImplementedError("Implement in subclass")

    @property
    def data(self):
        return self

    __rmul__ = __mul__


class NumpyWrapper(IWrapper):
    def __init__(self, nparray):
        self._nparray = nparray

    def dot(self, other):
        return np.inner(self.data, other.data)

    def __add__(self, other):
        return NumpyWrapper(self.data + other.data)

    def __sub__(self, other):
        return NumpyWrapper(self.data - other.data)

    def __mul__(self, scalar):
        return NumpyWrapper(self.data * scalar)

    def copy(self):
        return NumpyWrapper(np.copy(self.data))

    def norm(self):
        return np.linalg.norm(self.data)

    @property
    def data(self):
        return self._nparray

    __rmul__ = __mul__


def npwrap(x):
    if isinstance(x, np.ndarray):
        return NumpyWrapper(x)
    return x


def npwrapfunc(f, *args):
    return npwrap(f(*args))


def solve(grad_f, prox_h, x_init,
          max_iters=2500,
          eps=1e-6,
          alpha=1.01,
          beta=0.5,
          use_restart=True,
          gen_plots=False,
          quiet=False,
          use_gra=False,
          step_size=False,
          fixed_step_size=False,
          debug=False):

    df = partial(npwrapfunc, grad_f)
    ph = partial(npwrapfunc, prox_h)

    x_init = npwrap(x_init)

    x = x_init.copy()
    y = x.copy()
    g = df(y.data)
    theta = 1.

    if not step_size:
        # barzilai-borwein step-size initialization:
        t = 1. / g.norm()
        x_hat = x - t * g
        g_hat = df(x_hat.data)
        t = abs((x - x_hat).dot(g - g_hat) / (g - g_hat).norm() ** 2)
    else:
        t = step_size

    if gen_plots:
        errs = np.zeros(max_iters)

    k = 0
    err1 = np.nan
    iter_str = 'iter num %i, norm(Gk)/(1+norm(xk)): %1.2e, step-size: %1.2e'
    for k in range(max_iters):

        if not quiet and k % 100 == 0:
            print(iter_str % (k, err1, t))

        x_old = x.copy()
        y_old = y.copy()

        x = y - t * g

        if prox_h:
            x = ph(x.data, t)

        err1 = (y - x).norm() / (1 + x.norm()) / t

        if gen_plots:
            errs[k] = err1

        if err1 < eps:
            break

        if not use_gra:
            theta = 2. / (1 + np.sqrt(1 + 4 / (theta ** 2)))
        else:
            theta = 1.

        if not use_gra and use_restart and (y - x).dot(x - x_old) > 0:
            if debug:
                print('restart, dg = %1.2e' % (y - x).dot(x - x_old))
            x = x_old.copy()
            y = x.copy()
            theta = 1.
        else:
            y = x + (1 - theta) * (x - x_old)

        g_old = g.copy()
        g = df(y.data)

        # tfocs-style backtracking:
        if not fixed_step_size:
            t_old = t
            t_hat = 0.5 * ((y - y_old).norm() ** 2) / \
                abs((y - y_old).dot(g_old - g))
            t = min(alpha * t, max(beta * t, t_hat))
            if debug:
                if t_old > t:
                    print('back-track, t = %1.2e, t_old = %1.2e, t_hat = %1.2e' %
                          (t, t_old, t_hat))

    if not quiet:
        print(iter_str % (k, err1, t))
        print('terminated')
    if gen_plots:
        import matplotlib.pyplot as plt
        errs = errs[1:k]
        plt.figure()
        plt.semilogy(errs[1:k])
        plt.xlabel('iters')
        plt.title('||Gk||/(1+||xk||)')
        plt.draw()

    return x.data
