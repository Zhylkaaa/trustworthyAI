# coding=utf-8
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import numpy as np
from scipy.linalg import expm as matrix_exponential, schur
from scipy.sparse.linalg import expm_multiply
from scipy.spatial.distance import pdist
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.preprocessing import PolynomialFeatures

from numba import jit, prange
from numpy.linalg import norm


@jit(nopython=True, nogil=True, parallel=False)
def tr_exp_naive(X, m=200, k_max=10):
    V = np.random.choice(np.array([-1./m, 1./m], dtype=np.float32), size=(m, X.shape[0]))
    B = (V * m)@X
    res = np.sum(B * V)
    fk = 1
    for k in range(2, k_max):
        fk *= k
        B = B@X
        res += np.sum((B / fk) * V)
    del V, B
    return np.abs(res)


@jit(nopython=True, nogil=True, parallel=True)
def bfs_cycle(X, k=40):
    weights = np.zeros(X.shape[0], dtype=np.float32)

    for i in prange(X.shape[0]):
        path_weight = np.zeros(X.shape[0], dtype=np.float32)
        steps_reached = np.zeros(X.shape[0], dtype=np.int32)
        path_weight[i] = 0
        steps_reached[i] = 0
        head = 0
        tail = -1
        queue = np.zeros(X.shape[0], dtype=np.int32)
        queue[head] = i
        while tail < head:
            tail += 1
            v = queue[tail]
            if X[v, i] > 1e-3:
                weights[steps_reached[v]] += path_weight[v] + X[v, i]
                break
            for u in range(X.shape[0]):
                if steps_reached[u] == 0 and X[v, u] > 1e-3:
                    steps_reached[u] = steps_reached[v] + 1
                    path_weight[u] = path_weight[v] + X[v, u]
                    head += 1
                    queue[head] = u

    cyclicnes = np.float32(0.)
    for i in range(1, X.shape[0]):
        if weights[i] > 0:
            for j in range(i, k, i):
                cyclicnes += weights[i] * (j // i)
    return cyclicnes


def tr_scipy(X, m=200):
    V = np.random.choice(np.array([-1. / m, 1. / m], dtype=np.float32), size=(X.shape[0], m))
    tr = np.sum(V * expm_multiply(X, V))
    return np.abs(tr)


@jit(nopython=True, nogil=True)
def _lanczos_singe_vector_matrixes(A, v, k=10):
    k = min(k, v.shape[0])
    q_prev = np.zeros_like(v, dtype=np.float32)
    q = v / norm(v)
    beta = np.float32(0.)
    Q = np.zeros((v.shape[0], k), dtype=np.float32)
    T = np.zeros((k, k), dtype=np.float32)

    for i in range(k):
        Q[:, i] = q
        if i > 0:
            T[i - 1, i] = T[i, i - 1] = beta
        q_next = A@q - beta*q_prev
        alpha = q_next @ q
        T[i, i] = alpha
        q_next = q_next - alpha*q
        beta = norm(q_next)
        if beta < 1e-6:
            break
        q_prev = q
        q = q_next / beta

    return Q, T


def tr_krylov(A, m=200, k=10):
    V = np.random.choice(np.array([-1., 1.], dtype=np.float32), size=(m, A.shape[0]))
    # Qs, Ts = _lanczos_vectors_matrixes(A, V)
    res = 0.
    for v in V:
        Q, T = _lanczos_singe_vector_matrixes(A, v, k=k)
        res += (v * norm(v) / m) @ Q @ (matrix_exponential(T)[:, 0])
    return np.abs(res)


class get_Reward(object):

    _logger = logging.getLogger(__name__)

    def __init__(self, batch_num, maxlen, dim, inputdata, sl, su, lambda1_upper, 
                 score_type='BIC', reg_type='LR', l1_graph_reg=0.0, verbose_flag=True, exponent_type='original', **kwargs):
        self.batch_num = batch_num
        self.maxlen = maxlen # =d: number of vars
        self.dim = dim
        self.baseint = 2**maxlen
        self.d = {} # store results
        self.d_RSS = {} # store RSS for reuse
        self.inputdata = inputdata
        self.n_samples = inputdata.shape[0]
        self.l1_graph_reg = l1_graph_reg 
        self.verbose = verbose_flag
        self.sl = sl
        self.su = su
        self.lambda1_upper = lambda1_upper
        self.bic_penalty = np.log(inputdata.shape[0])/inputdata.shape[0]

        if score_type not in ('BIC', 'BIC_different_var'):
            raise ValueError('Reward type not supported.')
        if reg_type not in ('LR', 'QR', 'GPR'):
            raise ValueError('Reg type not supported')
        self.score_type = score_type
        self.reg_type = reg_type

        self.ones = np.ones((inputdata.shape[0], 1), dtype=np.float32)
        self.poly = PolynomialFeatures()
        self.exponent_type = exponent_type
        self.m = kwargs.get('m', 200)
        self.k = kwargs.get('k', 10)

    def cal_rewards(self, graphs, lambda1, lambda2):
        rewards_batches = []

        for graphi in graphs:
            reward_ = self.calculate_reward_single_graph(graphi, lambda1, lambda2)
            rewards_batches.append(reward_)

        return np.array(rewards_batches)


    ####### regression 

    def calculate_yerr(self, X_train, y_train):
        if self.reg_type == 'LR':
            return self.calculate_LR(X_train, y_train)
        elif self.reg_type == 'QR':
            return self.calculate_QR(X_train, y_train)
        elif self.reg_type == 'GPR':
            return self.calculate_GPR(X_train, y_train)
        else:
            # raise value error
            assert False, 'Regressor not supported'

    # faster than LinearRegression() from sklearn
    def calculate_LR(self, X_train, y_train):
        X = np.hstack((X_train, self.ones))
        XtX = X.T.dot(X)
        Xty = X.T.dot(y_train)
        theta = np.linalg.solve(XtX, Xty)
        y_err = X.dot(theta) - y_train
        return y_err

    def calculate_QR(self, X_train, y_train):
        X_train = self.poly.fit_transform(X_train)[:,1:]
        return self.calculate_LR(X_train, y_train)
    
    def calculate_GPR(self, X_train, y_train):
        med_w = np.median(pdist(X_train, 'euclidean'))
        gpr = GPR().fit(X_train/med_w, y_train)
        return y_train.reshape(-1,1) - gpr.predict(X_train/med_w).reshape(-1,1)

    ####### score calculations

    def calculate_reward_single_graph(self, graph_batch, lambda1, lambda2):
        graph_to_int = []
        graph_to_int2 = []

        for i in range(self.maxlen):
            graph_batch[i][i] = 0
            tt = np.int32(graph_batch[i])
            graph_to_int.append(self.baseint * i + int(''.join([str(ad) for ad in tt]), 2))
            graph_to_int2.append(int(''.join([str(ad) for ad in tt]), 2))

        graph_batch_to_tuple = tuple(graph_to_int2)

        if graph_batch_to_tuple in self.d:
            score_cyc = self.d[graph_batch_to_tuple]
            return self.penalized_score(score_cyc, lambda1, lambda2), score_cyc[0], score_cyc[1]

        RSS_ls = []

        for i in range(self.maxlen):
            col = graph_batch[i]
            if graph_to_int[i] in self.d_RSS:
                RSS_ls.append(self.d_RSS[graph_to_int[i]])
                continue

            # no parents, then simply use mean
            if np.sum(col) < 0.1:
                y_err = self.inputdata[:, i]
                y_err = y_err - np.mean(y_err)

            else:
                cols_TrueFalse = col > 0.5
                X_train = self.inputdata[:, cols_TrueFalse]
                y_train = self.inputdata[:, i]
                y_err = self.calculate_yerr(X_train, y_train)

            RSSi = np.sum(np.square(y_err))

            # if the regresors include the true parents, GPR would result in very samll values, e.g., 10^-13
            # so we add 1.0, which does not affect the monotoniticy of the score
            if self.reg_type == 'GPR':
                RSSi += 1.0

            RSS_ls.append(RSSi)
            self.d_RSS[graph_to_int[i]] = RSSi

        if self.score_type == 'BIC':
            BIC = np.log(np.sum(RSS_ls)/self.n_samples+1e-8) \
                  + np.sum(graph_batch)*self.bic_penalty/self.maxlen 
        elif self.score_type == 'BIC_different_var':
            BIC = np.sum(np.log(np.array(RSS_ls)/self.n_samples+1e-8)) \
                 + np.sum(graph_batch)*self.bic_penalty

        score = self.score_transform(BIC)
        if self.exponent_type == 'original':
            cycness = np.trace(matrix_exponential(np.array(graph_batch))) - self.maxlen
        elif self.exponent_type == 'trace_naive':
            cycness = tr_exp_naive(np.array(graph_batch, dtype=np.float32), m=self.m, k_max=self.k)
        elif self.exponent_type == 'trace_scipy':
            cycness = tr_scipy(np.array(graph_batch, dtype=np.float32), m=self.m)
        elif self.exponent_type == 'tr_krylov':
            cycness = tr_krylov(np.array(graph_batch, dtype=np.float32), m=self.m, k=self.k)
        elif self.exponent_type == 'tr_schur':
            cycness = np.sum(np.exp(np.diag(schur(np.array(graph_batch, dtype=np.float32))[0]))) - self.maxlen
        elif self.exponent_type == 'bfs':
            cycness = bfs_cycle(np.array(graph_batch, dtype=np.float32), k=self.k)
        else:
            raise ValueError('Unknown exponent type')
        reward = score + lambda1 * float(cycness>1e-5) + lambda2*cycness
            
        if self.l1_graph_reg > 0:
            reward = reward + self.l1_grapha_reg * np.sum(graph_batch)
            score = score + self.l1_grapha_reg * np.sum(graph_batch)

        self.d[graph_batch_to_tuple] = (score, cycness)

        if self.verbose:
            self._logger.info('BIC: {}, cycness: {}, returned reward: {}'.format(BIC, cycness, score))

        return reward, score, cycness

    #### helper
    
    def score_transform(self, s):
        return (s-self.sl)/(self.su-self.sl)*self.lambda1_upper

    def penalized_score(self, score_cyc, lambda1, lambda2):
        score, cyc = score_cyc
        return score + lambda1 * float(cyc > 1e-5) + lambda2 * cyc
    
    def update_scores(self, score_cycs, lambda1, lambda2):
        ls = []
        for score_cyc in score_cycs:
            ls.append(self.penalized_score(score_cyc, lambda1, lambda2))
        return ls
    
    def update_all_scores(self, lambda1, lambda2):
        score_cycs = list(self.d.items())
        ls = []
        for graph_int, score_cyc in score_cycs:
            ls.append((graph_int, (self.penalized_score(score_cyc, lambda1, lambda2), score_cyc[0], score_cyc[1])))
        return sorted(ls, key=lambda x: x[1][0])
