from scipy.stats import norm
from adadmire.mgm import *
import numpy as np
import random
from sklearn.metrics.pairwise import nan_euclidean_distances

__version__ = "1.0.0"


def pred_continuous(B, Rho, alphap, D_pred, X_pred):
    """Predict continuous values given estimated
    parameters of MGM and discrete values.

    Args:
        - B (numpy.ndarray): Matrix B.
        - Rho (numpy.ndarray): Continuous-discrete couplings.
        - alphap (numpy.ndarray): Continuous node potentials.
        - D_pred (numpy.ndarray): Vector of discrete states.
        - X_pred (numpy.ndarray): Vector of continuous values that should be estimated.

    Returns:
        numpy.ndarray: Predicted continuous values.
    """
    ind = np.array(D_pred) == 1
    r_sum = np.array([])
    for k in range(0, Rho.shape[1]):
        tmp = sum(Rho[ind, k])
        r_sum = np.append(r_sum, tmp)

    x_b = np.array([])
    for k in range(0, B.shape[0]):
        tmp = np.dot(B[k], X_pred)
        tmp = tmp - B[k, k] * X_pred[k]
        x_b = np.append(x_b, tmp)

    x_hat = -(alphap + r_sum + x_b) / (0.5 * np.diag(B))
    return x_hat


def pred_discrete(Rho, X_pred, D_pred, alphaq, Phi, levels, p):
    """Predict probabilities of observing states D_pred given estimated
    parameters of MGM and continuous values.

    Args:
        Rho (numpy.ndarray): Continuous-discrete couplings.
        X_pred (numpy.ndarray): Vector of continuous values.
        D_pred (numpy.ndarray): Vector of discrete states for which probabilities should be calculated.
        alphaq (numpy.ndarray): Continuous node potentials.
        Phi (numpy.ndarray): Discrete-discrete couplings.
        levels (list): Levels for discrete states.
        p (int): Number of continuous features.

    Returns:
        numpy.ndarray: Predicted probabilities of observing discrete states D_pred.
    """
    levels = levels.flatten()
    lSum = [0]
    lSum.extend(levels)
    lSum = np.cumsum(lSum)
    q = len(levels)
    q0 = np.sum(levels)
    prob_cat = np.array([])

    for j in range(q):
        D_del = np.array(D_pred)
        D_del[lSum[j]:lSum[j + 1]] = 0.
        t1 = Rho[lSum[j]:lSum[j + 1], :] @ X_pred.reshape((p, 1))
        t2 = alphaq[lSum[j]:lSum[j + 1]].reshape((levels[j], 1))
        t3 = Phi[lSum[j]:lSum[j + 1], :] @ D_del.reshape((q0, 1))
        denom = np.sum(np.exp(t1 + t2 + t3))
        numer = np.exp(t1 + t2 + t3)
        tmp = numer.flatten() / denom
        prob_cat = np.append(prob_cat, tmp)

    return prob_cat


def calc_mean(X, D):
    """Calculate the group mean of the continuous data matrix X according to states D.

    Args:
        - X (numpy.ndarray): continuous Data matrix, features in columns, samples in rows.
        - D (numpy.ndarray): Discrete states matrix in one-hot-encoding, features in columns, samples in rows.

    Returns:
        numpy.ndarray: Group mean values.
    """
    mean = np.zeros(X.shape)
    for i in range(X.shape[0]):
        ind = np.where((D == D[i]).all(axis=1))
        mean[i] = np.mean(X[ind], axis=0)
    return mean


def loo_cv_cor(X, D, levels, lambda_seq, oIterations=10000, oTol=1e-6, t=0.05):
    """
    Estimate continuous matrix X and discrete states D in a leave-one-out cross-validation approach using Mixed Graphical Models.

    Args:
        - X (numpy.ndarray): Continuous Data matrix, features in columns, samples in rows.
        - D (numpy.ndarray): Discrete states matrix in one-hot-encoding, features in columns, samples in rows.
        - levels (numpy.ndarray): List of levels for discrete states.
        - lambda_seq: (numpy.ndarray): Sequence of penalty values.
        - oIterations (int, optional): Number of iterations for fitting MGMs. Defaults to 10000.
        - oTol (float, optional): Tolerance for fitting MGMs. Defaults to 1e-6.
        - t (float, optional): Probability threshold value for smoothing corrections. Defaults to 0.05.

    Returns:
        tuple: Tuple containing the following elements:
            - prob_cont_old (numpy.ndarray): Estimated continuous probabilities.
            - Var_old (numpy.ndarray): Variances of the continuous estimates.
            - lam_opt_old (numpy.ndarray): Optimal Lambda.
            - X_hat_cor_xp_old (numpy.ndarray): Predicted continuous data matrix X.
            - D_hat_cor_xp_old (numpy.ndarray): Predicted discrete state matrix D.
    """
    means = calc_mean(X, D)
    MSE = 2e10
    MSE_old = 2e10
    j = 0
    X_hat = np.zeros(shape=X.shape)
    X_hat_cor_xp = np.zeros(shape=X.shape)
    D_hat = np.zeros(shape=D.shape)
    D_hat_cor_xp = np.zeros(shape=D.shape)
    prob_cont = np.zeros(shape=X.shape)
    Var = np.zeros(shape=X.shape)
    MSE_seq = np.array([])
    lam_opt = np.array([lambda_seq[0]])
    lam_opt_old = np.array([lambda_seq[0]])
    while ((MSE_old >= MSE) and (j < np.size(lambda_seq))):
        lam_opt_old = np.copy(lam_opt)
        MSE_old = np.copy(MSE)
        X_hat_cor_xp_old = np.copy(X_hat_cor_xp)
        D_hat_old = np.copy(D_hat)
        D_hat_cor_xp_old = np.copy(D_hat_cor_xp)
        prob_cont_old = np.copy(prob_cont)
        Var_old = np.copy(Var)
        lambda_seq_t = np.array([lambda_seq[j]])
        # loop over all samples
        for i in range(X.shape[0]):
            # sample which has to be predicted
            print('Sample Number:', i)
            X_pred = X[i]
            D_pred = D[i]
            # rest of the samples
            X_red = np.delete(X, i, 0)
            D_red = np.delete(D, i, 0)
            # learn models for all other samples and current lambda_seq
            Res = Fit_MGM(X_red, D_red, levels, lambda_seq_t,
                          oIterations, eps=oTol)
            Res = Res[0]
            # predict sample which has been left out
            B = Res[0][0]
            B = B + np.transpose(B)
            Phi = Res[0][2]
            Phi = Phi + np.transpose(Phi)
            alphap = Res[0][3]
            Rho = Res[0][1]
            alphaq = Res[0][4]
            p = B.shape[0]
            # predict sample
            x_hat = pred_continuous(B, Rho, alphap, D_pred, X_pred)
            # calculate probabilities
            dev = 0.5 * abs(np.diag(B))
            eps = abs(X_pred - x_hat)
            p_val = 2 * norm.cdf(x_hat - eps, loc=x_hat,
                                 scale=np.sqrt(1 / dev))
            # if p_val for data point < t replace it in other predictions by mean of feature
            x_cor = np.where(p_val < t, means[i], X_pred)
            # get discrete predictions
            d_hat = pred_discrete(Rho, X_pred, D_pred, alphaq, Phi, levels, p)
            # if probability of discrete data point < t change state to most likely one
            d_hat_cor = np.copy(d_hat)
            levelSum = np.cumsum(levels)
            levelSum = np.insert(levelSum, 0, 0)
            d_var = 0
            D_pred_cor = np.copy(D_pred)
            for k in range(len(d_hat)):
                if k == levelSum[d_var]:
                    d_var = d_var + 1
                if D_pred[k] == 1 and d_hat[k] < t:
                    tmp = np.zeros(
                        shape=len(d_hat[levelSum[d_var-1]:levelSum[d_var]]))
                    tmp[np.where(d_hat[levelSum[d_var-1]:levelSum[d_var]]
                                 == max(d_hat[levelSum[d_var-1]:levelSum[d_var]]))] = 1
                    D_pred_cor[levelSum[d_var-1]:levelSum[d_var]] = tmp

            # predict continuous and discrete again using corrected states and values
            x_hat_cor_xp = pred_continuous(B, Rho, alphap, D_pred_cor, x_cor)
            d_hat_cor_xp = pred_discrete(
                Rho, x_cor, D_pred_cor, alphaq, Phi, levels, p)
            X_hat[i] = x_hat
            X_hat_cor_xp[i] = x_hat_cor_xp
            D_hat[i] = d_hat
            D_hat_cor_xp[i] = d_hat_cor_xp
            eps = abs(X_pred - x_hat)
            p_val = 2 * norm.cdf(x_hat - eps, loc=x_hat,
                                 scale=np.sqrt(1 / dev))
            prob_cont[i] = p_val
            Var[i] = dev
        MSE = np.mean((X_hat - X)**2)
        MSE_seq = np.append(MSE_seq, MSE)
        lam_opt = lambda_seq_t
        j = j+1

    if (MSE < MSE_old):
        print('Minimum located at end of Lambda sequence. Generate new sequence with smaller end point.')
    if (j == 2):
        print('Minimum located at start of Lambda sequence. Generate new sequence with bigger start point.')
    if (np.size(lambda_seq) == 1):  # in case only one lambda_seq submitted
        X_hat_cor_xp_old = np.copy(X_hat_cor_xp)
        D_hat_cor_xp_old = np.copy(D_hat_cor_xp)
        prob_cont_old = np.copy(prob_cont)
        Var_old = np.copy(Var)

    return (prob_cont_old, Var_old, lam_opt_old, X_hat_cor_xp_old, D_hat_cor_xp_old)


def get_threshold_continuous(X, X_hat, dev):
    """
    Calculate the threshold for continuous anomaly detection and correct matrix X accordingly.

    Args:
        - X (numpy.ndarray): Continuous data matrix, features in columns, samples in rows.
        - X_hat (numpy.ndarray): Predicted continuous data matrix X.
        - dev (numpy.ndarray): Variances of the continuous estimates.

    Returns:
        Tuple: Tuple containing the following elements:
            - X_cor (numpy.ndarray): Data matrix X corrected for anomalies.
            - threshold (float): Threshold value for anomaly detection.
            - n_ano (int): Number of detected anomalies.
            - pos (numpy.ndarray): Indices of detected anomalies in matrix X.
    """
    np.random.seed(seed=671)
    mu = X_hat.flatten(order='F')
    dev = dev.flatten(order='F')
    X_flat = X.flatten(order='F')
    random_scores = np.zeros(shape=(mu.shape[0], 100))

    for i in range(mu.shape[0]):
        tmp = norm.rvs(loc=mu[i], scale=np.sqrt(1/dev[i]), size=100)
        random_scores[i, ] = np.divide(
            np.abs(tmp - mu[i]), np.sqrt(1/dev[i]))

    random_scores = np.sort(random_scores, axis=0)
    random_scores = np.flip(np.mean(random_scores, axis=1))
    observed_scores = np.abs(mu - X_flat) / np.sqrt(1/dev)
    observed_scores_sorted = np.flip(np.sort(observed_scores))
    threshold = random_scores[np.where(
        random_scores >= observed_scores_sorted)][0]
    n_ano = np.sum(observed_scores >= threshold)
    observed_scores = np.reshape(observed_scores, X.shape, 'F')
    X_cor = np.where(observed_scores <= threshold, X, X_hat)
    pos = np.transpose(
        (observed_scores >= threshold).nonzero())

    return (X_cor, threshold, n_ano, pos)


def get_threshold_discrete(D, levels, D_hat):
    """
    Calculate the threshold for discrete anomaly detection and determine the number of discrete anomalies.

    Args:
        - D (numpy.ndarray): Discrete states matrix in one-hot-encoding, features in columns, samples in rows.
        - levels (numpy.ndarray): List of levels for discrete states.
        - D_hat (numpy.ndarray): Predicted discrete state matrix D.

    Returns:
        Tuple: Tuple containing the following elements:
            - n_ano (int): Number of detected anomalies.
            - threshold (float): Threshold value for anomaly detection.
            - pos (numpy.ndarray): Indices of detected anomalies in matrix X.
    """
    np.random.seed(seed=321)
    ind = np.where(D == 1)
    p = D_hat[ind]
    observed_scores = -np.log(p)
    observed_scores_sorted = np.flip(np.sort(observed_scores))
    random_scores = [np.zeros(shape=(D.shape[0], 100))
                     for i in range(len(levels))]
    start = 0

    for i in range(len(levels)):
        end = int(start + levels[i])
        prob = D_hat[:, start:end]

        state_index = np.arange(levels[i])
        for j in range(D.shape[0]):
            random_obs = np.random.choice(
                state_index, p=prob[j], size=100)
            random_scores[i][j] = -np.log(prob[j, random_obs])
        start = end

    random_scores = np.concatenate((random_scores), axis=0)
    random_scores = np.sort(random_scores, axis=0)
    random_scores = np.flip(np.mean(random_scores, axis=1))
    threshold = random_scores[np.where(
        random_scores >= observed_scores_sorted)][0]
    n_ano = np.sum(observed_scores >= threshold)
    pos = np.array([ind[0][np.where(observed_scores >= threshold)],
                   ind[1][np.where(observed_scores >= threshold)]])
    return (n_ano, threshold, pos)


def transform_data(X):
    """
    Transform data to [0,1] range using Min-max transformation.

    Args:
        - X (numpy.ndarray): Continuous data matrix, features in columns, samples in rows.

    Returns:
        numpy.ndarray: Continuous data matrix X transformed to [0,1] range.
    """
    X_trans = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    return X_trans


def transform_back(X, X_scaled):
    """
    Transform data back to original scale.

    Args:
        - X (numpy.ndarray): Original continuous data matrix.
        - X_scaled (numpy.ndarray): Continuous data matrix X transformed to [0,1] range.

    Returns:
        numpy.ndarray: Continuous data matrix X retransformed to original scale.
    """
    X_back = X_scaled * (X.max(axis=0) - X.min(axis=0)) + X.min(axis=0)
    return X_back


def rel_dev(x, org):
    """
    Calculate relative deviation of x from org.

    Args:
        - x (float): Value to be compared.
        - org (float): Original value.

    Returns:
        float: Relative deviation.
    """
    return abs((x - org) / org)


def place_anomalies_continuous(X, n_ano, epsilon, positive=False):
    """Place anomalies in continuous data matrix X.

    Args:
        - X (numpy.ndarray): Continuous data matrix, features in columns, samples in rows.
        - n_ano (int): Number of anomalies to be placed.
        - epsilon (list): List of anomaly strengths. For each entry one simulation is generated.
        - positive (bool, optional): If True, ensures anomalies are positive. Defaults to False.

    Returns:
        tuple: Tuple containing the following elements:
            - list: List of matrices with placed anomalies.
            - numpy.ndarray: Positions of placed anomalies.
    """
    random.seed(987)
    # first transform data feature-wise to [0,1]
    X_scaled = transform_data(X)
    # Calculate 15% border
    Z = (0.15*X) / (X.max(axis=0) - X.min(axis=0))
    ano = [np.copy(X) for i in range(len(epsilon))]
    change = np.copy(X)
    dirm = np.copy(X)
    # for each element in X sample shift
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            lc = random.uniform(Z[i, j], (Z[i, j]+X_scaled[i, j]))
            uc = random.uniform(Z[i, j], (1+Z[i, j]-X_scaled[i, j]))
            # decide whether lower or upper change
            dir = random.randint(0, 1)
            if dir == 1:
                for k in range(len(epsilon)):
                    ano[k][i, j] = X_scaled[i, j] + epsilon[k]*uc
                    change[i, j] = uc
                    dirm[i, j] = 1
            else:
                for k in range(len(epsilon)):
                    ano[k][i, j] = X_scaled[i, j] - epsilon[k] * lc
                    change[i, j] = lc
                    dirm[i, j] = 0
    # transform anomalies back
    ano_retrans = [transform_back(X, tmp) for tmp in ano]
    # place anomalies
    position = np.array([])
    position.shape = (0, 2)
    X_ano = [np.copy(X) for i in range(len(epsilon))]
    k = 0
    while k < n_ano:
        # sample position
        row = random.randint(0, (X.shape[0]-1))
        col = random.randint(0, (X.shape[1]-1))
        # first check if anomaly already has been placed at that position
        if sum((position == [[row, col]]).all(axis=1)) == 0:
            # check if introduced anomaly > 15% deviation (only for epsilon < 1 relevant) and anomaly still positive
            if positive == True:
                if rel_dev(ano_retrans[0][row, col], X[row, col]) > 0.15 and ano_retrans[(len(epsilon)-1)][row, col] > 0:
                    k = k+1
                    position = np.append(position, [[row, col]], axis=0)
                    for l in range(len(epsilon)):
                        X_ano[l][row, col] = ano_retrans[l][row, col]
            else:
                if rel_dev(ano_retrans[0][row, col], X[row, col]) > 0.15:
                    k = k+1
                    position = np.append(position, [[row, col]], axis=0)
                    for l in range(len(epsilon)):
                        X_ano[l][row, col] = ano_retrans[l][row, col]
    return (X_ano, position)


def impute(X, D, levels, lambda_seq, oIterations=10000, oTol=1e-6):
    """Impute missing values in the data using MGM.

    Args:
        - X (numpy.ndarray): Continuous Data matrix with missing values (np.nan), features in columns, samples in rows.
        - D (numpy.ndarray): Discrete states matrix in one-hot-encoding with missing values (np.nan), features in columns, samples in rows.
        - levels (numpy.ndarray): List of levels for discrete states.
        - lambda_seq: (numpy.ndarray): Sequence of penalty values.
        - oIterations (int, optional): Number of iterations for fitting MGMs. Defaults to 10000.
        - oTol (float, optional): Tolerance for fitting MGMs. Defaults to 1e-6.

    Returns:
        tuple: Tuple containing the following elements:
            - numpy.ndarray: Imputed continuous data.
            - numpy.ndarray: Imputed discrete data.
            - float: Optimal lambda_seq value used for imputation.
    """
    # calculate Euclidean distance of all samples to each other
    # ignore NaN values
    dist = nan_euclidean_distances(X, X)
    np.fill_diagonal(dist, np.inf)
    X_preimp = np.copy(X)
    D_preimp = np.copy(D)

    for i in range(X.shape[0]):
        ind_disc = np.where(np.isnan(D[i]))[0]
        ind_cont = np.where(np.isnan(X[i]))[0]
        # impute continuous with value of closest sample
        for j in ind_cont:
            sample = np.argmin(dist[i])
            dist_new = np.copy(dist)
            while np.isnan(X[sample, j]):  # check if value for imputation is also nan
                dist_new[i, sample] = np.inf
                # use second, third, .. closest sample
                sample = np.argmin(dist_new[i])
            X_preimp[i, j] = X[sample, j]
        # same for discrete
        for j in ind_disc:
            sample = np.argmin(dist[i])
            dist_new = np.copy(dist)
            while np.isnan(D[sample, j]):  # check if value for imputation is also nan
                dist_new[i, sample] = np.inf
                # use second, third, .. closest sample
                sample = np.argmin(dist_new[i])
            D_preimp[i, j] = D[sample, j]

    # for each lam in lambda_seq fit MGM on all data
    # calculate MSE

    MSE = 2e10
    MSE_old = 2e10
    j = 0
    X_hat = np.zeros(shape=X.shape)
    D_hat = np.zeros(shape=D.shape)
    MSE_seq = np.array([])
    lam_opt = np.array([lambda_seq[0]])
    lam_opt_old = np.array([lambda_seq[0]])
    while ((MSE_old >= MSE) and (j < np.size(lambda_seq))):
        lam_opt_old = np.copy(lam_opt)
        MSE_old = np.copy(MSE)
        X_hat_old = np.copy(X_hat)
        D_hat_old = np.copy(D_hat)
        lambda_seq_t = np.array([lambda_seq[j]])
        # fit model on all samples
        Res = Fit_MGM(X_preimp, D_preimp, levels,
                      lambda_seq_t, oIterations, eps=oTol)
        Res = Res[0]
        B = Res[0][0]
        B = B + np.transpose(B)
        Phi = Res[0][2]
        Phi = Phi + np.transpose(Phi)
        alphap = Res[0][3]
        Rho = Res[0][1]
        alphaq = Res[0][4]
        p = B.shape[0]
        # loop over all samples
        for i in range(X.shape[0]):
            X_pred = X_preimp[i]
            D_pred = D_preimp[i]
            # predict sample
            x_hat = pred_continuous(B, Rho, alphap, D_pred, X_pred)
            # get discrete predictions
            d_hat = pred_discrete(Rho, X_pred, D_pred, alphaq, Phi, levels, p)
            levelSum = np.cumsum(levels)
            levelSum = np.insert(levelSum, 0, 0)
            d_var = 0
            d_hat_state = np.copy(D_pred)
            for k in range(len(d_hat)):
                if k == levelSum[d_var]:
                    d_var = d_var + 1
                    tmp = np.zeros(
                        shape=len(d_hat[levelSum[d_var-1]:levelSum[d_var]]))
                    tmp[np.where(d_hat[levelSum[d_var-1]:levelSum[d_var]]
                                 == max(d_hat[levelSum[d_var-1]:levelSum[d_var]]))] = 1
                    d_hat_state[levelSum[d_var-1]:levelSum[d_var]] = tmp
            X_hat[i] = x_hat
            D_hat[i] = d_hat_state
        MSE = np.mean((X_hat - X_preimp)**2)
        MSE_seq = np.append(MSE_seq, MSE)
        lam_opt = lambda_seq_t
        j = j+1
    ind_cont = np.where(np.isnan(X))
    ind_disc = np.where(np.isnan(D))
    X_imp = np.copy(X)
    X_imp[ind_cont] = X_hat_old[ind_cont]
    D_imp = np.copy(D)
    D_imp[ind_disc] = D_hat_old[ind_disc]
    return (X_imp, D_imp, lam_opt_old)


def penalty(X, D, min, max, step):
    """Calculate a sequence of penalty parameters for regularization in the form of scaled exponentials.

    Args:
        - X (numpy.ndarray): Continuous Data matrix, features in columns, samples in rows.
        - D (numpy.ndarray): Discrete states matrix in one-hot-encoding, features in columns, samples in rows.
        - min (float): The minimum exponent for the penalty parameter.
        - max (float): The maximum exponent for the penalty parameter.
        - step (float): The step size between consecutive exponents.

    Returns:
        - lambda_seq: (numpy.ndarray): Sequence of penalty values.
    """
    lam_zero = np.sqrt(np.log(X.shape[1] + D.shape[1]/2)/X.shape[0])
    seq = np.flip(np.arange(min, max, step))
    lambda_seq = [pow(2, x) for x in seq]
    lambda_seq = np.array(lambda_seq)
    lambda_seq = lam_zero * lambda_seq
    return lambda_seq


def admire(X, D, levels, lam, oIterations=10000, oTol=1e-6, t=0.05):
    """
    Detect and correct data anomalies in continuous data matrix X and discrete states matrix D.

    Args:
        - X (numpy.ndarray): Continuous Data matrix, features in columns, samples in rows.
        - D (numpy.ndarray): Discrete states matrix in one-hot-encoding, features in columns, samples in rows.
        - levels (numpy.ndarray): List of levels for discrete states.
        - lam: (numpy.ndarray): Sequence of penalty values.
        - oIterations (int, optional): Number of iterations for fitting MGMs. Defaults to 10000.
        - oTol (float, optional): Tolerance for fitting MGMs. Defaults to 1e-6.

    Returns:
        tuple: Tuple containing the following elements:
            - X_cor (numpy.ndarray): Continuous data matrix X corrected for anomalies.
            - n_cont (int): Number of detected continuous anomalies.
            - position_cont (numpy.ndarray): Positions of detected continuous anomalies in X.
            - D_cor (numpy.ndarray): Discrete states matrix D corrected for anomalies.
            - n_disc (int): Number of detected discrete anomalies.
            - position_disc (numpy.ndarray): Positions of detected discrete anomalies in D.
    """
    # perform cross validation
    prob_hat, B_m, lam_opt,  x_hat, d_hat = loo_cv_cor(X, D, levels, lam)
    # determine continuous threshold
    X_cor, threshold_cont, n_cont,  position_cont = get_threshold_continuous(X, x_hat, B_m)
    # determine discrete threshold
    n_disc, threshold_disc, position_disc = get_threshold_discrete(D, levels, d_hat)
    # correct D
    D_cor = np.copy(D)
    levelSum = np.cumsum(levels)
    for i in range(n_disc):
        d_h = d_hat[position_disc[0, i], :]
        d_old = D[position_disc[0, i], :]
        d_var = 0
        for k in range(len(d_h)):
            if k == levelSum[d_var]:
                d_var = d_var + 1
            if d_old[k] == 1 and k == position_disc[1, i]:
                tmp = np.zeros(
                    shape=len(d_h[levelSum[d_var-1]:levelSum[d_var]]))
                tmp[np.where(d_h[levelSum[d_var-1]:levelSum[d_var]] == max(d_h[levelSum[d_var-1]:levelSum[d_var]]))] = 1
                D_cor[position_disc[0, i], levelSum[d_var-1]:levelSum[d_var]] = tmp
    return X_cor, n_cont, position_cont, D_cor, n_disc, position_disc
