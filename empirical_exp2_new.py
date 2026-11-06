"""Transfer learning, debiasing, decomposition, and heterogeneous-source experiments."""
import argparse
import os
os.environ.setdefault("MPLBACKEND", "Agg")
from paper_utils import add_output_argument, prepare_output
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, PowerNorm
import numpy as np
from scipy.linalg import inv, norm, circulant
from scipy.optimize import fsolve, root_scalar
from sklearn.model_selection import KFold, cross_val_score
from concurrent.futures import ProcessPoolExecutor
from itertools import chain
import math
from scipy.linalg import toeplitz
parser = argparse.ArgumentParser()
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from scipy.optimize import root


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.strip().lower()
    if v in {"true", "t", "1", "yes", "y"}:
        return True
    if v in {"false", "f", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {v}")

parser.add_argument('--b', type=float, default=1, help="Value for b")
parser.add_argument('--number_of_models', type=int, default=3, help="Number of models")
parser.add_argument('--file_name', type=str, default='pizza')
parser.add_argument('--relation_description', type=str, default='identity')
parser.add_argument('--rotation', type=float, default=1/25)
parser.add_argument('--ones_ratio', type=float, default=0.5)
parser.add_argument('--cov_x_des', type=str, default='identity', help="Covariance matrix for x")

parser.add_argument('--sigma_eta', type=float, default=0.31623, help="Sigma eta values")
parser.add_argument('--sigma_epsilon', type=float, default=0.31623, help="Sigma epsilon values")
parser.add_argument('--sigma_xi', type=float, default=0.31623, help="Sigma xi values")
parser.add_argument('--free_text', type=str, default=None, help="Free text for output files name")

parser.add_argument(
    '--debias',
    type=str2bool,
    nargs='?',
    const=True,
    default=False,
    help="True if the assumed H_tilde is debiased in d/n in the overparam regime"
)

parser.add_argument('--assumption', type=str2bool, nargs='?', const=True, default=False, help="h_tilde = h assumption")

parser.add_argument('--empirical_experiment', type=str2bool, nargs='?', const=True, default=False, help="Run empirical experiment")
parser.add_argument('--number_of_runs', type=int, default=200, help="Number of runs")
parser.add_argument('--n_tilde', type=int, nargs='+', default=[128], help="n tilde values")
parser.add_argument('--n', type=int, default=32, help="n values")
parser.add_argument('--d', type=int, default=128, help="d values")
parser.add_argument('--r', type=int, nargs='?', default=128, help="r subspace dimension, default is dimension of the data")
parser.add_argument('--heat_map', type=str2bool, nargs='?', const=True, default=False)
parser.add_argument('--kappa', type=float, default=10, help="condition number for circulant matrices")
parser.add_argument('--permutation', type=str2bool, nargs='?', const=True, default=False, help="permutation for circulant matrices")
parser.add_argument('--decomposition', type=str2bool, nargs='?', const=True, default=False)
parser.add_argument('--same_param', type=str2bool, nargs='?', const=True, default=True, help="True if all source tasks have the same param")
parser.add_argument('--wker', type=float, default=1/50, help="Kernel width for circulant matrix")
parser.add_argument('--radius', type=float, default=0.1, help="Radius for simplex circulant matrices")
parser.add_argument('--decay', type=float, default=1-1/128, help="Decay for circulant_from_eigs")
parser.add_argument('--peak', type=float, default=0.1, help="Peak for circulant_from_eigs")
parser.add_argument('--cov_z_q', type=float, default=0, help="Decay parameter for exponential covariance matrix of source data")
parser.add_argument('--debias_factor_val', type=str2bool, nargs='?', const=True, default=False, help="If True, find the debiasing factor value by validation")
parser.add_argument('--allaniso', type=str2bool, nargs='?', const=True, default=False, help="If True, are things anisotropic")
parser.add_argument('--opt_debias', type=str2bool, nargs='?', const=True, default=False, help="If True, use optimal shrinkage matrix for the source tasks")
parser.add_argument('--dim_mean', type=float, default=96)
parser.add_argument('--dim_var', type=float, default=256)
parser.add_argument('--condition_number_var', type=float, default=10, help="Variance for the condition number of the circulant matrices, only relevant if relation_description is rand_circulant")
parser.add_argument('--bad_models', type=str2bool, nargs='?', const=True, default=False)
parser.add_argument('--cv', type=str2bool, nargs='?', const=True, default=False, help="If true, find optimal alpha and c with cross-validation")
add_output_argument(parser)
parser.add_argument('--workers', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', '1')))
parser.add_argument('--source-ratios', type=float, nargs='+', default=None,
                    help='Optional explicit d/n_tilde grid; omitted preserves the recovered grid.')
parser.add_argument('--alpha-points', type=int, default=25)
parser.add_argument('--decomposition-main-runs', type=int, default=40)
parser.add_argument('--decomposition-sub-runs', type=int, default=15)
parser.add_argument('--decomposition-points', type=int, default=25)

args = parser.parse_args()
if args.number_of_runs < 1 or args.workers < 1 or args.alpha_points < 1:
    parser.error('runs, workers, and alpha-points must be positive')
if args.source_ratios and any(x <= 0 or x > args.d for x in args.source_ratios):
    parser.error('source-ratios must be positive and no larger than d')
if args.bad_models and (args.d != 128 or args.number_of_models != 20):
    parser.error('the recovered heterogeneous-source experiment requires d=128 and m=20')
if args.bad_models and args.debias_factor_val:
    parser.error('the heterogeneous-source branch does not support factor tuning')
output_root = prepare_output(args, experiment=__name__ == '__main__')

cv = args.cv
bad_models = args.bad_models
dim_mean = args.dim_mean
dim_var = args.dim_var
condition_number_var = args.condition_number_var
file_name = args.file_name
relation_description = args.relation_description
assumption = args.assumption
rotation = args.rotation
p = args.ones_ratio
r = args.r
cov_x_des = args.cov_x_des
b = args.b
number_of_models = args.number_of_models
sigma_eta = args.sigma_eta
sigma_epsilon = args.sigma_epsilon
sigma_xi = args.sigma_xi
debias = args.debias
heat_map = args.heat_map
decomposition = args.decomposition
same_param = args.same_param
kappa = args.kappa
permutation = args.permutation
wker = args.wker
radius = args.radius
free_text = args.free_text
peak = args.peak
decay = args.decay
opt_debias = args.opt_debias
cov_z_q = args.cov_z_q
debias_factor_val = args.debias_factor_val
allaniso = args.allaniso
empirical_experiment = args.empirical_experiment
number_of_runs = args.number_of_runs
n_tilde = args.n_tilde
n = args.n
print(n)
d = args.d
print(d)
gamma_tgt = d/n

exp_param = None


if relation_description == 'rotation':
    exp_param = rotation
if relation_description == 'random_ratio_ones':
    exp_param = p
if relation_description == 'subspace':
    exp_param = r
if relation_description == 'eng_subspace':
    exp_param = r
if relation_description == 'circulant':
    exp_param = f"kappa={kappa}_perm={permutation}"
if relation_description == 'simplex_circulant':
    exp_param = wker
if relation_description == 'circulant_2':
    exp_param = wker
if relation_description == 'symmetric_circulant':
    exp_param = peak
if relation_description == 'filter':
    exp_param = 1/wker
if relation_description == 'rand_eng_sub':
    exp_param = f'mean={dim_mean}_var={dim_var}'
if relation_description == 'rand_sub':
    exp_param = f'mean={dim_mean}_var={dim_var}'


MASTER_SEED = 12345
ss_master = np.random.SeedSequence(MASTER_SEED)
master_rng = np.random.default_rng(MASTER_SEED)

print("b: ", b, " n:", n, " d:", d, " relation_description:", relation_description, " sigma_eta:", sigma_eta, " sigma_epsilon:", sigma_epsilon, " sigma_xi:", sigma_xi, " rotation: ", rotation, " p: ", p, " number of runs: ", number_of_runs)

def anti_isotropic_covariance_matrix(d,h=0,l=-1):
    lambdas = np.logspace(h, l, d)  # exponentially decaying eigenvalues
    return np.diag(lambdas)

def exponential_covariance_matrix(d, sigma_squared=1.0, rho=0.5):
    """
    Generates an n x n covariance matrix with exponentially decaying entries from the diagonal.

    Parameters:
    - n (int): Size of the matrix (number of variables).
    - sigma_squared (float): Variance term (default is 1.0).
    - rho (float): Decay rate (0 < rho < 1).

    Returns:
    - cov_matrix (ndarray): The generated covariance matrix.
    """


    first_row = sigma_squared * rho ** np.arange(d)

    cov_matrix = toeplitz(first_row)

    return cov_matrix

def gen_data(n, cov, mean, ss=None):
    if ss is None:
        return np.random.multivariate_normal(mean, cov, n)
    else:
        return ss.multivariate_normal(mean, cov, n)

def gen_Y(X, beta, sigma_epsilon, ss=None):
    if ss is None:
        return X @ beta + np.random.normal(0, sigma_epsilon, X.shape[0])
    else:
        return X @ beta + ss.normal(0, sigma_epsilon, X.shape[0])

def gen_theta(beta, H, sigma_eta, ss=None):
    cov_eta = sigma_eta**2/d*np.eye(d)
    if ss is None:
        return H @ beta + np.random.multivariate_normal(np.zeros(d), cov_eta)
    else:
        return H @ beta + ss.multivariate_normal(np.zeros(d), cov_eta)

def l2_norm_solution(X, Y):
    return np.linalg.pinv(X) @ Y

def transfer_learning_solution(X, Y, H_tildes_sum, H_theta_tildes_sum, alpha_tl):
    n = X.shape[0]
    return np.linalg.solve(X.T @ X + n*alpha_tl*H_tildes_sum, X.T @ Y + n*alpha_tl*H_theta_tildes_sum)

def estimate_solution(beta_tilde, beta, n_test, sigma_epsilon, cov_x, ss=None):
    data = gen_data(n_test, cov_x, np.zeros(d), ss)
    Y = gen_Y(data, beta, sigma_epsilon, ss)
    return np.linalg.norm(Y - data @ beta_tilde)**2 / n_test

def compute_q0_from_eigs(eigs: np.ndarray, gamma_src: float,
                         tol: float = 1e-12, max_iter: int = 200) -> float:
    """
    Solve:
        mean( q0 * mu / (1 + q0 * mu) ) = 1 / gamma_src
    for q0 > 0, assuming gamma_src > 1.
    """
    if gamma_src <= 1:
        raise ValueError("This shrinkage form is for the overparameterized case gamma_src > 1.")

    target = 1.0 / gamma_src

    def f(q):
        return np.mean((q * eigs) / (1.0 + q * eigs)) - target

    lo, hi = 0.0, 1.0
    while f(hi) < 0:
        hi *= 2.0
        if hi > 1e12:
            raise RuntimeError("Could not bracket q0.")

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        val = f(mid)
        if abs(val) < tol:
            return mid
        if val < 0:
            lo = mid
        else:
            hi = mid

    return 0.5 * (lo + hi)

def anisotropic_shrinkage_matrix_ar1(Sigma: np.ndarray, gamma_src: float):
    """
    Returns:
        Sigma   : AR(1) covariance
        q0      : scalar from the trace-matching equation
        S       : shrinkage matrix q0*Sigma*(I + q0*Sigma)^(-1)
        eigvals : eigenvalues of Sigma
        basis   : eigenvectors of Sigma
        shrink_eigs : eigenvalues of S in the Sigma eigenbasis
    """
    eigvals, basis = np.linalg.eigh(Sigma)  # ascending
    q0 = compute_q0_from_eigs(eigvals, gamma_src)

    shrink_eigs = (q0 * eigvals) / (1.0 + q0 * eigvals)
    S = basis @ np.diag(shrink_eigs) @ basis.T

    return S

def sample_estimator_entries(d, r, s, m, rng=None):
    if d < 0 or int(d) != d:
        raise ValueError("d must be a nonnegative integer")
    if not (0 <= r <= d):
        raise ValueError("r must satisfy 0 <= r <= d")
    if s < 0:
        raise ValueError("s must be nonnegative")
    if m <= 0 or int(m) != m:
        raise ValueError("m must be a positive integer")

    xs = np.arange(d + 1, dtype=float)

    if s == 0:
        if abs(r - round(r)) > 1e-12:
            raise ValueError("If s=0, r must be an integer")
        return np.full(m, int(round(r)), dtype=int)

    max_var = r * (d - r)
    if s > max_var + 1e-12:
        raise ValueError(f"variance too large; max possible is {max_var}")

    def probs_from_params(mu, log_sigma):
        sigma = np.exp(log_sigma)
        logw = -0.5 * ((xs - mu) / sigma) ** 2
        logw -= np.max(logw)
        w = np.exp(logw)
        return w / w.sum()

    def objective(params):
        mu, log_sigma = params
        p = probs_from_params(mu, log_sigma)
        mean = np.sum(xs * p)
        var = np.sum((xs - mean) ** 2 * p)
        return np.array([mean - r, var - s])

    x0 = np.array([np.clip(r, 0, d), np.log(max(np.sqrt(s), 1e-3))])
    sol = root(objective, x0)

    if not sol.success:
        raise RuntimeError("Could not fit distribution")

    p = probs_from_params(sol.x[0], sol.x[1])

    if rng is None:
        rng = np.random.default_rng()

    estimator = rng.choice(xs.astype(int), size=m, p=p)
    return estimator


def transfer_learning_experiment(n, n_src, d, cov_x,  sigma_eta, sigma_epsilon, sigma_xi, b, H_tildes, Hs, alpha_tl_range, n_test, ss=None):

    cov_matrix = (b/d) * np.eye(d)

    if allaniso == False:
        mean = np.zeros(d)  # Zero mean vector
        if ss is None:
            beta = np.random.multivariate_normal(mean, cov_matrix)
        else:
            beta = ss.multivariate_normal(mean, cov_matrix)
    else:
        Sigma = cov_x          # AR(1) covariance
        eigvals, eigvecs = np.linalg.eigh(Sigma)                  # ascending
        U_d = eigvecs[:, -16:]                                     # top-d eigenvectors
        a = np.random.standard_normal(16)
        beta = U_d @ a
        beta = beta / np.linalg.norm(beta)
    thetas = []
    thetas_tildes = []
    if cov_z_q == 0:
        cov_z = np.eye(d)
    else:
        cov_z = exponential_covariance_matrix(d, sigma_squared=1.0, rho=cov_z_q)

    if same_param:
        for i in range(len(Hs)):
            theta = gen_theta(beta, Hs[i], sigma_eta, ss)
            thetas.append(theta)
            Z = gen_data(n_src, cov_z, np.zeros(d), ss)
            V = gen_Y(Z, theta, sigma_xi, ss)
            theta_tilde = l2_norm_solution(Z, V)
            thetas_tildes.append(theta_tilde)
    else:
        m = len(Hs)
        x = d/n_src
        bs = np.linspace(0.5/(m+1), 0.5*m/(m+1), m)
        slop = 1-bs
        for i in range(m):
            theta = gen_theta(beta, Hs[i], sigma_eta, ss)
            thetas.append(theta)
            x = bs[i]+slop[i]*x
            n_src = math.floor(d/x)
            Z = gen_data(n_src, cov_z, np.zeros(d), ss)
            V = gen_Y(Z, theta, sigma_xi, ss)
            theta_tilde = l2_norm_solution(Z, V)
            thetas_tildes.append(theta_tilde)

    X = gen_data(n, cov_x, np.zeros(d), ss)
    Y = gen_Y(X, beta, sigma_epsilon, ss)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    best_alpha = None
    best_err = float('inf')

    sum_1 = np.sum([H.T @ H for H in H_tildes], axis=0)  # Ensure it's a NumPy array
    sum_2 = np.sum([H.T @ theta_tilde for H, theta_tilde in zip(H_tildes, thetas_tildes)], axis=0)


    Y_val = gen_Y(X_val, beta, sigma_epsilon, ss)

    if not debias_factor_val:

        for alpha_tl in alpha_tl_range:
            if cv == False:
                beta_tilde = transfer_learning_solution(X, Y, sum_1, sum_2, alpha_tl)
                Y_ped = np.dot(X_val, beta_tilde)
                loss = np.mean((Y_val - Y_ped) ** 2)
                if loss < best_err:
                    best_err = loss
                    best_alpha = alpha_tl
            if cv == True:
                errors = []
                for train_index, val_index in kf.split(X):
                    X_train, X_val_cv = X[train_index], X[val_index]
                    Y_train, Y_val_cv = Y[train_index], Y[val_index]
                    beta_tilde = transfer_learning_solution(X_train, Y_train, sum_1, sum_2, alpha_tl)
                    Y_ped_cv = np.dot(X_val_cv, beta_tilde)
                    error = np.mean((Y_val_cv - Y_ped_cv) ** 2)
                    errors.append(error)
                avg_error = np.mean(errors)
                if avg_error < best_err:
                    best_err = avg_error
                    best_alpha = alpha_tl

        beta_tilde_final = transfer_learning_solution(X, Y, sum_1, sum_2, best_alpha)
        return estimate_solution(beta_tilde_final, beta, n_test, sigma_epsilon, cov_x, ss), best_alpha
    else:
        if cv == False:
            for alpha_tl in alpha_tl_range:
                a_s, a_e, a_size = n_src/d, 2.5, 15
                a_s = min(a_s, 0.5)
                fac_arr = np.unique(np.sort(np.concatenate([np.geomspace(a_s, a_e, a_size-1), [1.0]])))
                for factor in fac_arr:
                    sum_1 = np.sum([factor* H.T @ H*factor for H in H_tildes], axis=0)
                    sum_2 = np.sum([factor* H.T @ theta_tilde for H, theta_tilde in zip(H_tildes, thetas_tildes)], axis=0)
                    beta_tilde = transfer_learning_solution(X, Y, sum_1, sum_2, alpha_tl)
                    Y_ped = np.dot(X_val, beta_tilde)
                    loss = np.mean((Y_val - Y_ped) ** 2)
                    if loss < best_err:
                        best_err = loss
                        best_alpha = alpha_tl
                        best_factor = factor
        if cv == True:
            alpha_tl_range = np.logspace(-4, 2, 10)
            for alpha_tl in alpha_tl_range:
                a_s, a_e, a_size = n_src/d, 2.5, 10
                a_s = min(a_s, 0.5)
                fac_arr = np.unique(np.sort(np.concatenate([np.geomspace(a_s, a_e, a_size-1), [1.0]])))
                for factor in fac_arr:
                    errors = []
                    for train_index, val_index in kf.split(X):
                        X_train, X_val_cv = X[train_index], X[val_index]
                        Y_train, Y_val_cv = Y[train_index], Y[val_index]
                        sum_1 = np.sum([factor* H.T @ H*factor for H in H_tildes], axis=0)
                        sum_2 = np.sum([factor* H.T @ theta_tilde for H, theta_tilde in zip(H_tildes, thetas_tildes)], axis=0)
                        beta_tilde = transfer_learning_solution(X_train, Y_train, sum_1, sum_2, alpha_tl)
                        Y_ped_cv = np.dot(X_val_cv, beta_tilde)
                        error = np.mean((Y_val_cv - Y_ped_cv) ** 2)
                        errors.append(error)
                    avg_error = np.mean(errors)
                    if avg_error < best_err:
                        best_err = avg_error
                        best_alpha = alpha_tl
                        best_factor = factor

        sum_1 = np.sum([best_factor* H.T @ H*best_factor for H in H_tildes], axis=0)
        sum_2 = np.sum([best_factor* H.T @ theta_tilde for H, theta_tilde in zip(H_tildes, thetas_tildes)], axis=0)
        beta_tilde_final = transfer_learning_solution(X, Y, sum_1, sum_2, best_alpha)
        return estimate_solution(beta_tilde_final, beta, n_test, sigma_epsilon, cov_x, ss), best_factor*H_tildes[0][0][0], best_alpha


def circulant_from_eigs_matrix(lambda_vec):
    """
    Build circulant C given its eigenvalues (length n) using a unitary DFT.
    Returns an (n,n) complex (or real) ndarray.
    """
    lam = np.asarray(lambda_vec)
    n = lam.size

    j, k = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
    U = np.exp(-2j * np.pi * j * k / n) / np.sqrt(n)

    C = U.conj().T @ (lam * U.T).T   # same as U.conj().T @ np.diag(lam) @ U but without forming diag
    return np.real_if_close(C, tol=1e5)


def circulant_matrix_gaussian_filter(d, conv_gaussian_width):
    first_half = np.exp(-conv_gaussian_width * np.linspace(0, 1, d // 2))
    second_half = np.exp(-conv_gaussian_width * np.linspace(1, 0, -(-d // 2)))  # ceil(d/2) with trick

    H_circular_first_row = np.concatenate([first_half, second_half])

    H_circular_first_row[0] += 1.0

    H_circular_first_row = H_circular_first_row / np.sqrt(np.sum(H_circular_first_row**2))

    H_circular_first_column = np.concatenate([[H_circular_first_row[0]], H_circular_first_row[1:][::-1]])

    H = toeplitz(H_circular_first_column, H_circular_first_row)

    return H


def make_eigen_values(kappa,d):
  """
  Makes List of decaying eigen values for circulant_from_eigs_matrix.
  For circulant matrix to have real entiries it needs to be symatric around d/2+1 in Lambda diagonal in the decomposition F* @ Lambda @F
  We want 1/d|Lambda|_f^2 = 1
  Works only for even dimension d
  """
  smallest_eigen = np.sqrt(2/(kappa**2+1))
  bigest_eigen = kappa* smallest_eigen
  eigens_arr_squared =  np.linspace(smallest_eigen**2,bigest_eigen**2,int(d/2) +1 )

  eigen_vals = np.sqrt(eigens_arr_squared)
  return eigen_vals


def generate_circulants_matrices_with_condition_number(kappa,d, m, permutation = False):
    Hs = []
    egs = make_eigen_values(kappa,d)
    mid_egs = egs[1:-1]
    if permutation:
      for i in range(m):
        mid_egs = np.roll(mid_egs, math.ceil((d/2-1)/m))
        rev = mid_egs[::-1]
        egs[1:-1] = mid_egs
        final_egs = np.concatenate((egs,rev))
        H = circulant_from_eigs_matrix(final_egs)
        Hs.append(H)
    else:
      H = circulant_from_eigs_matrix(np.concatenate((egs,egs[-2:0:-1])))
      print(np.concatenate((egs,egs[-2:0:-1])))
      Hs = [H for _ in range(m)]
    return Hs


def generate_random_subspace(d, r):
    """
    Samples a uniformly random r-dimensional subspace of R^d.

    Returns:
    - Q: (d x r) matrix whose columns form an orthonormal basis for the subspace
    - P: (d x d) orthogonal projection matrix onto the subspace
    - project_fn: function that projects a given vector x ∈ R^d onto the subspace
    """
    A = master_rng.standard_normal((d, r))

    Q, _ = np.linalg.qr(A)

    P = Q @ Q.T

    return P

def generate_random_energy_subspace(d, r):
    """
    Samples a uniformly random r-dimensional subspace of R^d.

    Returns:
    - Q: (d x r) matrix whose columns form an orthonormal basis for the subspace
    - P: (d x d) orthogonal projection matrix onto the subspace
    - project_fn: function that projects a given vector x ∈ R^d onto the subspace
    """
    A = master_rng.standard_normal((d, r))

    Q, _ = np.linalg.qr(A)

    scalar = np.sqrt(d/r)
    P = scalar * Q @ Q.T

    return P

def generate_random_scaled_subspace(d, r):
    """
    Samples a uniformly random r-dimensional subspace of R^d.

    Returns:
    - Q: (d x r) matrix whose columns form an orthonormal basis for the subspace
    - P: (d x d) orthogonal projection matrix onto the subspace
    - project_fn: function that projects a given vector x ∈ R^d onto the subspace
    """
    A = master_rng.standard_normal((d, r))

    Q, _ = np.linalg.qr(A)

    alp = 10
    diag = master_rng.gamma(alp, 1/alp, r)
    M = np.diag(diag)

    P = Q @ M @ Q.T

    return P


def generate_circulant_matrix(d, wker):
    """
    Generates a circulant matrix of dimension d x d based on the given wker parameter.

    Parameters:
    - d (int): Dimension of the matrix (d x d).
    - wker (float): Kernel width parameter controlling the exponential decay.

    Returns:
    - H (numpy.ndarray): A d x d circulant matrix.
    """
    if d <= 0:
        raise ValueError("Dimension d must be a positive integer.")
    if wker <= 0:
        raise ValueError("wker must be a positive float.")

    tau = np.linspace(0, 1, d)

    h = np.exp(-np.abs(tau - 0.5) / wker)
    h[0] += 1  # Adding the delta function contribution at k=0

    H = circulant(h)
    row_norms = np.linalg.norm(H, axis=1, keepdims=True)
    H_normalized = H / row_norms

    return H_normalized

def generate_scaled_circulant_matrix(d, wker, u=0):
    """
    Generates a circulant matrix of dimension d x d based on the given wker parameter.

    Parameters:
    - d (int): Dimension of the matrix (d x d).
    - wker (float): Kernel width parameter controlling the exponential decay.

    Returns:
    - H (numpy.ndarray): A d x d circulant matrix.
    """
    if d <= 0:
        raise ValueError("Dimension d must be a positive integer.")
    if wker <= 0:
        raise ValueError("wker must be a positive float.")

    tau = np.linspace(0, 1, d)

    h = u * np.exp(-np.abs(tau - 0.5) / wker)
    h[0] += 1  # Adding the delta function contribution at k=0

    H = circulant(h)
    return H


def generate_batch_matrix(d,i,j):
    a = np.zeros(i)
    b = np.ones(j-i)
    c = np.zeros(d-j)
    d = np.concatenate([a,b,c])
    return np.diag(d)


def generate_seperate_batch_matrices(d,m):
    batch_size = int(d/m)
    return [generate_batch_matrix(d, i*batch_size, (i+1)*batch_size) for i in range(m)]


def generate_random_diagonal_matrix(p, d):
    """
    Generates a d x d matrix where exactly p*d diagonal elements are randomly set to 1.

    Parameters:
    p (float): A fraction between 0 and 1, representing the proportion of the diagonal to be set to 1.
    d (int): The dimension of the square matrix.

    Returns:
    numpy.ndarray: A d x d matrix with p*d ones on the diagonal, randomly placed.
    """
    matrix = np.zeros((d, d), dtype=int)

    num_ones = int(p * d)

    diagonal_indices = np.arange(d)

    chosen_indices = master_rng.choice(diagonal_indices, size=num_ones, replace=False)

    matrix[chosen_indices, chosen_indices] = 1

    return matrix

def generate_symmetric_circulant(d = 128, decay = 1-1/128,peak = 0.1):
  c = [peak*decay**np.abs((d/2-i)) for i in range(d)]
  for i in range(int(d/2)-1):
    if c[i+1] != c[d-i-1]:
      print("c is not symetric")
      print(c)
      return
  c[0]+=1
  C = circulant(c)
  C *= np.sqrt(d) / norm(C, 'fro')
  return C


def generate_random_rotation_matrix(d, theta):
    """
    Generate a random rotation matrix in d dimensions with angles drawn from [-theta, theta].

    This optimized version updates only the two affected columns in each 2D rotation,
    avoiding the creation and full multiplication of d x d matrices.

    Parameters:
      d: int, dimension of the rotation matrix.
      theta: float, maximum angle in radians.

    Returns:
      rotation_matrix: np.ndarray, a d x d rotation matrix.
    """
    rotation_matrix = np.eye(d)

    num_planes = d * (d - 1) // 2
    angles = master_rng.uniform(-theta, theta, size=num_planes)

    plane_index = 0
    for i in range(d):
        for j in range(i + 1, d):
            angle = angles[plane_index]
            plane_index += 1
            c, s = np.cos(angle), np.sin(angle)

            col_i = rotation_matrix[:, i].copy()
            col_j = rotation_matrix[:, j].copy()

            rotation_matrix[:, i] = c * col_i - s * col_j
            rotation_matrix[:, j] = s * col_i + c * col_j

    return rotation_matrix

def random_rotation_matrix(d, theta):
    """
    Generate a random d x d rotation matrix R,
    rotating by angle +theta or -theta in a random 2D subspace.
    """

    sign = master_rng.choice([-1, 1])
    angle = sign * theta

    v1 = master_rng.standard_normal(d)
    v1 /= np.linalg.norm(v1)

    v2 = master_rng.standard_normal(d)
    v2 -= np.dot(v2, v1) * v1
    v2_norm = np.linalg.norm(v2)
    if v2_norm < 1e-15:
        return random_rotation_matrix(d, theta)
    v2 /= v2_norm

    c = np.cos(angle)
    s = np.sin(angle)

    v1v1T = np.outer(v1, v1)
    v2v2T = np.outer(v2, v2)
    v1v2T = np.outer(v1, v2)
    v2v1T = np.outer(v2, v1)

    I = np.eye(d)
    R = (I
         + (c - 1) * (v1v1T + v2v2T)
         + s * (v2v1T - v1v2T))

    return R

def generate_rotation_matrices(d, m, theta):
    return [generate_random_rotation_matrix(d, theta) for _ in range(m)]

def m_func(alpha_opt_infinity, gamma_tgt):
    part1 = (1 - gamma_tgt + alpha_opt_infinity)
    part2 = np.sqrt((1 - gamma_tgt + alpha_opt_infinity) ** 2 + 4 * gamma_tgt * alpha_opt_infinity)
    return (-part1 + part2) / (2 * gamma_tgt * alpha_opt_infinity)

def ridge_regression_error(n, d, sigma_epsilon, b):
    gamma_tgt = d/n
    sigma_opt = sigma_epsilon ** 2 / b

    alpha_opt_infinity = sigma_opt * gamma_tgt


    temp = m_func(alpha_opt_infinity, gamma_tgt)

    generalization_error = sigma_epsilon ** 2 * (1 + gamma_tgt * temp)

    return generalization_error

def ols_regression_error(n, d, sigma_epsilon, b):
    gamma_tgt = d/n
    generalization_error = sigma_epsilon ** 2 * (1 + gamma_tgt)

    return generalization_error

def circulant_eigenvalues_from_first_row(first_row):
    """Eigenvalues of a circulant with first row 'first_row'."""
    return np.fft.fft(np.asarray(first_row, dtype=complex))

def first_row_from_eigenvalues_symmetric(lmbda):
    """
    Given a conjugate-symmetric spectrum lmbda (length d),
    return a real first row that generates a real circulant.
    """
    h = np.fft.ifft(lmbda)
    return np.real_if_close(h, tol=1000)  # force tiny imag parts to 0


def _pairs_conj_sym(d):
    dc_idx = 0
    nyq_idx = (d // 2) if (d % 2 == 0) else None
    pairs = [(k, d - k) for k in range(1, d // 2 + (0 if d % 2 else 0))]
    return dc_idx, nyq_idx, pairs

def _T_from_real_params(d):
    """
    Build T: R^q -> C^d so that x = T @ y is conjugate-symmetric:
      x_0, (x_{d/2} if even) real, and x_{d-k} = conj(x_k).
    y packs [DC, (Nyquist?), Re/Im for each pair].
    """
    dc_idx, nyq_idx, pairs = _pairs_conj_sym(d)
    q = 1 + (1 if nyq_idx is not None else 0) + 2 * len(pairs)
    T = np.zeros((d, q), dtype=complex)
    pos = 0
    T[dc_idx, pos] = 1.0; pos += 1
    if nyq_idx is not None:
        T[nyq_idx, pos] = 1.0; pos += 1
    for (k, dk) in pairs:
        T[k,  pos] = 1.0
        T[dk, pos] = 1.0
        T[k,  pos+1] = 1j
        T[dk, pos+1] = -1j
        pos += 2
    return T  # so any x in CS has the form x = T y, y real

def _regular_simplex(m):
    """Rows are vertices of a regular simplex in R^{m-1}, unit row-norm, zero-sum, equidistant."""
    if m < 2:
        raise ValueError("m >= 2 required.")
    A = np.eye(m) - np.ones((m, m)) / m
    Q, _ = np.linalg.qr(A)
    W = Q[:, :m-1]
    V = A @ W
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    return V

def _nullspace_real(A, tol=1e-12):
    """Right nullspace (real) of A using SVD. Returns columns forming a basis."""
    U, S, Vt = np.linalg.svd(A, full_matrices=True)
    r = np.sum(S > tol)
    return Vt[r:].T


def make_real_circulants_around_lambda(lmbda, m, r=0.1, symmetrize_target=True, verify=True):
    """
    Inputs
    ------
    lmbda : (d,) complex array
        Target eigenvalues of the base circulant.
    m : int
        Number of circulant matrices to generate.
    r : float
        Simplex radius in the constrained subspace => controls ||lambda^(j)-lambda||.
    symmetrize_target : bool
        If True, project 'lmbda' to the conjugate-symmetric set first (recommended).

    Returns
    -------
    Hs : list of m real circulant matrices (each d x d)
    Ls : (m, d) array of complex spectra (each conjugate-symmetric)
    checks : dict of booleans & diagnostics
    """
    lmbda = np.asarray(lmbda)
    d = lmbda.size

    def symmetrize_cs(x):
        x = np.asarray(x, dtype=complex).copy()
        dc, nyq, pairs = _pairs_conj_sym(d)
        x[dc] = np.real(x[dc])
        if nyq is not None:
            x[nyq] = np.real(x[nyq])
        for (k, dk) in pairs:
            z = 0.5 * (x[k] + np.conj(x[dk]))
            x[k], x[dk] = z, np.conj(z)
        return x

    if symmetrize_target:
        lmbda = symmetrize_cs(lmbda)

    T = _T_from_real_params(d)
    q = T.shape[1]

    ones = np.ones(d, dtype=complex)
    g1 = T.conj().T @ ones
    gl = T.conj().T @ lmbda

    A = np.vstack([np.real(g1), np.real(gl), np.imag(gl)])  # shape (<=3) x q
    N = _nullspace_real(A)  # q x s
    s = N.shape[1]
    if s < m - 1:
        raise ValueError(f"Not enough room for a simplex: dim={s} < m-1={m-1}. "
                         "Increase d or reduce m (or relax constraints).")

    NR = N[:, :m-1]              # q x (m-1)
    Bc = T @ NR                  # d x (m-1) complex
    B, _ = np.linalg.qr(Bc)      # d x (m-1), orthonormal columns

    V = _regular_simplex(m)      # m x (m-1)
    E = V @ B.T                  # m x d complex (rows in the constrained CS-subspace)
    E *= r

    Ls = lmbda[None, :] + E      # m x d
    Hs = []
    for j in range(m):
        h_first = first_row_from_eigenvalues_symmetric(Ls[j])
        Hs.append(circulant(h_first).real)  # should be real by construction

    checks = {}
    checks['mean_matches'] = np.allclose(Ls.mean(axis=0), lmbda, atol=1e-10)

    norms  = np.sqrt(np.real(np.einsum('ij,ij->i', Ls.conj(), Ls)))
    dist1  = np.sqrt(np.real(np.einsum('ij,ij->i', (Ls-1).conj(), (Ls-1))))
    checks['equal_norms'] = (np.max(np.abs(norms - norms[0])) < 1e-10)
    checks['equal_dist_from_ones'] = (np.max(np.abs(dist1 - dist1[0])) < 1e-10)

    pd = []
    for i in range(m):
        for j in range(i+1, m):
            diff = Ls[i] - Ls[j]
            pd.append(np.sqrt(np.real(np.vdot(diff, diff))))
    pd = np.array(pd)
    checks['pairwise_equidistant'] = (np.max(pd) - np.min(pd) < 1e-10)
    checks['pairwise_distance_matches_formula'] = np.allclose(
        np.mean(pd), np.sqrt(2*m/(m-1))*r, rtol=1e-10, atol=1e-12
    )

    checks['E_orth_to_ones']   = np.allclose(E @ np.ones(d), 0, atol=1e-10)
    checks['E_orth_to_lambda'] = np.allclose(E @ lmbda, 0, atol=1e-10)

    Er = np.sqrt(np.real(np.einsum('ij,ij->i', E.conj(), E)))
    checks['radius_r'] = np.max(np.abs(Er - r)) < 1e-12

    def cs_ok(row, atol=1e-10):
        dc, nyq, pairs = _pairs_conj_sym(row.size)
        ok = np.allclose(row[dc].imag, 0, atol=atol)
        if nyq is not None:
            ok &= np.allclose(row[nyq].imag, 0, atol=atol)
        for (k, dk) in pairs:
            ok &= np.allclose(row[dk], np.conj(row[k]), atol=atol)
        return ok
    checks['all_rows_conj_symmetric'] = all(cs_ok(Ls[i]) for i in range(m))

    if verify:
        ok_spec = True
        for j in range(m):
            evals_fft = circulant_eigenvalues_from_first_row(Hs[j][0])  # first row defines circulant
            ok_spec &= np.allclose(np.sort_complex(evals_fft),
                                   np.sort_complex(Ls[j]), atol=1e-8)
        checks['eigs_match_ifft'] = ok_spec

    checks['Hs_all_real'] = all(np.allclose(H, H.real, atol=1e-12) for H in Hs)

    return Hs, Ls, checks
def is_nearly_real(X, tol=1e-10):
    """
    Returns True if all entries of X have |imag| <= tol (max-wise).
    """
    return np.max(np.abs(np.imag(X))) <= tol

def enforce_real_if_small_imag(Hs, imag_tol=1e-10, drop_if_not=False, verbose=True):
    """
    Iterate over a list of matrices Hs.
    - If a matrix is 'nearly real' (by imag_tol), replace it with H.real.
    - If not nearly real:
        * if drop_if_not=True: drop it from the list,
        * else: keep as-is (complex).
    Returns (Hs_out, kept_mask), where kept_mask[i] indicates whether original Hs[i] survived.
    """
    Hs_out = []
    kept_mask = []
    for idx, H in enumerate(Hs):
        if is_nearly_real(H, tol=imag_tol):
            if verbose:
                print(f"[ok] H[{idx}] nearly real (max |Im|={np.max(np.abs(np.imag(H))):.2e}); using real part.")
            Hs_out.append(H.real)
            kept_mask.append(True)
        else:
            msg = f"[warn] H[{idx}] NOT nearly real (max |Im|={np.max(np.abs(np.imag(H))):.2e})"
            if drop_if_not:
                if verbose: print(msg + " -> DROPPED.")
                kept_mask.append(False)
                continue
            else:
                if verbose: print(msg + " -> kept as complex.")
                Hs_out.append(H)
                kept_mask.append(True)
    return Hs_out, kept_mask

def generate_simplex_circulant_matrices(wker,m,d,radius=0.5):
    H_base = generate_circulant_matrix(d, wker)
    lmbda = circulant_eigenvalues_from_first_row(H_base[0])
    Hs, Ls, checks = make_real_circulants_around_lambda(lmbda, m, radius, symmetrize_target=True, verify=True)
    Hs, kept_mask = enforce_real_if_small_imag(Hs, imag_tol=1e-10, drop_if_not=False, verbose=True)
    if len(Hs) < m:
        print(f"Warning: only {len(Hs)} out of {m} circulants kept after realness check.")
    return Hs


if cov_x_des == 'identity':
        cov_x = np.eye(d, d)
if cov_x_des == 'exp_decay':
        cov_x = exponential_covariance_matrix(d)
if cov_x_des == 'anti_isotropic':
        cov_x = anti_isotropic_covariance_matrix(d, h=0, l=-0.5)

error = []
param_lvl = []
m = number_of_models

alpha_tl_range = np.logspace(-4, 2, args.alpha_points)
X_val = gen_data(1000, cov_x, np.zeros(d), master_rng)


def sample_beta_from_mean_var(m, mean, var, rng=None):
    if not (0 < mean < 1):
        raise ValueError("mean must be between 0 and 1")
    if not (0 < var < mean * (1 - mean)):
        raise ValueError("var must satisfy 0 < var < mean*(1-mean)")

    k = mean * (1 - mean) / var - 1
    alpha = mean * k
    beta = (1 - mean) * k

    if rng is None:
        samples = np.random.beta(alpha, beta, size=m)
    else:
        samples = rng.beta(alpha, beta, size=m)

    return samples


import numpy as np

def generate_condition_numbers(n_samples, target_mean, target_var):
    """
    מגריל מספרי מצב (Condition Numbers) עם תמך [1, infinity)
    לפי ממוצע ושונות מבוקשים, על בסיס התפלגות גמא מוזזת.
    """
    if target_mean <= 1:
        raise ValueError("Target mean must be strictly greater than 1 for condition numbers.")

    shifted_mean = target_mean - 1.0

    k = (shifted_mean ** 2) / target_var     # Shape parameter
    theta = target_var / shifted_mean        # Scale parameter

    X = np.random.gamma(shape=k, scale=theta, size=n_samples)

    condition_numbers = X + 1.0

    return condition_numbers


if relation_description == 'identity':
    Hs = [np.eye(d, d) for _ in range(m)]
if relation_description == 'circulant_2':
    Hs = [generate_circulant_matrix(d,wker) for _ in range(m)]
if relation_description == 'scaled_circulant':
    us = master_rng.uniform(-0.5, 0.5, m)
    Hs = [generate_scaled_circulant_matrix(d, 2 / 75, us[_]) for _ in range(m)]
if relation_description == 'rotation':
    Hs = generate_rotation_matrices(d, m, np.pi*rotation)
if relation_description == 'seperate_batch':
    Hs = generate_seperate_batch_matrices(d,m)
if relation_description == 'random_ratio_ones':
    Hs = [generate_random_diagonal_matrix(p,d) for _ in range(m)]
if relation_description == 'subspace':
    Hs = [generate_random_subspace(d, args.r) for _ in range(m)]
if relation_description == 'scaled_subspace':
    Hs = [generate_random_scaled_subspace(d, args.r) for _ in range(m)]
if relation_description == 'eng_subspace':
    Hs = [generate_random_energy_subspace(d, args.r) for _ in range(m)]
if relation_description == 'circulant':
    Hs = generate_circulants_matrices_with_condition_number(kappa,d,m, permutation)
if relation_description == 'simplex_circulant':
    Hs = generate_simplex_circulant_matrices(wker,m,d,radius=0.5)
if relation_description == 'symmetric_circulant':
    H = generate_symmetric_circulant(d=d, decay=decay, peak=peak)
    Hs = [H for _ in range(m)]
if relation_description == 'filter':
    H = circulant_matrix_gaussian_filter(d, 1/wker)
    Hs = [H for _ in range(m)]
if relation_description == 'rand_eng_sub':
    dims = sample_estimator_entries(d, dim_mean , dim_var, m)
    Hs = [generate_random_energy_subspace(d,r) for r in dims]
if relation_description == 'rand_sub':
    dims = sample_estimator_entries(d, dim_mean , dim_var, m)
    Hs = [generate_random_subspace(d,r) for r in dims]
if relation_description == 'rand_circulant':
    conds = generate_condition_numbers(m, kappa, condition_number_var)
    Hs = [generate_circulants_matrices_with_condition_number(kappa, d, 1, False)[0] for kappa in conds]


if assumption == False:
    H_tildes = [np.eye(d) for i in range(m)]
else:
    H_tildes = Hs
num_runs = number_of_runs

x_temp = np.linspace(0.1, 5.1, 35)
y = [1.02]
x_values = np.sort(np.concatenate([y, x_temp]))
if args.source_ratios is not None:
    x_values = np.asarray(args.source_ratios)
src_size_range = [math.floor(d/i) for i in x_values]

for n_tilde in src_size_range:
    param_lvl.append(d / n_tilde)
    error.append(0)


experiment_args = {
    'n': n,
    'd': d,
    'cov_x': cov_x,
    'sigma_eta': sigma_eta,
    'sigma_epsilon': sigma_epsilon,
    'sigma_xi': sigma_xi,
    'b': b,
    'H_tildes': H_tildes,
    'Hs': Hs,
    'alpha_tl_range': alpha_tl_range,
    'n_test': 1000
}

def single_run(src_size_range, experiment_args, ss=None):

    cov_z = exponential_covariance_matrix(d, sigma_squared=1.0, rho=cov_z_q)
    rng = np.random.default_rng(ss)
    run_errors = []
    factors = []
    runs_alpha = []
    if debias_factor_val == False:
        for n_tilde in src_size_range:
            if (d > n_tilde) & debias:
                debiased_H = [H_tilde / (d/n_tilde) for H_tilde in experiment_args['H_tildes']]
                temp_error, temp_alpha = transfer_learning_experiment(
                experiment_args['n'], n_tilde, experiment_args['d'],
                experiment_args['cov_x'], experiment_args['sigma_eta'],
                experiment_args['sigma_epsilon'], experiment_args['sigma_xi'],
                experiment_args['b'], debiased_H,
                experiment_args['Hs'], experiment_args['alpha_tl_range'],
                experiment_args['n_test'], ss=rng
                )
                run_errors.append(temp_error)
                runs_alpha.append(temp_alpha)
            elif(d> n_tilde) & opt_debias:
                S = anisotropic_shrinkage_matrix_ar1(cov_z, d/n_tilde)
                debiased_H = [S @H_tilde for H_tilde in experiment_args['H_tildes']]
                temp_error, temp_alpha = transfer_learning_experiment(
                experiment_args['n'], n_tilde, experiment_args['d'],
                experiment_args['cov_x'], experiment_args['sigma_eta'],
                experiment_args['sigma_epsilon'], experiment_args['sigma_xi'],
                experiment_args['b'], debiased_H,
                experiment_args['Hs'], experiment_args['alpha_tl_range'],
                experiment_args['n_test'], ss=rng
                )
                run_errors.append(temp_error)
                runs_alpha.append(temp_alpha)
            else:
                temp_error, temp_alpha = transfer_learning_experiment(
                experiment_args['n'], n_tilde, experiment_args['d'],
                experiment_args['cov_x'], experiment_args['sigma_eta'],
                experiment_args['sigma_epsilon'], experiment_args['sigma_xi'],
                experiment_args['b'], experiment_args['H_tildes'],
                experiment_args['Hs'], experiment_args['alpha_tl_range'],
                experiment_args['n_test'], ss=rng
                )
                run_errors.append(temp_error)
                runs_alpha.append(temp_alpha)
        return run_errors, runs_alpha
    else:
        for n_tilde in src_size_range:
            if (d > n_tilde) & debias:
                debiased_H = [H_tilde / (d/n_tilde) for H_tilde in experiment_args['H_tildes']]
                temp_error, factor, temp_alpha = transfer_learning_experiment(
                experiment_args['n'], n_tilde, experiment_args['d'],
                experiment_args['cov_x'], experiment_args['sigma_eta'],
                experiment_args['sigma_epsilon'], experiment_args['sigma_xi'],
                experiment_args['b'], debiased_H,
                experiment_args['Hs'], experiment_args['alpha_tl_range'],
                experiment_args['n_test'], ss=rng
                )
                run_errors.append(temp_error)
                factors.append(factor)
                runs_alpha.append(temp_alpha)
            else:
                temp_error, factor, temp_alpha = transfer_learning_experiment(
                experiment_args['n'], n_tilde, experiment_args['d'],
                experiment_args['cov_x'], experiment_args['sigma_eta'],
                experiment_args['sigma_epsilon'], experiment_args['sigma_xi'],
                experiment_args['b'], experiment_args['H_tildes'],
                experiment_args['Hs'], experiment_args['alpha_tl_range'],
                experiment_args['n_test'], ss=rng
                )
                run_errors.append(temp_error)
                factors.append(factor)
                runs_alpha.append(temp_alpha)

        return run_errors, factors, runs_alpha


if __name__ == '__main__':
    import time as time
    from concurrent.futures import ProcessPoolExecutor

    child_seqs = ss_master.spawn(num_runs)
    start = time.time()

    num_cpus = min(num_runs, args.workers)
    print(num_cpus)

    if heat_map == False and decomposition == False and debias_factor_val == False and bad_models == False:

        print(Hs, flush = True)
        print(H_tildes, flush = True)
        results = []
        best_alpha = []
        with ProcessPoolExecutor(max_workers=num_cpus) as executor:
            futures = [executor.submit(single_run, src_size_range, experiment_args, child_seqs[_]) for _ in range(num_runs)]

            for future in futures:
                results.append(future.result()[0])
                best_alpha.append(future.result()[1])

        stds = np.std(results, axis=0)
        stderr_means = stds / np.sqrt(num_runs)
        error = np.sum(results, axis=0)
        average_error = [e / num_runs for e in error]
        best_alpha = np.mean(best_alpha, axis=0)
        average_alpha = best_alpha


        end = time.time()
        print("Time taken: ", end - start)

        Xs = [d / n_tilde for n_tilde in src_size_range]
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(Xs, average_error, color='orange')
        plt.xlabel('d/n_tilde')
        plt.ylabel('Error')
        plt.title('Error vs d/n_tilde')
        plt.ylim(0, 1.4)
        plt.subplot(1, 2, 2)
        plt.plot(Xs, average_alpha, color='blue')
        plt.yscale('log')
        plt.xlabel('d/n_tilde')
        plt.ylabel('Best Alpha')
        plt.title('Best Alpha vs d/n_tilde')
        plt.ylim(0, max(average_alpha)*1.4)
        plt.suptitle('Debias Factor, Error, and Alpha vs d/n_tilde')
        output_path = f"{output_root}/graphs/debias={debias}_{free_text}_{relation_description}_{exp_param}_num_models={m}_{n}_{end}_only_empiric.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')  # High-quality image

        plt.tight_layout()
        target_dir = f"./results/{free_text}_{relation_description}_debias={debias}_assumption={assumption}_{exp_param}_{gamma_tgt}_{sigma_eta}_{sigma_xi}_{sigma_epsilon}_{num_runs}_cov={cov_x_des}_cov_z={cov_z_q}_cv={cv}_debias_factor_val={debias_factor_val}_opt_debias={opt_debias}_bad_models={bad_models}"
        os.makedirs(target_dir, exist_ok=True)

        final_result = np.stack((Xs, average_error, stderr_means, average_alpha), axis=0)
        exp_name = file_name + "_" +   f"{relation_description}_num_runs={num_runs}_tgt_param_{gamma_tgt}_num_models={m}_sigama_eta={sigma_eta}_sigma_xi={sigma_xi}_sigma_epsilon={sigma_epsilon}_rotation={rotation}_p={p}_r={args.r}.npy"

        file_path = os.path.join(target_dir, exp_name)
        np.save(file_path, final_result)

    if heat_map == True:
        ridge_err = ridge_regression_error(n,d,sigma_epsilon,b)
        r_range = range(1,d+1,2)
        heatmap_data = np.zeros((len(r_range), len(src_size_range)))
        j = 0
        for size_r in r_range:
            print("Size of the subspace: ", size_r)
            Hs = [generate_random_subspace(d, size_r) for _ in range(m)]
            results = []
            with ProcessPoolExecutor(max_workers=num_cpus) as executor:
                futures = [executor.submit(single_run, src_size_range, experiment_args) for _ in range(num_runs)]

                for future in futures:
                    results.append(future.result())

            stds = np.std(results, axis=0)
            error = np.sum(results, axis=0)
            average_error = [e / num_runs for e in error]
            heatmap_data[j, :] = average_error
            j+=1

        plt.figure(figsize=(10, 6))
        ax = sns.heatmap(heatmap_data, annot=False, cmap='coolwarm', center=ridge_err, vmin= sigma_epsilon**2 , vmax = 1.4,  xticklabels=False, yticklabels=False)
        x_vals = np.linspace(0, 5.1, heatmap_data.shape[1])
        y_vals = np.linspace(0, 1.0, heatmap_data.shape[0])

        x_ticks = [np.abs(x_vals - val).argmin() for val in [1, 2, 3, 4, 5]]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([1, 2, 3, 4, 5], fontsize=10)

        y_ticks = [np.abs(y_vals - val).argmin() for val in [0.2, 0.4, 0.6, 0.8]]
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([0.2, 0.4, 0.6, 0.8], fontsize=10)


        plt.title("Example Heatmap")
        ax.set_xlabel("Source tasks parameterization (x-axis)")
        ax.set_ylabel("Subspace size")
        output_path = f"{output_root}/graphs/num_models={m}_{free_text}_{n}_r={args.r}_{num_runs}_heat_map.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')  # High-quality image
        plt.tight_layout()
        target_dir = f"./results/{free_text}_{relation_description}_debias={debias}_assumption={assumption}_{exp_param}_{gamma_tgt}_{sigma_eta}_{sigma_xi}_{sigma_epsilon}_{num_runs}_cov={cov_x_des}_cov_z={cov_z_q}_cv={cv}_debias_factor_val={debias_factor_val}_opt_debias={opt_debias}_bad_models={bad_models}"
        os.makedirs(target_dir, exist_ok=True)
        np.save(os.path.join(target_dir, 'heatmap_data.npy'), heatmap_data)

    if decomposition == True:
        print('Began decomposition')
        cov_z = None
        if cov_z_q == 0:
            cov_z = np.eye(d)
        else:
            cov_z = exponential_covariance_matrix(d, sigma_squared=1.0, rho=cov_z_q)
        def exp_beta(n,d,sigma_epsilon, beta, sigma_eta, sigma_xi, m, Hs, n_tilde, H_tildes):
            E_beta = np.zeros(d)
            number_of_runs = args.decomposition_sub_runs
            for _ in range(number_of_runs):

                X = gen_data(n, cov_x, np.zeros(d))
                Zs = [gen_data(n_tilde, cov_z, np.zeros(d)) for i in range(m)]
                thetas = [gen_theta(beta, Hs[i], sigma_eta) for i in range(m)]
                Y = gen_Y(X, beta, sigma_epsilon)
                V = [gen_Y(Zs[i], thetas[i], sigma_xi) for i in range(m)]
                thetas_tildes = [l2_norm_solution(Zs[i], V[i]) for i in range(m)]
                sum_1 = np.sum([H_tilde.T @ H_tilde for H_tilde in H_tildes], axis=0)
                sum_2 = np.sum([H_tilde.T @ theta_tilde for H_tilde, theta_tilde in zip(H_tildes, thetas_tildes)], axis=0)
                alpha_tl_range = np.logspace(-4, 2, 20)
                best_err = float('inf')
                best_alpha = None
                for alpha_tl in alpha_tl_range:
                    beta_tilde = transfer_learning_solution(X, Y, sum_1, sum_2, alpha_tl)
                    err = norm(beta - beta_tilde)**2
                    if err < best_err:
                        best_err = err
                        best_alpha = alpha_tl
                beta_tilde = transfer_learning_solution(X, Y, sum_1, sum_2, best_alpha)
                E_beta += beta_tilde
            E_beta /= number_of_runs
            return E_beta

        def bias(E_beta, beta):
            bias_vec = E_beta - beta
            res =  bias_vec.T @ cov_x @ bias_vec

            return res

        def cov_beta(E_beta,n, d, sigma_epsilon, beta, sigma_eta, sigma_xi, m, Hs, n_tilde, H_tildes):
            Var = np.zeros((d,d))
            num_of_runs = args.decomposition_sub_runs
            for _ in range(num_of_runs):
                X = gen_data(n, cov_x, np.zeros(d))
                Zs = [gen_data(n_tilde, cov_z, np.zeros(d)) for j in range(m)]
                thetas = [gen_theta(beta, Hs[i], sigma_eta) for i in range(m)]
                Y = gen_Y(X, beta, sigma_epsilon)
                V = [gen_Y(Zs[i], thetas[i], sigma_xi) for i in range(m)]
                thetas_tildes = [l2_norm_solution(Zs[i], V[i]) for i in range(m)]
                sum_1 = np.sum([H.T @ H for H in H_tildes], axis=0)
                sum_2 = np.sum([H.T @ theta_tilde for H, theta_tilde in zip(H_tildes, thetas_tildes)], axis=0)
                alpha_tl_range = np.logspace(-4, 2, 25)
                best_err = float('inf')
                best_alpha = None
                for alpha_tl in alpha_tl_range:
                    beta_tilde = transfer_learning_solution(X, Y, sum_1, sum_2, alpha_tl)
                    err = norm(beta - beta_tilde)**2
                    if err < best_err:
                        best_err = err
                        best_alpha = alpha_tl
                beta_tilde = transfer_learning_solution(X, Y, sum_1, sum_2, best_alpha)
                Var += np.outer(beta_tilde - E_beta, beta_tilde - E_beta)
            Var /= num_of_runs
            return Var

        def variance(Cov):
            return np.trace(cov_x @ Cov)


        bias_list = []
        variance_list = []
        total_error_list = []
        time_start = time.time()
        x_values = (np.asarray(args.source_ratios) if args.source_ratios is not None
                    else np.linspace(0.1, 5.1, args.decomposition_points))
        src_size_range = [math.floor(d/i) for i in x_values]

        for _ in range(args.decomposition_main_runs):
            start = time.time()
            temp_bias_list = []
            temp_variance_list = []
            temp_total_error_list = []
            cov_matrix = (b/d) * np.eye(d)
            mean = np.zeros(d)  # Zero mean vector
            for n_tilde in src_size_range:
                beta = np.random.multivariate_normal(mean, cov_matrix)
                if (d > n_tilde) & debias:
                    debiased_H = [H_tilde / (d/n_tilde) for H_tilde in H_tildes]
                    E_beta = exp_beta(n,d,sigma_epsilon, beta, sigma_eta, sigma_xi, m, Hs, n_tilde, debiased_H)
                    B = bias(E_beta, beta)
                    C = cov_beta(E_beta,n,d,sigma_epsilon, beta, sigma_eta, sigma_xi, m, Hs, n_tilde, debiased_H)
                    V = variance(C)
                    temp_bias_list.append(B)
                    temp_variance_list.append(V)
                    temp_total_error_list.append(B+V+ sigma_epsilon**2)
                elif (d > n_tilde) & opt_debias:
                    S = anisotropic_shrinkage_matrix_ar1(cov_z, d/n_tilde)
                    debiased_H = [S @H_tilde for H_tilde in H_tildes]
                    E_beta = exp_beta(n,d,sigma_epsilon, beta, sigma_eta, sigma_xi, m, Hs, n_tilde, debiased_H)
                    B = bias(E_beta, beta)
                    C = cov_beta(E_beta,n,d,sigma_epsilon, beta, sigma_eta, sigma_xi, m, Hs, n_tilde, debiased_H)
                    V = variance(C)
                    temp_bias_list.append(B)
                    temp_variance_list.append(V)
                    temp_total_error_list.append(B+V+ sigma_epsilon**2)
                else:
                    E_beta = exp_beta(n,d,sigma_epsilon, beta, sigma_eta, sigma_xi, m, Hs, n_tilde, H_tildes)
                    B = bias(E_beta, beta)
                    C = cov_beta(E_beta,n,d,sigma_epsilon, beta, sigma_eta, sigma_xi, m, Hs, n_tilde, H_tildes)
                    V = variance(C)
                    temp_bias_list.append(B)
                    temp_variance_list.append(V)
                    temp_total_error_list.append(B+V+ sigma_epsilon**2)
            end = time.time()
            print(f'Run {_}, time taken {end - start}', flush=True)

            bias_list.append(temp_bias_list)
            variance_list.append(temp_variance_list)
            total_error_list.append(temp_total_error_list)
        bias_list = np.mean(bias_list, axis=0)
        variance_list = np.mean(variance_list, axis=0)
        total_error_list = np.mean(total_error_list, axis=0)
        Xs = [d / n_tilde for n_tilde in src_size_range]
        plt.plot(Xs, bias_list, label='Bias^2')
        plt.plot(Xs, variance_list, label='Variance')
        plt.plot(Xs, total_error_list, label='Total Error')
        plt.xlabel('d/n_tilde')
        plt.ylabel('Error Decomposition')
        plt.title('Error Decomposition vs d/n_tilde')
        plt.ylim(0, 1.4)
        plt.legend()
        output_path = f"{output_root}/graphs/{relation_description}_{time_start}_num_models={m}_{free_text}_{n}_r={args.r}_decomposition.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')  # High-quality image
        plt.tight_layout()
        final_result = np.stack((Xs, bias_list, variance_list, total_error_list), axis=0)
        target_dir = f"./decomposition_2/decomposition_{free_text}_{relation_description}_debias={debias}_assumption={assumption}_{exp_param}_{gamma_tgt}_{sigma_eta}_{sigma_xi}_{sigma_epsilon}_{num_runs}_cov={cov_x_des}_cov_z={cov_z_q}_cv={cv}_debias_factor_val={debias_factor_val}_opt_debias={opt_debias}_bad_models={bad_models}"
        os.makedirs(target_dir, exist_ok=True)
        exp_name = f"num_models={m}_debias={debias}_{relation_description}_tgt_param_{gamma_tgt}_sigama_eta={sigma_eta}_sigma_xi={sigma_xi}_sigma_epsilon={sigma_epsilon}_r={args.r}_decomposition.npy"
        np.save(os.path.join(target_dir, exp_name), final_result)

    if debias_factor_val and bad_models == False:
        print(Hs, flush = True)
        print(H_tildes, flush = True)
        results = []
        best_factor = []
        best_alpha = []
        with ProcessPoolExecutor(max_workers=num_cpus) as executor:
            futures = [executor.submit(single_run, src_size_range, experiment_args, child_seqs[_]) for _ in range(num_runs)]

        for future in futures:
            results.append(future.result()[0])
            best_factor.append(future.result()[1])
            best_alpha.append(future.result()[2])

        error = np.sum(results, axis=0)
        average_error = [e / num_runs for e in error]
        best_factors = np.sum(best_factor, axis=0)
        average_factor = [e / num_runs for e in best_factors]
        best_alphas = np.sum(best_alpha, axis=0)
        average_alpha = [e / num_runs for e in best_alphas]

        end = time.time()
        print("Time taken: ", end - start)
        Xs = [d / n_tilde for n_tilde in src_size_range]
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 3, 1)
        plt.plot(Xs, average_factor, color='green')
        plt.xlabel('d/n_tilde')
        plt.ylabel('Best Debias Factor')
        plt.title('Best Debias Factor vs d/n_tilde')
        plt.ylim(0, max(average_factor)*1.4)
        plt.subplot(1, 3, 2)
        plt.plot(Xs, average_error, color='orange')
        plt.xlabel('d/n_tilde')
        plt.ylabel('Error')
        plt.title('Error vs d/n_tilde')
        plt.ylim(0, 1.4)
        plt.subplot(1, 3, 3)
        plt.plot(Xs, average_alpha, color='blue')
        plt.xlabel('d/n_tilde')
        plt.ylabel('Best Alpha')
        plt.yscale('log')
        plt.title('Best Alpha vs d/n_tilde')
        plt.ylim(0, max(average_alpha)*1.4)
        plt.suptitle('Debias Factor, Error, and Alpha vs d/n_tilde')
        output_path = f"{output_root}/graphs/debias_factor_and_error_{free_text}_{relation_description}_{exp_param}_num_models={m}_{n}_{end}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', transparent=True)  # High-quality image
        target_dir = f"./results/{free_text}_{relation_description}_debias_factor_{exp_param}_{gamma_tgt}_{sigma_eta}_{sigma_xi}_{sigma_epsilon}_{num_runs}_cov={cov_x_des}_cov_z={cov_z_q}_cv={cv}_debias_factor_val={debias_factor_val}_opt_debias={opt_debias}_bad_models={bad_models}"
        os.makedirs(target_dir, exist_ok=True)
        final_result = np.stack((Xs, average_error, average_factor, average_alpha), axis=0)
        exp_name = file_name + "_" +   f"{relation_description}_num_runs={num_runs}_tgt_param_{gamma_tgt}_num_models={m}_sigama_eta={sigma_eta}_sigma_xi={sigma_xi}_sigma_epsilon={sigma_epsilon}_rotation={rotation}_p={p}_r={args.r}.npy"
        file_path = os.path.join(target_dir, exp_name)
        np.save(file_path, final_result)


    if bad_models == True:
        make_dir = f"./results/bad_models_{free_text}_debias={debias}_debias_factor_val={debias_factor_val}_cov_z={cov_z_q}_num_models={m}_{n}_num_runs={num_runs}"
        os.makedirs(make_dir, exist_ok=True)
        for k in [0, 2, 4, 6, 8, 10]:
            time_start = time.time()
            if relation_description == 'subspace':
                H_bad =[generate_random_subspace(d,32) for _ in range(k)]
                H_god = [generate_random_subspace(d,96) for _ in range(20-k)]
                Hs = H_bad + H_god
            if relation_description == 'eng_subspace':
                H_bad = [generate_random_energy_subspace(d,32) for _ in range(k)]
                H_god = [generate_random_energy_subspace(d,96) for _ in range(20-k)]
                Hs = H_bad + H_god
            H_tildes = [np.eye(d) for _ in range(20)]
            results = []
            best_alpha = []
            experiment_args = {
                'n': n,
                'd': d,
                'cov_x': cov_x,
                'sigma_eta': sigma_eta,
                'sigma_epsilon': sigma_epsilon,
                'sigma_xi': sigma_xi,
                'b': b,
                'H_tildes': H_tildes,
                'Hs': Hs,
                'alpha_tl_range': alpha_tl_range,
                'n_test': 1000
            }
            with ProcessPoolExecutor(max_workers=num_cpus) as executor:
                futures = [executor.submit(single_run, src_size_range, experiment_args) for _ in range(num_runs)]

                for future in futures:
                    results.append(future.result()[0])
                    best_alpha.append(future.result()[1])
            stds = np.std(results, axis=0)
            error = np.sum(results, axis=0)
            average_error = [e / num_runs for e in error]
            Xs = [d / n_tilde for n_tilde in src_size_range]
            plt.plot(Xs, average_error, label=f'{k} bad models')
            final_result = np.stack((Xs, average_error), axis=0)
            exp_name = f"num_models={m}_bad_models={k}_debias={debias}_debias_factor_val={debias_factor_val}_{relation_description}_tgt_param_{gamma_tgt}_sigama_eta={sigma_eta}_sigma_xi={sigma_xi}_sigma_epsilon={sigma_epsilon}_r={args.r}.npy"
            file_path = os.path.join(make_dir, exp_name)
            np.save(file_path, final_result)
            end = time.time()
            time_taken = end - time_start
            print(f'Finished {k} bad models in {time_taken:.2f} seconds', flush=True)

        ridge_err = ridge_regression_error(n,d,sigma_epsilon,b)
        plt.axhline(y=ridge_err, color='red', linestyle='--', label='Ridge Regression Error')
        plt.xlabel('d/n_tilde')
        plt.ylabel('Error')
        plt.title('Error vs d/n_tilde with bad models')
        plt.ylim(0, 1.4)
        plt.legend()
        output_path = f"{output_root}/graphs/bad_models_{free_text}_debias={debias}_debias_factor_val={debias_factor_val}_cov_z={cov_z_q}_num_models={m}_{n}_num_runs={num_runs}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')  # High-quality image
