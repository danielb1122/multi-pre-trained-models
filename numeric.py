"""Numerical evaluation of transfer-learning theory with multiple pretrained models."""
import argparse
import os
os.environ.setdefault("MPLBACKEND", "Agg")
from paper_utils import add_output_argument, prepare_output, str2bool
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import Normalize, PowerNorm
import numpy as np
from scipy.linalg import inv, norm, circulant
from scipy.optimize import fsolve, root_scalar
import os
from datetime import datetime
matplotlib.use('Agg')
from scipy.linalg import toeplitz
import seaborn as sns
parser = argparse.ArgumentParser()
from pathlib import Path
import math
import scipy

parser.add_argument('--b', type=float, default=1, help="Value for b")
parser.add_argument('--number_of_models', type=int, nargs='+', default=[1,2,10,20], help="Number of models")
parser.add_argument('--relation_description', type=str, default='identity')
parser.add_argument('--rotation', type=float, default=1/16)
parser.add_argument('--ones_ratio', type=float, default=0.75)
parser.add_argument('--cov_x_des', type=str, default='identity', help="Covariance matrix for x")
parser.add_argument('--free_text', type=str, default=None, help="Free text for output files name")
parser.add_argument('--k', type=float, default=2)
parser.add_argument('--sum', type=str2bool, default=False)
parser.add_argument('--scalar', type=float, default=1, help="Scalar for the H_tilde matrix")
parser.add_argument('--r', type=int, nargs='?', default=128, help="r subspace dimension, default is dimension of the data")
parser.add_argument('--hm', type=str2bool, default=False, help="heat map experiment")
parser.add_argument('--m_hm', type = int, default=10, help="number of models for heat map experiment")
parser.add_argument('--assumption', type=str2bool, default=False, help="h_tilde = h assumption")
parser.add_argument('--kappa', type = float, default=10, help="condition number for circulant matrices")
parser.add_argument('--permutation', type=str2bool, default=False, help="permutation for circulant matrices")
parser.add_argument('--wker', type=float, default=1/150, help="Kernel width for circulant matrix")
parser.add_argument('--radius', type=float, default=0.1, help="Radius for simplex circulant matrices")
parser.add_argument('--decay', type=float, default=0.2718, help="Decay for circulant_from_eigs")
parser.add_argument('--peak', type=float, default=0.5, help="Peak for circulant_from_eigs")


parser.add_argument('--sigma_eta', type=float, default=0.31623, help="Sigma eta values")
parser.add_argument('--sigma_epsilon', type=float, default=0.31623, help="Sigma epsilon values")
parser.add_argument('--sigma_xi', type=float, default=0.31623, help="Sigma xi values")
parser.add_argument('--gamma_tgt', type=float, nargs='+', default=[4], help="Gamma target values")
parser.add_argument('--gamma_src', type=float, nargs='+', default=[1.2], help="Gamma source values")


parser.add_argument('--empirical_experiment', type=str2bool, default=False, help="Run empirical experiment")
parser.add_argument('--number_of_runs', type=int, default=100, help="Number of runs")
parser.add_argument('--n_tilde', type=int, nargs='+', default=[128], help="n tilde values")
parser.add_argument('--n', type=int, default=32, help="n values")
parser.add_argument('--d', type=int, default=128, help="d values")
parser.add_argument('--debias', type=str2bool, default=False)
parser.add_argument('--bad_models', type=str2bool, default = False)
add_output_argument(parser)
parser.add_argument('--source-ratios', type=float, nargs='+', default=None)
parser.add_argument('--alpha-points', type=int, default=20)
args = parser.parse_args()
if args.source_ratios and any(x <= 0 or x == 1 for x in args.source_ratios):
    parser.error('the theoretical curve requires positive source-ratios other than 1')
if args.alpha_points < 1:
    parser.error('alpha-points must be positive')
output_root = prepare_output(args, experiment=True)

relation_description = args.relation_description
assumption = args.assumption
rotation = args.rotation
b = args.b
p = args.ones_ratio
k = args.k
r = args.r
kappa = args.kappa
permutation = args.permutation
scalar = args.scalar
sum_sample = args.sum
cov_x_des = args.cov_x_des
number_of_models = args.number_of_models
sigma_eta = args.sigma_eta
sigma_epsilon = args.sigma_epsilon
sigma_xi = args.sigma_xi
gamma_src = args.gamma_src
bad_models = args.bad_models
empirical_experiment = args.empirical_experiment
number_of_runs = args.number_of_runs
n_tilde = args.n_tilde
n = args.n
d = args.d
debias = args.debias

free_text = args.free_text
exp_param = None
hm = args.hm
m_hm = args.m_hm

wker = args.wker
radius = args.radius
peak = args.peak
decay = args.decay


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
if relation_description == 'circulant_2':
    exp_param = wker
if relation_description == 'symmetric_circulant':
    exp_param = peak
if relation_description == 'filter':
    exp_param = 1/wker

def calculate_R(h_tildes):
    A = sum(h_tilde.T @ h_tilde for h_tilde in h_tildes)
    R = scipy.linalg.sqrtm(A)
    return R

def compute_WM_inv(c, alpha_TL, W, d):
    """
    Computes T = W @ (c*W + alpha_TL*I_d)^-1 using stable linear solving.
    """
    I_d = np.eye(d)
    M = c * W + alpha_TL * I_d

    try:
        T = np.linalg.solve(M, W.T).T
        return T
    except np.linalg.LinAlgError as e:
        print(f"Warning: np.linalg.solve failed due to singular matrix M. Error: {e}")
        raise

def calculate_W(R, SIGMA_X):
    return np.linalg.inv(R) @  SIGMA_X @ np.linalg.inv(R.T)


def first_sum_term_under(h_tilde, sigma_eta, gamma_src, sigma_xi, b, h):
    term = (sigma_eta ** 2 + gamma_src * sigma_xi ** 2 / (1 - gamma_src)) * np.eye(d, d) + b * (h - h_tilde) @ (
            h - h_tilde).T
    return h_tilde.T @ term @ h_tilde


def calculate_gamma_under_param_regime(d, R, h_tildes, sigma_etas, gamma_src, sigma_xis, b, hs):
    temp_1 = sum(
        first_sum_term_under(h_tilde, sigma_eta, gamma_src, sigma_xi, b, h) for h_tilde, sigma_eta, sigma_xi, h in
        zip(h_tildes, sigma_etas, sigma_xis, hs))

    double_sum = 0

    for i in range(len(hs)):
        for j in range(len(hs)):
            if i != j:
                double_sum += h_tildes[i].T @ (hs[i] - h_tildes[i]) @ (hs[j] - h_tildes[j]).T @ h_tildes[j]
    return 1 / d * np.linalg.inv(R) @ (temp_1 + b * double_sum) @ np.linalg.inv(R)


def first_sum_term_over(h_tilde, sigma_eta, gamma_src, sigma_xi, b, h):
    term = b * (gamma_src - 1) / (d * gamma_src ** 2) * (
            gamma_src * h_tilde @ h_tilde.T - h @ h.T + 1 / d * np.linalg.norm(h, ord='fro') ** 2 * np.eye(d,
                                                                                                           d) - 1 / d * np.diag(
        np.diag(h @ h.T))) + b / (d * gamma_src) * (h - h_tilde) @ (h - h_tilde).T + 1 / (d * gamma_src) * (
                   sigma_eta ** 2 + gamma_src * sigma_xi ** 2 / (gamma_src - 1)) * np.eye(d, d)
    return h_tilde.T @ term @ h_tilde


def calculate_gamma_over_param_regime(d, R, h_tildes, sigma_etas, gamma_src, sigma_xis, b, hs):
    temp_1 = sum(
        first_sum_term_over(h_tilde, sigma_eta, gamma_src, sigma_xi, b, h) for h_tilde, sigma_eta, sigma_xi, h in
        zip(h_tildes, sigma_etas, sigma_xis, hs))

    double_sum = 0
    for i in range(len(hs)):
        for j in range(len(hs)):
            if i != j:
                double_sum += h_tildes[i].T @ (1/gamma_src * hs[i] - h_tildes[i]) @ (1/gamma_src * hs[j] - h_tildes[j]).T @ \
                              h_tildes[j]

    return np.linalg.inv(R) @ (temp_1 + b / d * double_sum) @ np.linalg.inv(R)


def equation_for_c(c, alpha_TL, W, gamma_tgt, d):
    term = np.linalg.inv(c * W + alpha_TL * np.eye(d, d))
    trace_value = np.trace(W @ term)
    return 1 / c - 1 - (gamma_tgt / d) * trace_value


def calculate_c(alpha_TL, W, gamma_tgt, d):
    sol_u = fsolve(lambda u: equation_for_c(np.exp(u), alpha_TL, W, gamma_tgt, d), x0=0.0)[0]
    return float(np.exp(sol_u))


def solve_c(alpha_TL, gamma_tgt, W, d, c_initial_guess = 0.1):
    """
    Solves for c(alpha_TL) given alpha_TL, gamma_tgt, W, and d.

    Parameters:
    alpha_TL (float): Given value of alpha_TL.
    gamma_tgt (float): Gamma target constant.
    W (numpy.ndarray): The matrix W (should be square and symmetric).
    d (int): Dimension of the identity matrix I_d.

    Returns:
    float: Solution for c(alpha_TL).
    """
    def equation(c):
        I_d = np.eye(d)
        T = compute_WM_inv(c, alpha_TL, W, d)
        term = np.trace(T)
        rhs = 1 + (gamma_tgt / d) * term
        return 1 / c - rhs


    c_solution = root_scalar(equation, bracket=[5*10**(-9), 100], method='bisect').root
    return c_solution

def calculate_c_tag(c, alpha_TL, W, gamma_tgt, d):
    T = compute_WM_inv(c, alpha_TL, W, d)
    fro_norm_sq = np.trace(T.T @ T)
    numerator = (gamma_tgt / d) * fro_norm_sq
    denominator = (c ** -2 - numerator)

    return numerator / denominator


def calculate_error(param_level, alpha_TL, c, c_tag, W, R, b, gamma_tgt, gamma_src, d, sigma_etas, sigma_xis,
                    sigma_epsilon, h_tildes, hs):
    if debias == True  and param_level > 1:
        h_tildes = [1/param_level * _ for _ in h_tildes]
        R = calculate_R(h_tildes)
        W = calculate_W(R, sigma_x)
        c = solve_c(alpha_TL, gamma_tgt, W, d)
        c_tag = calculate_c_tag(c, alpha_TL, W, gamma_tgt, d)

    if c < 0 or c_tag < 0:
        return float('inf')
    if param_level < 1:
        Gamma = calculate_gamma_under_param_regime(d, R, h_tildes, sigma_etas, gamma_src, sigma_xis, b, hs)
    else:
        Gamma = calculate_gamma_over_param_regime(d, R, h_tildes, sigma_etas, gamma_src, sigma_xis, b, hs)

    first_term = gamma_tgt * np.trace(1 / d * W @ inv(c * W + alpha_TL * np.eye(d)))
    second_term = gamma_tgt * np.trace(
        (alpha_TL**2 / (gamma_tgt * sigma_epsilon ** 2) * Gamma - alpha_TL / d * np.eye(d, d)) @ inv(
            c * W + alpha_TL * np.eye(d)) @ ((c_tag + 1) * W) @ inv(c * W + alpha_TL * np.eye(d)))
    return sigma_epsilon ** 2 * (1 + first_term + second_term)

def circulant_matrix_gaussian_filter(d, conv_gaussian_width):

    first_half = np.exp(-conv_gaussian_width * np.linspace(0, 1, d // 2))

    second_half = np.exp(-conv_gaussian_width * np.linspace(1, 0, -(-d // 2)))

    H_circular_first_row = np.concatenate([first_half, second_half])

    H_circular_first_row[0] += 1.0


    H_circular_first_row = H_circular_first_row / np.sqrt(np.sum(H_circular_first_row**2))

    H_circular_first_column = np.concatenate([[H_circular_first_row[0]], H_circular_first_row[1:][::-1]])

    H = toeplitz(H_circular_first_column, H_circular_first_row)

    return H

def generate_symmetric_circulant(d = 128, decay = 0.5,peak = 0.5):
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


def generate_batch_matrix(d,i,j):
    a = np.zeros(i)
    b = np.ones(j-i)
    c = np.zeros(d-j)
    d = np.concatenate([a,b,c])
    return np.diag(d)


def generate_seperate_batch_matrices(d,m):
    batch_size = int(d/m)
    return [generate_batch_matrix(d, i*batch_size, (i+1)*batch_size) for i in range(m)]

def generate_random_scaled_subspace(d, r):
    """
    Samples a uniformly random r-dimensional subspace of R^d.

    Returns:
    - Q: (d x r) matrix whose columns form an orthonormal basis for the subspace
    - P: (d x d) orthogonal projection matrix onto the subspace
    - project_fn: function that projects a given vector x ∈ R^d onto the subspace
    """
    A = np.random.randn(d, r)

    Q, _ = np.linalg.qr(A)

    alp = 10
    diag = np.random.gamma(alp, 1/alp, r)
    M = np.diag(diag)

    P = Q @ M @ Q.T

    return P

def generate_random_energy_subspace(d, r):
    """
    Samples a uniformly random r-dimensional subspace of R^d.

    Returns:
    - Q: (d x r) matrix whose columns form an orthonormal basis for the subspace
    - P: (d x d) orthogonal projection matrix onto the subspace
    - project_fn: function that projects a given vector x ∈ R^d onto the subspace
    """
    A = np.random.randn(d, r)

    Q, _ = np.linalg.qr(A)

    scalar = np.sqrt(d / r)  # Scale factor to ensure the projection matrix has the right trace
    P =scalar * Q @ Q.T

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

def generate_matrices(d, m, func):
    if func == "circulant":
        random_wker = np.random.uniform(2 / 75, 2 / 65, m)
        hs = [generate_circulant_matrix(d, wker) for wker in random_wker]
        return hs

def generate_random_orthonormal_matrix(d):
    return np.linalg.qr(np.random.randn(d, d))[0]

def generate_orthonormal_matrices(d, m):
    return [generate_random_orthonormal_matrix(d) for _ in range(m)]


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
    angles = np.random.uniform(-theta, theta, size=num_planes)

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

    sign = np.random.choice([-1, 1])
    angle = sign * theta

    v1 = np.random.randn(d)
    v1 /= np.linalg.norm(v1)

    v2 = np.random.randn(d)
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
    return [generate_random_rotation_matrix(d,theta) for _ in range(m)]


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

    chosen_indices = np.random.choice(diagonal_indices, size=num_ones, replace=False)

    matrix[chosen_indices, chosen_indices] = 1

    return matrix

def generate_random_subspace(d, r):
    """
    Samples a uniformly random r-dimensional subspace of R^d.

    Returns:
    - Q: (d x r) matrix whose columns form an orthonormal basis for the subspace
    - P: (d x d) orthogonal projection matrix onto the subspace
    - project_fn: function that projects a given vector x ∈ R^d onto the subspace
    """
    A = np.random.randn(d, r)

    Q, _ = np.linalg.qr(A)

    P = Q @ Q.T

    return P

def generate_random_diagonal_blocks(p,d):
    matrix = np.zeros((d, d), dtype=int)
    num_ones = int(p * d)
    start_of_block_idx = np.random.randrange(d)

def generate_close_to_identity(n, epsilon=0.075):
    R = np.random.randn(n, n)
    H = R - R.T  # This makes it skew-symmetric: H^T = -H

    A = np.eye(n) + epsilon * H

    frob_identity = np.sqrt(n)
    frob_A = np.linalg.norm(A, 'fro')
    A_scaled = A * (frob_identity / frob_A)

    return A_scaled


gamma_tgt = d / n


erros = []

output_directory = f"{output_root}/numeric_result_icecream/{free_text}_{relation_description}_{exp_param}_debias={debias}_assumption={assumption}_{gamma_tgt}_{sigma_eta}_{sigma_xi}_{sigma_epsilon}_cov={cov_x_des}"  # Change to your desired path
if not os.path.exists(output_directory):
    os.makedirs(output_directory)


def min_l2_solution_for_overparam(n,d,sigma_epsilon,b):
    return (1+n/(d-n-1))*sigma_epsilon**2+(1-n/d)*b

def m(alpha_opt_infinity, gamma_tgt):
    part1 = (1 - gamma_tgt + alpha_opt_infinity)
    part2 = np.sqrt((1 - gamma_tgt + alpha_opt_infinity) ** 2 + 4 * gamma_tgt * alpha_opt_infinity)
    return (-part1 + part2) / (2 * gamma_tgt * alpha_opt_infinity)

def ridge_regression_error(n, d, sigma_epsilon, b):
    gamma_tgt = d/n
    sigma_opt = sigma_epsilon ** 2 / b

    alpha_opt_infinity = sigma_opt * gamma_tgt


    temp = m(alpha_opt_infinity, gamma_tgt)

    generalization_error = sigma_epsilon ** 2 * (1 + gamma_tgt * temp)

    return generalization_error


ridge_err = ridge_regression_error(n,d,sigma_epsilon,b)
plt.axhline(y=ridge_err, color='blue', linestyle='--', label='Optimal Ridge Solution')

l2_over_solutin = min_l2_solution_for_overparam(n,d,sigma_epsilon,b)
plt.axhline(y=l2_over_solutin, color='black', linestyle='--', label='L2 Norm Solution')

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


def scale_to_trace(Sigma, target_trace):
    """
    Scale a covariance matrix so that its trace equals target_trace.
    """
    current_trace = np.trace(Sigma)
    if current_trace == 0:
        raise ValueError("Current trace is zero; cannot scale matrix.")
    return Sigma * (target_trace / current_trace)

def generate_isotropic(p, trace=None):
    """
    Generates an isotropic covariance matrix (identity) of dimension p.
    Optionally scales to have a given trace.
    """
    Sigma = np.eye(p)
    if trace is not None:
        Sigma = scale_to_trace(Sigma, trace)
    return Sigma

def generate_toeplitz(p, rho, trace=None):
    """
    Generates an AR(1) Toeplitz covariance of dimension p with correlation rho.
    Optionally scales to have a given trace.
    """
    idx = np.arange(p)
    Sigma = rho ** np.abs(idx[:, None] - idx[None, :])
    if trace is not None:
        Sigma = scale_to_trace(Sigma, trace)
    return Sigma

def generate_spiked(p, r, spike_vals, sigma=1.0, trace=None, random_state=None):
    """
    Generates a spiked covariance matrix of dimension p:
    - r: number of spikes
    - spike_vals: sequence of r eigenvalues
    - sigma: noise std on diagonal
    Optionally scales to have a given trace.
    """
    if random_state is not None:
        np.random.seed(random_state)
    Q, _ = np.linalg.qr(np.random.randn(p, p))
    spikes = np.zeros(p)
    spikes[:r] = spike_vals
    Sigma = Q @ np.diag(spikes) @ Q.T + sigma**2 * np.eye(p)
    if trace is not None:
        Sigma = scale_to_trace(Sigma, trace)
    return Sigma

def generate_power_law(p, alpha, scale=1.0, trace=None):
    """
    Generates a diagonal covariance with polynomial eigen-decay:
    lambda_i ∝ scale * i^(-alpha).
    Optionally scales to have a given trace.
    """
    lambdas = scale * (np.arange(1, p+1) ** (-alpha))
    Sigma = np.diag(lambdas)
    if trace is not None:
        Sigma = scale_to_trace(Sigma, trace)
    return Sigma

def generate_exp_decay(p, c, scale=1.0, trace=None):
    """
    Generates a diagonal covariance with exponential eigen-decay:
    lambda_i ∝ scale * exp(-c*i).
    Optionally scales to have a given trace.
    """
    lambdas = scale * np.exp(-c * np.arange(p))
    Sigma = np.diag(lambdas)
    if trace is not None:
        Sigma = scale_to_trace(Sigma, trace)
    return Sigma

def generate_block_diag(block_matrices, trace=None):
    """
    Constructs a block-diagonal covariance from a list of matrices.
    Optionally scales to have a given trace.
    """
    Sigma = block_diag(*block_matrices)
    if trace is not None:
        Sigma = scale_to_trace(Sigma, trace)
    return Sigma

def anti_isotropic_covariance_matrix(d,h=0,l=-1):
    lambdas = np.logspace(h, l, d)  # exponentially decaying eigenvalues
    return np.diag(lambdas)


if not hm and not bad_models:
    for m in number_of_models:
        print(m, flush=True)
        erros = []
        param = []
        ns = []
        if relation_description == 'identity':
            hs = [np.eye(d, d) for _ in range(m)]
        if relation_description == 'rotation':
            hs = generate_rotation_matrices(d, m, np.pi*rotation)

        if relation_description == 'seperate_batch':
            hs = generate_seperate_batch_matrices(d,m)
        if relation_description == 'random_ratio_ones':
            hs = [generate_random_diagonal_matrix(p,d) for _ in range(m)]
        if relation_description == 'scalar':
            hs = [k * np.eye(d, d) for _ in range(m)]
        if relation_description == 'non_random_rotation':
            mtx = generate_random_rotation_matrix(d,np.pi*rotation)
            hs = [mtx for _ in range(m)]
        if relation_description == 'close_to_id':
            mtx = generate_close_to_identity(d)
            hs = [mtx for _ in range(m)]

        if relation_description == 'scaled_circulant':
            us = np.random.uniform(-0.5, 0.5, m)
            hs = [generate_scaled_circulant_matrix(d, 2 / 75, us[_]) for _ in range(m)]

        if relation_description == 'subspace':
            hs = [generate_random_subspace(d, args.r) for _ in range(m)]

        if relation_description == 'scaled_subspace':
            hs = [generate_random_scaled_subspace(d, args.r) for _ in range(m)]

        if relation_description == 'eng_subspace':
            hs = [generate_random_energy_subspace(d, args.r) for _ in range(m)]

        if relation_description == 'circulant':
            hs = generate_circulants_matrices_with_condition_number((args.kappa),d, m, permutation = args.permutation)

        if relation_description == 'circulant_2':
            hs = [generate_circulant_matrix(d,wker) for _ in range(m)]

        if relation_description == 'symmetric_circulant':
            h = generate_symmetric_circulant(d, decay = args.decay, peak = args.peak)
            hs = [h for _ in range(m)]

        if relation_description == 'filter':
            h = circulant_matrix_gaussian_filter(d,1/wker)
            hs = [h for _ in range(m)]


        if assumption == False:
            h_tildes = [scalar*np.eye(d, d) for _ in range(m)]
        else:
            h_tildes = hs
        for i in range(len(hs)):
            print(f'{i} H matrix is:{hs[i]}')
        for i in range(len(hs)):
            print(f'{i} H_tilde matrix is:{h_tildes[i]}')
        R = calculate_R(h_tildes)
        if cov_x_des == 'identity':
            sigma_x = np.eye(d, d)
        if cov_x_des == 'exp_decay':
            sigma_x = exponential_covariance_matrix(d)
        if cov_x_des == 'toeplitz':
            sigma_x = generate_toeplitz(d, rho=0.5)
        if cov_x_des == 'isotropic':
            sigma_x = generate_isotropic(p=d, trace=d)
        if cov_x_des == 'spiked':
            sigma_x = generate_spiked(p=d, r=int(d/10), spike_vals=np.linspace(1, 10, int(d/10)), sigma=1.0, trace=d)
        if cov_x_des == 'power_law':
            sigma_x = generate_power_law(p=d, alpha=1.2, scale=1.0, trace=d)
        if cov_x_des == 'anti_isotropic':
            sigma_x = anti_isotropic_covariance_matrix(d, h=0, l=-0.5)

        W = calculate_W(R, sigma_x)
        sigma_etas = np.ones(m) * sigma_eta
        sigma_xis = np.ones(m) * sigma_xi

        c_dict = {}
        c_tag_dict = {}
        alpha_tl_range = np.logspace(-4, 2.5, args.alpha_points)
        print(alpha_tl_range, flush=True)
        for alpha_TL in alpha_tl_range:
            c = solve_c(alpha_TL, gamma_tgt, W, d)
            c_tag = calculate_c_tag(c, alpha_TL, W, gamma_tgt, d)
            if c < 0 or c_tag < 0:
                print(
                    f"for alpha = {alpha_TL} and number of models = {m}, c or c tag is negative. c = {c}, c_tag = {c_tag}")
            else:
                c_dict[alpha_TL] = c
                c_tag_dict[alpha_TL] = c_tag

        print(c_dict, flush=True)
        print(c_tag_dict, flush=True)
        x_values = np.linspace(0.1, 5.1, 40)  # Avoiding division by zero
        if args.source_ratios is not None:
            x_values = np.asarray(args.source_ratios)
        n_sample = d / x_values
        for n_tilde in n_sample:
            TEMP_ERRORS = []
            TEMP_ALPHAS = []
            param_lvl = d / n_tilde
            gamma_src = param_lvl

            for alpha_TL in alpha_tl_range:
                if c_dict.get(alpha_TL):
                    c = c_dict.get(alpha_TL)
                    c_tag = c_tag_dict.get(alpha_TL)
                    if sum_sample == True:
                        error = calculate_error(param_lvl*m, alpha_TL, c, c_tag, W, R, b, gamma_tgt, gamma_src*m, d, sigma_etas, sigma_xis,                                sigma_epsilon, h_tildes, hs)
                    else:
                        error = calculate_error(param_lvl, alpha_TL, c, c_tag, W, R, b, gamma_tgt, gamma_src, d, sigma_etas, sigma_xis,
                                                sigma_epsilon, h_tildes, hs)
                    TEMP_ERRORS.append(error)
                    TEMP_ALPHAS.append(alpha_TL)
            i = np.argmin(TEMP_ERRORS)
            best_error_for_param_lvl = min(TEMP_ERRORS)
            best_alpha = TEMP_ALPHAS[i]
            print(f"for gamma_src = {d/n_tilde}, best alpha = {best_alpha}", flush=True)
            erros.append(best_error_for_param_lvl)
            param.append(param_lvl)
            ns.append(n_tilde)


        results = np.array([param, erros])  # First row: param_lvl, Second row: errors
        output_file = os.path.join(output_directory, f"debias={debias}_{free_text}_{relation_description}_assumingH={assumption}_tgt_param_{gamma_tgt}_num_models_{m}_sigama_eta={sigma_eta}_sigma_xi={sigma_xi}_sigma_epsilon={sigma_epsilon}_r={args.r}.npy")
        np.save(output_file, results)
        print(f"Ended  m= {m}")
        if sum_sample == True:
            plt.plot(ns, erros, label=f"m={m}")
        else:
            plt.plot(param, erros, label=f"m={m}")

    output_file = os.path.join(output_directory, f"ridge_{relation_description}_tgt_param_{gamma_tgt}_ratio_{p}_rotation_{rotation}_results_num_models_{0}_sigama_eta={sigma_eta}_sigma_xi={sigma_xi}_sigma_epsilon={sigma_epsilon}_r={args.r}.npy")
    np.save(output_file, ridge_err)

    output_file = os.path.join(output_directory, f"l2_{relation_description}_tgt_param_{gamma_tgt}_ratio_{p}_rotation_{rotation}_results_num_models_{0}_sigama_eta={sigma_eta}_sigma_xi={sigma_xi}_sigma_epsilon={sigma_epsilon}_r={args.r}.npy")
    np.save(output_file, l2_over_solutin)


    plt.ylim(0, 1.6)


    plt.xlabel("Source tasks parameterization (x-axis)", fontsize=12)
    plt.ylabel("Expected Test Error (Target Task)", fontsize=12)
    plt.title("Comparison of Test Error Across Models", fontsize=14)
    plt.legend(fontsize=10, loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    plt.grid(alpha=0.5)

    current_time = datetime.now()

    timestamp = current_time.strftime("%m-%d_%H-%M")


    output_path = f"{output_root}/graphs/{timestamp}_only_analitic_plot_debias={debias}_{relation_description}_assumingH={assumption}_expparam={exp_param}_tgt_param_{gamma_tgt}_sigama_eta={sigma_eta}_sigma_xi={sigma_xi}_sigma_epsilon={sigma_epsilon}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')  # High-quality image

    plt.tight_layout()


if bad_models == True:
    output_directory = f"{output_root}/numeric_result_icecream/{bad_models}_{relation_description}_{exp_param}_debias={debias}_assumption={assumption}_{gamma_tgt}_{sigma_eta}_{sigma_xi}_{sigma_epsilon}_cov={cov_x_des}"  # Change to your desired path

    for k in range(0,11, 2):
        print(f"bad model iteration {k}", flush=True)
        errors = []
        param = []
        ns = []
        h_god = [generate_random_energy_subspace(d, 96) for _ in range(20-k)]
        h_bad = [generate_random_energy_subspace(d, 32) for _ in range(k)]
        hs = h_god + h_bad
        if assumption == False:
            h_tildes = [scalar*np.eye(d, d) for _ in range(m)]
        else:
            h_tildes = hs
        for i in range(len(hs)):
            print(f'{i} H matrix is:{hs[i]}')
        for i in range(len(hs)):
            print(f'{i} H_tilde matrix is:{h_tildes[i]}')
        R = calculate_R(h_tildes)
        if cov_x_des == 'identity':
            sigma_x = np.eye(d, d)
        if cov_x_des == 'exp_decay':
            sigma_x = exponential_covariance_matrix(d)
        if cov_x_des == 'toeplitz':
            sigma_x = generate_toeplitz(d, rho=0.5)
        if cov_x_des == 'isotropic':
            sigma_x = generate_isotropic(p=d, trace=d)
        if cov_x_des == 'spiked':
            sigma_x = generate_spiked(p=d, r=int(d/10), spike_vals=np.linspace(1, 10, int(d/10)), sigma=1.0, trace=d)
        if cov_x_des == 'power_law':
            sigma_x = generate_power_law(p=d, alpha=1.2, scale=1.0, trace=d)
        if cov_x_des == 'anti_isotropic':
            sigma_x = anti_isotropic_covariance_matrix(d, h=0, l=-0.5)

        W = calculate_W(R, sigma_x)
        sigma_etas = np.ones(m) * sigma_eta
        sigma_xis = np.ones(m) * sigma_xi

        c_dict = {}
        c_tag_dict = {}


        print(c_dict, flush=True)
        print(c_tag_dict, flush=True)
        x_values = np.linspace(0.1, 5.1, 40)  # Avoiding division by zero
        n_sample = d / x_values
        alpha_tl_range = np.logspace(-4, 2.5, 20)
        print(alpha_tl_range, flush=True)
        for alpha_TL in alpha_tl_range:
            c = solve_c(alpha_TL, gamma_tgt, W, d)
            c_tag = calculate_c_tag(c, alpha_TL, W, gamma_tgt, d)
            if c < 0 or c_tag < 0:
                print(
                    f"for alpha = {alpha_TL} and number of models = {m}, c or c tag is negative. c = {c}, c_tag = {c_tag}")
            else:
                c_dict[alpha_TL] = c
                c_tag_dict[alpha_TL] = c_tag

        print(c_dict, flush=True)
        print(c_tag_dict, flush=True)
        x_values = np.linspace(0.1, 5.1, 40)  # Avoiding division by zero
        n_sample = d / x_values
        for n_tilde in n_sample:
            TEMP_ERRORS = []
            TEMP_ALPHAS = []
            param_lvl = d / n_tilde
            gamma_src = param_lvl

            for alpha_TL in alpha_tl_range:
                if c_dict.get(alpha_TL):
                    c = c_dict.get(alpha_TL)
                    c_tag = c_tag_dict.get(alpha_TL)
                    if sum_sample == True:
                        error = calculate_error(param_lvl*m, alpha_TL, c, c_tag, W, R, b, gamma_tgt, gamma_src*m, d, sigma_etas, sigma_xis,                                sigma_epsilon, h_tildes, hs)
                    else:
                        error = calculate_error(param_lvl, alpha_TL, c, c_tag, W, R, b, gamma_tgt, gamma_src, d, sigma_etas, sigma_xis,
                                                sigma_epsilon, h_tildes, hs)
                    TEMP_ERRORS.append(error)
                    TEMP_ALPHAS.append(alpha_TL)
            i = np.argmin(TEMP_ERRORS)
            best_error_for_param_lvl = min(TEMP_ERRORS)
            best_alpha = TEMP_ALPHAS[i]
            print(f"for gamma_src = {d/n_tilde}, best alpha = {best_alpha}", flush=True)
            erros.append(best_error_for_param_lvl)
            param.append(param_lvl)
            ns.append(n_tilde)


        results = np.array([param, erros])  # First row: param_lvl, Second row: errors
        output_file = os.path.join(output_directory, f"debias={debias}_{free_text}_{relation_description}_assumingH={assumption}_tgt_param_{gamma_tgt}_num_bad_models_{k}_sigama_eta={sigma_eta}_sigma_xi={sigma_xi}_sigma_epsilon={sigma_epsilon}_r={args.r}.npy")
        np.save(output_file, results)
        print(f"Ended  m= {k}")
        if sum_sample == True:
            plt.plot(ns, erros, label=f"m={k}")
        else:
            plt.plot(param, erros, label=f"m={k}")

    output_file = os.path.join(output_directory, f"ridge_{relation_description}_tgt_param_{gamma_tgt}_ratio_{p}_rotation_{rotation}_results_num_models_{0}_sigama_eta={sigma_eta}_sigma_xi={sigma_xi}_sigma_epsilon={sigma_epsilon}_r={args.r}.npy")
    np.save(output_file, ridge_err)

    output_file = os.path.join(output_directory, f"l2_{relation_description}_tgt_param_{gamma_tgt}_ratio_{p}_rotation_{rotation}_results_num_models_{0}_sigama_eta={sigma_eta}_sigma_xi={sigma_xi}_sigma_epsilon={sigma_epsilon}_r={args.r}.npy")
    np.save(output_file, l2_over_solutin)


    plt.ylim(0, 1.6)


    plt.xlabel("Source tasks parameterization (x-axis)", fontsize=12)
    plt.ylabel("Expected Test Error (Target Task)", fontsize=12)
    plt.title("Comparison of Test Error Across Models", fontsize=14)
    plt.legend(fontsize=10, loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    plt.grid(alpha=0.5)

    current_time = datetime.now()

    timestamp = current_time.strftime("%m-%d_%H-%M")


    output_path = f"{output_root}/graphs/{timestamp}_only_analitic_plot_debias={debias}_{relation_description}_assumingH={assumption}_expparam={exp_param}_tgt_param_{gamma_tgt}_sigama_eta={sigma_eta}_sigma_xi={sigma_xi}_sigma_epsilon={sigma_epsilon}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')  # High-quality image

    plt.tight_layout()


if hm == True:
    h_tildes = [np.eye(d, d) for _ in range(m_hm)]
    R = calculate_R(h_tildes)
    sigma_x = np.eye(d, d)
    W = calculate_W(R, sigma_x)
    sigma_etas = np.ones(m_hm) * sigma_eta
    sigma_xis = np.ones(m_hm) * sigma_xi
    c_dict = {}
    c_tag_dict = {}
    alpha_tl_range = np.logspace(-4, 2, 25)
    for alpha_TL in alpha_tl_range:
        c = solve_c(alpha_TL, gamma_tgt, W, d)
        c_tag = calculate_c_tag(c, alpha_TL, W, gamma_tgt, d)
        if c < 0 or c_tag < 0:
            print(
                f"for alpha = {alpha_TL} and number of models = {m_hm}, c or c tag is negative. c = {c}, c_tag = {c_tag}")
        else:
            c_dict[alpha_TL] = c
            c_tag_dict[alpha_TL] = c_tag
    x_values = np.linspace(0.1, 5.1, 128)  # Avoiding division by zero
    n_sample = d / x_values

    r_range = np.arange(1, d+1, 1)  # Range of r values
    heatmap_data = np.zeros((len(r_range), len(x_values)))
    j = 0
    for r in r_range:
        hs = []
        if relation_description == 'subspace':
            hs = [generate_random_subspace(d, r) for _ in range(m_hm)]
        if relation_description == 'eng_subspace':
            hs = [generate_random_energy_subspace(d, r) for _ in range(m_hm)]

        print(r, flush=True)
        erros = []
        param = []
        ns = []
        for n_tilde in n_sample:
            TEMP_ERRORS = []
            TEMP_ALPHAS = []
            param_lvl = d / n_tilde
            gamma_src = param_lvl

            for alpha_TL in alpha_tl_range:
                if c_dict.get(alpha_TL):
                    c = c_dict.get(alpha_TL)
                    c_tag = c_tag_dict.get(alpha_TL)
                    error = calculate_error(param_lvl, alpha_TL, c, c_tag, W, R, b, gamma_tgt, gamma_src, d, sigma_etas, sigma_xis,
                                                sigma_epsilon, h_tildes, hs)
                    TEMP_ERRORS.append(error)
                    TEMP_ALPHAS.append(alpha_TL)
            i = np.argmin(TEMP_ERRORS)
            best_error_for_param_lvl = min(TEMP_ERRORS)
            best_alpha = TEMP_ALPHAS[i]
            print(f"for gamma_src = {d/n_tilde}, best alpha = {best_alpha}", flush=True)
            erros.append(best_error_for_param_lvl)
            param.append(param_lvl)
            ns.append(n_tilde)
        results = np.array([param, erros])  # First row: param_lvl, Second row: errors
        heatmap_data[j, :] = erros  # Store errors for this r value
        j += 1
    ridge_err = ridge_regression_error(n,d,sigma_epsilon,b)
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
    plt.savefig(f"{output_root}/graphs/{free_text}_{n}_{relation_description}_{sigma_eta}_{sigma_xi}_heatmap.png", dpi=300, bbox_inches='tight')

    folder_name = f"{output_root}/heatmap/n{n}_desc{relation_description}_eta{sigma_eta}_xi{sigma_xi}_eps{sigma_epsilon}_b{b}_tgt{gamma_tgt}"
    Path(folder_name).mkdir(parents=True, exist_ok=True)

    np.save(f"{folder_name}/heatmap_mhm{m_hm}.npy", heatmap_data)
