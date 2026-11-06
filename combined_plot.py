"""Plot recovered empirical/theoretical curves and covariance-factor experiments."""
import os
os.environ.setdefault("MPLBACKEND", "Agg")
from pathlib import Path
from paper_utils import add_output_argument, prepare_output, str2bool
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
from datetime import datetime
matplotlib.use('Agg')
import argparse
import os
from sklearn.linear_model import Ridge
from numpy.linalg import inv, pinv
import warnings
from scipy.linalg import toeplitz


parser = argparse.ArgumentParser()

parser.add_argument('--std', type=str2bool, default=False)
parser.add_argument('--noneid', type=str2bool, default=False, help='If True, do not plot the identity line')
parser.add_argument('--n', type=int, default=32, help='Target training sample count; must match experiment')
parser.add_argument('--d', type=int, default=128)
parser.add_argument('--baseline-trials', type=int, default=100,
                    help='Monte Carlo trials for the recovered oracle ridge/min-norm baselines')
parser.add_argument('--cov_x_des', type=str, default="identity", help='Covariance matrix of the features (identity or expdecay)')
parser.add_argument('--epsilon', type=float, default=0.31623, help='Noise std in training and test labels')
parser.add_argument('--empirical', type=str, required=True)
parser.add_argument('--numeric', type=str, default=None)
parser.add_argument('--emp_only', type=str2bool, default=False, help='If True, plot only empirical results')
parser.add_argument('--factor', type=str2bool, default= False)
add_output_argument(parser)


args = parser.parse_args()
if not args.emp_only and args.numeric is None:
    parser.error('--numeric is required unless --emp_only True')
if args.baseline_trials < 1:
    parser.error('--baseline-trials must be positive')
output_root = prepare_output(args)
std = args.std
noneid = args.noneid
n = args.n
cov_x_des = args.cov_x_des
epsilon = args.epsilon
factor = args.factor


empirical_directory = args.empirical
numeric_directory = args.numeric
alphas = [1,1,1,1,1,1]

last_part = Path(empirical_directory).name


def has_target_ratio(s):
    ratios = [128/64, 128/32, 128/96, 128/112]

    patterns = {f"_{r:.1f}_" for r in ratios}

    for p in patterns:
        if p in s:
            return p.strip("_")  # Return the ratio without underscores
    return "4"  # Default value if no pattern is found

tgt = args.d / n
empirical_files = [f for f in os.listdir(empirical_directory) if f.endswith(".npy")]
numeric_files = ([f for f in os.listdir(numeric_directory) if f.endswith(".npy")]
                 if numeric_directory else [])
if not empirical_files:
    parser.error('no empirical .npy files found')

plt.figure(figsize=(8, 6))
plt.axhline(y=1 + epsilon**2, color='gray', linestyle='-.', label='Null estimate')  # Horizontal reference line

all_models = sorted({int(f.split("num_models=")[1].split("_")[0].strip()) for f in empirical_files},reverse=True)
colors = plt.cm.magma(np.linspace(0, 0.8, len(all_models)))
color_map = {model: color for model, color in zip(all_models, colors)}


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

cov_x = np.eye(args.d)
if noneid or cov_x_des == 'exp_decay':
    cov_x = exponential_covariance_matrix(args.d, sigma_squared=1.0, rho=0.5)

def simulate_expected_errors(n_train, d, cov_x, b, sigma_eps=1.0, n_test=5000,
                              n_trials=100, alphas=np.logspace(-6, 2, 50), seed=None):
    """
    Simulates optimal ridge regression and minimum L2-norm interpolator errors.

    Parameters:
        n_train (int): Number of training samples
        d (int): Data dimension
        cov_x (ndarray): Covariance matrix of the features (d x d)
        b (float): Ground truth beta ~ N(0, b/d * I_d)
        sigma_eps (float): Noise std in training and test labels
        n_test (int): Number of test samples per trial
        n_trials (int): Number of repetitions
        alphas (ndarray): Grid of ridge regularization parameters
        seed (int): Random seed for reproducibility

    Returns:
        dict: Averaged test errors for min-L2 and optimal ridge regression
    """
    if seed is not None:
        np.random.seed(seed)

    L = np.linalg.cholesky(cov_x)  # for sampling from N(0, Σ)
    ridge_errors = []
    min_l2_errors = []

    for _ in range(n_trials):
        beta = np.random.randn(d) * np.sqrt(b / d)

        X_train = np.dot(np.random.randn(n_train, d), L.T)
        y_train = X_train @ beta + np.random.randn(n_train) * sigma_eps

        X_test = np.dot(np.random.randn(n_test, d), L.T)
        y_test = X_test @ beta + np.random.randn(n_test) * sigma_eps

        best_err = np.inf
        for alpha in alphas:
            try:
                beta_ridge = inv(X_train.T @ X_train + alpha * np.eye(d)) @ X_train.T @ y_train
                err = np.mean((X_test @ beta_ridge - y_test) ** 2)
                if err < best_err:
                    best_err = err
            except np.linalg.LinAlgError:
                continue
        ridge_errors.append(best_err)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if n_train >= d:
                beta_l2 = pinv(X_train) @ y_train
            else:
                beta_l2 = X_train.T @ pinv(X_train @ X_train.T) @ y_train
        err = np.mean((X_test @ beta_l2 - y_test) ** 2)
        min_l2_errors.append(err)

    return {
        "ridge_error": np.mean(ridge_errors),
        "min_l2_error": np.mean(min_l2_errors),
    }


if "exp_decay" in empirical_directory:
    print("exp_decay in file name", flush=True)
    cov_x = exponential_covariance_matrix(args.d, sigma_squared=1.0, rho=0.5)
    noneid = True

ridge_error = None
min_l2_error = None

res = simulate_expected_errors(n, args.d, cov_x, 1.0, sigma_eps=epsilon,
                               n_trials=args.baseline_trials,
                               alphas=np.logspace(-6, 2, 50), seed=42)
ridge_error = res["ridge_error"]
min_l2_error = res["min_l2_error"]
print(f"Optimal ridge error: {ridge_error}", flush=True)
print(f"Minimum L2-norm error: {min_l2_error}", flush=True)


i = 0
if not args.emp_only:
    for file_name in numeric_files:
        file_path = os.path.join(numeric_directory, file_name)
        data = np.load(file_path)

        if file_name.startswith("ridge"):
            if ridge_error is None:
                print(file_name, flush=True)
                plt.axhline(y=data, color='blue', linestyle='--', label='Oracle-tuned ridge')
            else:
                print(file_name, flush=True)
                plt.axhline(y=ridge_error, color='blue', linestyle='--', label='Oracle-tuned ridge')

        elif file_name.startswith("l2"):
            if min_l2_error is None:
                print(file_name, flush=True)
                plt.axhline(y=data, color='green', linestyle='--', label='ML2N')
            else:
                print(file_name, flush=True)
                plt.axhline(y=min_l2_error, color='green', linestyle='--', label='ML2N')
        else:
            print(file_name, flush=True)
            x_values = data[0]  # First row: x-axis
            y_values = data[1]  # Second row: y-axis

            num_of_models = file_name.split("num_models_")[1].split("_")[0].strip()

            num_of_models_int = int(num_of_models)
            plt.plot(x_values, y_values, linestyle='-', color=color_map[num_of_models_int], linewidth=2.5, alpha=alphas[i]*0.8, label=f"{num_of_models} models")
            i += 1

        plt.axhline(y=epsilon**2, color='saddlebrown', linestyle='dotted', label='Bayes optimal error' )

    i = 0
    for file_name in empirical_files:
        file_path = os.path.join(empirical_directory, file_name)
        data = np.load(file_path)

        x_values = data[0]  # First row: x-axis
        y_values = data[1]  # Second row: y-axis


        num_of_models = file_name.split("num_models=")[1].split("_")[0].strip()

        if std:
            std_values = data[2]  # Third row: standard deviation
            print(std_values, flush=True)
            plt.fill_between(x_values, y_values - std_values* factor, y_values + std_values*factor, color=color_map[int(num_of_models)], alpha=0.3)
            plt.plot(x_values, y_values, linestyle='', marker='o', markersize=5, alpha=alphas[i]*0.7, color=color_map[int(num_of_models)])
        else:
            plt.plot(x_values, y_values, linestyle='', marker='o', markersize=5, alpha=alphas[i]*0.7, color=color_map[int(num_of_models)])

        i += 1

    plt.ylim(0,1.4 )
    plt.xlim(0,5.1)


    plt.xlabel(r"Source parameterization ($d/\tilde{n}$)", fontsize=24)
    plt.ylabel("Expected test error (target task)", fontsize=24)
    plt.xticks(fontsize=16)  # Make x-tick labels larger
    plt.yticks(fontsize=16)
    plt.grid(alpha=0.5)

    current_time = datetime.now()

    timestamp = current_time.strftime("%m-%d_%H-%M")


    output_path = f"{output_root}/graphs/{timestamp}_{last_part}.png"

    plt.tight_layout()

    plt.savefig(output_path, dpi=300,transparent=True, bbox_inches='tight')  # High-quality image
    print(f"Plot saved as {output_path}")

else:
    def extract_bad_models(file_name):
        return int(file_name.split("bad_models=")[1].split("_")[0].strip())

    def save_bad_models_legend(bad_color_map, output_path):
        fig_leg = plt.figure(figsize=(14, 1.2))
        ax_leg = fig_leg.add_subplot(111)

        handles = []
        labels = []

        for k in sorted(bad_color_map.keys()):
            h, = ax_leg.plot(
                [],
                [],
                linestyle='-',
                marker= None,
                linewidth=2.5,
                color=bad_color_map[k],
                label=fr"$k={k}$"
            )
            handles.append(h)
            labels.append(fr"$k={k}$")

        ax_leg.legend(
            handles,
            labels,
            loc='center',
            ncol=len(labels),
            frameon=False,
            fontsize=16,
            handlelength=2.5,
            columnspacing=1.5
        )

        ax_leg.axis("off")
        fig_leg.savefig(output_path, dpi=300, transparent=True, bbox_inches='tight')
        plt.close(fig_leg)
        print(f"Legend saved as {output_path}", flush=True)

    if "bad_models_" in empirical_directory or any("bad_models=" in f for f in empirical_files):
        has_alpha = all(np.load(os.path.join(empirical_directory, f), mmap_mode='r').shape[0] >= 3
                        for f in empirical_files)
        if factor and not has_alpha:
            parser.error('these heterogeneous-source arrays contain only x and error, not factors')
        bad_models_values = sorted({
            extract_bad_models(f) for f in empirical_files if "bad_models=" in f
        })

        bad_colors = plt.cm.magma(np.linspace(0, 0.8, len(bad_models_values)))
        bad_color_map = {k: c for k, c in zip(bad_models_values, bad_colors)}

        file_name_part = empirical_directory.rstrip("/").split("/")[-1]

        plt.figure(1, figsize=(8, 6))

        if factor:
            plt.figure(2, figsize=(8, 6))  # factor
            plt.figure(3, figsize=(8, 6))  # alpha
        else:
            plt.figure(2, figsize=(8, 6))  # alpha

        max_alpha_seen = 0
        x_ref = None

        for file_name in sorted(empirical_files, key=extract_bad_models):
            if "bad_models=" not in file_name:
                continue

            file_path = os.path.join(empirical_directory, file_name)
            data = np.load(file_path)

            x_values = data[0]
            y_values = data[1]
            k_bad = extract_bad_models(file_name)
            color = bad_color_map[k_bad]

            x_ref = x_values

            if factor:
                fac_values = data[2]
                alpha_values = data[-1]
            else:
                alpha_values = data[-1]

            max_alpha_seen = max(max_alpha_seen, np.max(alpha_values))

            plt.figure(1)
            plt.plot(
                x_values, y_values,
                linestyle='-',
                marker=None,
                linewidth=2.5,
                color=color,
                alpha=0.85
            )

            if factor:
                plt.figure(2)
                plt.plot(
                    x_values, fac_values,
                    linestyle='-',
                    linewidth=2.5,
                    color=color,
                    alpha=0.85
                )

                plt.figure(3)
                plt.plot(
                    x_values, alpha_values,
                    linestyle='-',
                    linewidth=2.5,
                    color=color,
                    alpha=0.85
                )
            elif has_alpha:
                plt.figure(2)
                plt.plot(
                    x_values, alpha_values,
                    linestyle='',
                    marker='o',
                    markersize=5,
                    color=color,
                    alpha=0.85
                )

        plt.figure(1)
        plt.axhline(y=epsilon**2, color='saddlebrown', linestyle='dotted')
        plt.axhline(y=1 + epsilon**2, color='gray', linestyle='-.')
        plt.axhline(y=ridge_error, color='blue', linestyle='--')
        plt.axhline(y=min_l2_error, color='green', linestyle='--')

        plt.ylim(0, 1.4)
        plt.xlim(0, 5.1)
        plt.xlabel(r"Source parameterization ($d/\tilde{n}$)", fontsize=24)
        plt.ylabel("Expected test error (target)", fontsize=24)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.grid(alpha=0.5)

        output_path = f"{output_root}/graphs/{file_name_part}_bad_models_empirical_only.png"
        plt.figure(1).savefig(output_path, dpi=300, transparent=True, bbox_inches='tight')
        print(f"Plot saved as {output_path}", flush=True)

        legend_output_path = f"{output_root}/graphs/{file_name_part}_bad_models_legend.png"
        save_bad_models_legend(bad_color_map, legend_output_path)

        if factor:
            plt.figure(2)
            y_ref = np.where(x_ref <= 1.0, 1.0, 1.0 / x_ref)
            plt.plot(x_ref, y_ref, linestyle='dotted', color="#109999", linewidth=2.5)

            plt.ylim(0, 1.8)
            plt.xlim(0, 5.1)
            plt.xlabel(r"Source parameterization ($d/\tilde{n}$)", fontsize=24)
            plt.ylabel(r"Empirical factor $\tilde{\rho}$", fontsize=24)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.grid(alpha=0.5)

            output_path = f"{output_root}/graphs/{file_name_part}_bad_models_factor_plot.png"
            plt.figure(2).savefig(output_path, dpi=300, transparent=True, bbox_inches='tight')
            print(f"Plot saved as {output_path}", flush=True)

            plt.figure(3)
            plt.yscale('log')
            plt.ylim(1e-6, max_alpha_seen * 1.1)
            plt.xlim(0, 5.1)
            plt.xlabel(r"Source parameterization ($d/\tilde{n}$)", fontsize=24)
            plt.ylabel(r"Best alpha", fontsize=24)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.grid(alpha=0.5)

            output_path = f"{output_root}/graphs/{file_name_part}_bad_models_alpha_plot.png"
            plt.figure(3).savefig(output_path, dpi=300, transparent=True, bbox_inches='tight')
            print(f"Plot saved as {output_path}", flush=True)

        elif has_alpha:
            plt.figure(2)
            plt.yscale('log')
            plt.ylim(1e-6, max_alpha_seen * 1.1)
            plt.xlim(0, 5.1)
            plt.xlabel(r"Source parameterization ($d/\tilde{n}$)", fontsize=24)
            plt.ylabel(r"Best alpha", fontsize=24)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.grid(alpha=0.5)

            output_path = f"{output_root}/graphs/{file_name_part}_bad_models_alpha_plot.png"
            plt.figure(2).savefig(output_path, dpi=300, transparent=True, bbox_inches='tight')
            print(f"Plot saved as {output_path}", flush=True)

        exit()
    if factor== False:
        i = 0
        for file_name in empirical_files:
            file_path = os.path.join(empirical_directory, file_name)
            data = np.load(file_path)

            x_values = data[0]  # First row: x-axis
            y_values = data[1]  # Second row: y-axis
            alpha_values = data[-1]  # Fourth row: alpha values

            num_of_models = file_name.split("num_models=")[1].split("_")[0].strip()

            plt.figure(1, figsize=(8, 6))
            plt.plot(x_values, y_values, linestyle='', marker='o', markersize=5, alpha=alphas[i]*0.7, color=color_map[int(num_of_models)])
            plt.figure(2, figsize=(8, 6))
            plt.plot(x_values, alpha_values, linestyle='', marker='o', markersize=5, alpha=alphas[i]*0.7, color=color_map[int(num_of_models)])
            i += 1

        plt.figure(1)
        plt.axhline(y=epsilon**2, color='saddlebrown', linestyle='dotted', label='Bayes optimal error' )
        plt.axhline(y=1 + epsilon**2, color='gray', linestyle='-.', label='Null estimate')  # Horizontal reference line
        plt.axhline(y=ridge_error, color='blue', linestyle='--', label='Oracle-tuned ridge')
        plt.axhline(y=min_l2_error, color='green', linestyle='--', label='ML2N')
        plt.ylim(0,1.4 )
        plt.xlim(0,5.1)
        plt.xlabel(r"Source parameterization ($d/\tilde{n}$)", fontsize=24)
        plt.ylabel("Expected test error (target task)", fontsize=24)
        plt.xticks(fontsize=16)  # Make x-tick labels larger
        plt.yticks(fontsize=16)
        plt.grid(alpha=0.5)
        current_time = datetime.now()
        timestamp = current_time.strftime("%m-%d_%H-%M-%S")
        file_name_part = Path(empirical_directory).name
        output_path = f"{output_root}/graphs/{file_name_part}_empirical_only.png"
        plt.figure(1).savefig(output_path, dpi=300,transparent=True, bbox_inches='tight')  # High-quality image
        print(f"Plot saved as {output_path}")
        plt.figure(2)
        plt.yscale('log')
        plt.ylim(0,max(alpha_values)*1.1)
        plt.xlim(0,5.1)
        plt.xlabel(r"Source parameterization ($d/\tilde{n}$)", fontsize=24)
        plt.ylabel(r"Best alpha", fontsize=24)
        plt.xticks(fontsize=16)  # Make x-tick labels larger
        plt.yticks(fontsize=16)
        plt.grid(alpha=0.5)
        current_time = datetime.now()
        timestamp = current_time.strftime("%m-%d_%H-%M-%S")
        output_path = f"{output_root}/graphs/{file_name_part}_alpha_plot.png"
        plt.figure(2).savefig(output_path, dpi=300,transparent=True, bbox_inches='tight')  # High-quality image
        print(f"Plot saved as {output_path}")


    else:
        i = 0
        for file_name in empirical_files:
            file_path = os.path.join(empirical_directory, file_name)
            data = np.load(file_path)

            x_values = data[0]  # First row: x-axis
            y_values = data[1]  # Second row: y-axis
            fac_values = data[2]  # Third row: standard deviation
            alpha_values = data[-1]  # Fourth row: alpha values

            num_of_models = file_name.split("num_models=")[1].split("_")[0].strip()

            plt.figure(1, figsize=(8, 6))
            plt.plot(
                x_values, y_values,
                linestyle='-',
                alpha=alphas[i]*0.7,
                color=color_map[int(num_of_models)]
            )

            plt.figure(2, figsize=(8, 6))
            plt.plot(
                x_values, fac_values,
                linestyle='-',
                alpha=alphas[i]*0.7,
                color=color_map[int(num_of_models)]
            )
            plt.figure(3, figsize=(8, 6))
            plt.plot(
                x_values, alpha_values,
                linestyle='-',
                alpha=alphas[i]*0.7,
                color=color_map[int(num_of_models)]
            )
            i += 1

        plt.figure(1)
        plt.axhline(y=epsilon**2, color='saddlebrown', linestyle='dotted', label='Bayes optimal error' )
        plt.axhline(y=1 + epsilon**2, color='gray', linestyle='-.', label='Null estimate')  # Horizontal reference line
        plt.axhline(y=ridge_error, color='blue', linestyle='--', label='Oracle-tuned ridge')
        plt.axhline(y=min_l2_error, color='green', linestyle='--', label='ML2N')

        plt.figure(2)


        y_ref = np.where(x_values <= 1.0, 1.0, 1.0 / x_values)

        plt.plot(x_values, y_ref, linestyle='dotted',
                color = "#109999", label=r'$\rho$')

        plt.ylim(0,1.8)
        plt.xlim(0,5.1)
        plt.legend(fontsize=10, loc='upper right',bbox_to_anchor=(1, 1))
        plt.xlabel(r"Source parameterization ($d/\tilde{n}$)", fontsize=24)
        plt.ylabel(r"Empirical factor $\tilde{\rho}$", fontsize=24)
        plt.xticks(fontsize=16)  # Make x-tick labels larger
        plt.yticks(fontsize=16)
        plt.grid(alpha=0.5)
        current_time = datetime.now()
        file_name_part = Path(empirical_directory).name
        output_path = f"{output_root}/graphs/{file_name_part}_factor_plot.png"
        plt.figure(2).savefig(output_path, dpi=300,transparent=True, bbox_inches='tight')  # High-quality image

        print(f"Plot saved as {output_path}")
        plt.figure(1)
        plt.ylim(0,1.4 )
        plt.xlim(0,5.1)
        plt.xlabel(r"Source parameterization ($d/\tilde{n}$)", fontsize=24)
        plt.ylabel("Expected test error (target)", fontsize=24)
        plt.xticks(fontsize=16)  # Make x-tick labels larger
        plt.yticks(fontsize=16)
        plt.grid(alpha=0.5)
        current_time = datetime.now()

        output_path = f"{output_root}/graphs/{file_name_part}_empirical_only.png"
        plt.figure(1).savefig(output_path, dpi=300,transparent=True, bbox_inches='tight')  # High-quality image
        print(f"Plot saved as {output_path}")

        plt.figure(3)
        plt.yscale('log')
        plt.ylim(0,max(alpha_values)*1.1)
        plt.xlim(0,5.1)
        plt.xlabel(r"Source parameterization ($d/\tilde{n}$)", fontsize=24)
        plt.ylabel(r"Best alpha", fontsize=24)
        plt.xticks(fontsize=16)  # Make x-tick labels larger
        plt.yticks(fontsize=16)
        plt.grid(alpha=0.5)
        current_time = datetime.now()
        output_path = f"{output_root}/graphs/{file_name_part}_alpha_plot.png"
        plt.figure(3).savefig(output_path, dpi=300,transparent=True, bbox_inches='tight')  # High-quality image
        print(f"Plot saved as {output_path}")
