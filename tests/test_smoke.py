"""Small CPU integration checks, not a reproduction of the manuscript scores."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from de_be import difference_curves, load_curves
from paper_utils import str2bool


class PlotValidationTests(unittest.TestCase):
    def test_boolean_false_is_false(self):
        self.assertFalse(str2bool('False'))
        self.assertTrue(str2bool('True'))

    def test_difference_sign_and_grid(self):
        a = {1: np.array([[0.5, 2], [0.4, 0.8]])}
        b = {1: np.array([[0.5, 2], [0.2, 0.9]])}
        np.testing.assert_allclose(difference_curves(a, b)[1][1], [0.2, -0.1])
        with self.assertRaises(ValueError):
            difference_curves(a, {2: b[1]})
        with self.assertRaises(ValueError):
            difference_curves(a, {1: np.array([[0.6, 2], [0.2, 0.9]])})

    def test_duplicate_curves_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ['a_num_models=1_a.npy', 'b_num_models=1_a.npy']:
                np.save(Path(tmp) / name, [[0.5, 2], [0.4, 0.8]])
            with self.assertRaises(ValueError):
                load_curves(tmp)


class PipelineSmokeTests(unittest.TestCase):
    def run_script(self, script, *args):
        env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', MPLBACKEND='Agg')
        result = subprocess.run([sys.executable, str(ROOT / script), *map(str, args)],
                                cwd=ROOT, env=env, text=True, capture_output=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout[-3000:] + result.stderr[-4000:])
        return result

    def assert_finite_arrays(self, root):
        paths = list(root.rglob('*.npy'))
        self.assertTrue(paths)
        for path in paths:
            self.assertTrue(np.isfinite(np.load(path, allow_pickle=False)).all(), str(path))

    def test_help_and_shell_syntax(self):
        for script in ['empirical_exp2_new.py', 'numeric.py', 'combined_plot.py',
                       'decomposition_plot.py', 'de_be.py']:
            self.run_script(script, '--help')
        for path in (ROOT / 'slurm').glob('*.sbatch'):
            subprocess.run(['bash', '-n', str(path)], check=True)

    def test_small_empirical_theory_and_all_plotters(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = ['--d', 8, '--n', 4, '--r', 4, '--number_of_models', 1,
                    '--source-ratios', 0.5, 2, '--alpha-points', 3, '--free_text', 'smoke']
            empirical = root / 'empirical'
            debiased = root / 'debiased'
            theory = root / 'theory'
            decomposition = root / 'decomposition'
            plots = root / 'plots'
            for out, debias in [(empirical, 'False'), (debiased, 'True')]:
                self.run_script('empirical_exp2_new.py', *base, '--number_of_runs', 2,
                                '--workers', 1, '--debias', debias, '--output-dir', out)
                self.assert_finite_arrays(out)
            self.run_script('numeric.py', *base, '--debias', 'False', '--output-dir', theory)
            self.assert_finite_arrays(theory)
            e = next((empirical / 'results').iterdir())
            d = next((debiased / 'results').iterdir())
            t = next((theory / 'numeric_result_icecream').iterdir())
            self.run_script('combined_plot.py', '--empirical', e, '--numeric', t,
                            '--n', 4, '--d', 8, '--baseline-trials', 1, '--output-dir', plots)
            self.run_script('de_be.py', '--first', d, '--second', e, '--output-dir', plots)
            self.run_script('empirical_exp2_new.py', *base, '--number_of_runs', 1,
                            '--decomposition', 'True', '--decomposition-main-runs', 1,
                            '--decomposition-sub-runs', 2, '--output-dir', decomposition)
            self.assert_finite_arrays(decomposition)
            self.run_script('decomposition_plot.py', '--input', decomposition / 'decomposition_2',
                            '--output-dir', plots)
            self.assertGreaterEqual(len(list((plots / 'graphs').glob('*.png'))), 3)
            # Source files are portable: neither repository nor original cluster paths are written.
            self.assertTrue(list(empirical.glob('configs/*.json')))

    def test_factor_tuning(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self.run_script('empirical_exp2_new.py', '--d', 8, '--n', 6, '--r', 4,
                            '--number_of_models', 1, '--number_of_runs', 1,
                            '--source-ratios', 2, '--cv', 'True', '--debias_factor_val', 'True',
                            '--cov_z_q', 0.5, '--cov_x_des', 'exp_decay', '--output-dir', out)
            self.assert_finite_arrays(out)
            folder = next((out / 'results').iterdir())
            self.run_script('combined_plot.py', '--empirical', folder,
                            '--emp_only', 'True', '--factor', 'True', '--n', 6, '--d', 8,
                            '--baseline-trials', 1, '--output-dir', out)
            self.assertTrue(list(out.glob('graphs/*factor_plot.png')))

    def test_heterogeneous_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self.run_script('empirical_exp2_new.py', '--d', 128, '--n', 32, '--r', 96,
                            '--number_of_models', 20, '--number_of_runs', 1,
                            '--source-ratios', 2, '--alpha-points', 3, '--debias', 'True',
                            '--bad_models', 'True', '--relation_description', 'subspace',
                            '--output-dir', out)
            self.assert_finite_arrays(out)
            folder = next((out / 'results').iterdir())
            self.assertEqual(len(list(folder.glob('*.npy'))), 6)
            self.run_script('combined_plot.py', '--empirical', folder, '--emp_only', 'True',
                            '--baseline-trials', 1, '--output-dir', out)
            self.assertTrue(list(out.glob('graphs/*bad_models_empirical_only.png')))
            self.assertFalse(list(out.glob('graphs/*alpha_plot.png')))


if __name__ == '__main__':
    unittest.main()
