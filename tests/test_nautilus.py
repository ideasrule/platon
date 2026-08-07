import sys
import types
import unittest
from unittest import mock

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from platon.combined_retriever import CombinedRetriever
from platon.plotter import Plotter
from platon.retrieval_result import RetrievalResult


class _Param:
    def __init__(self, best_guess):
        self.best_guess = best_guess


class _FitInfo:
    fit_param_names = ["T", "x"]
    all_params = {"profile_type": _Param("isothermal")}

    def _get_num_fit_params(self):
        return 2

    def _from_unit_interval_array(self, cube):
        return np.array([800 + 800 * cube[0], cube[1]])

    def _ln_prior(self, params):
        return 0

    def _interpret_param_array(self, params):
        return {"T": params[0]}


class _Sampler:
    init_kwargs = None
    run_kwargs = None

    def __init__(self, prior, likelihood, **kwargs):
        type(self).init_kwargs = kwargs
        cubes = np.array([[0.1, 0.2], [0.5, 0.5], [0.9, 0.8]])
        self.samples = np.array([prior(cube) for cube in cubes])
        self.logl = np.array([likelihood(point) for point in self.samples])
        self.log_z = -12.5
        self.n_eff = 10000

    def run(self, **kwargs):
        type(self).run_kwargs = kwargs
        return True

    def posterior(self):
        return self.samples, np.log([0.2, 0.5, 0.3]), self.logl


class TestNautilus(unittest.TestCase):
    def test_lazy_import_error(self):
        with mock.patch.dict(sys.modules, {"nautilus": None}):
            with self.assertRaisesRegex(
                    ImportError, "pip install nautilus-sampler"):
                CombinedRetriever().run_nautilus(
                    None, None, None, None, None, None, _FitInfo())

    def test_result_defaults_and_plots(self):
        retriever = CombinedRetriever()

        def fake_ln_like(params, *args, ret_best_fit=False,
                         lnlike_per_point=False, **kwargs):
            if ret_best_fit:
                return None, None, None, None
            values = np.array([
                -0.5 * ((params[0] - 1200) / 100)**2,
                -0.5 * params[1]**2])
            if lnlike_per_point:
                retriever.params_to_lnlike[tuple(params)] = values
                return values
            return values.sum()

        module = types.SimpleNamespace(Sampler=_Sampler)
        with mock.patch.dict(sys.modules, {"nautilus": module}), \
             mock.patch.object(retriever, "_ln_like",
                               side_effect=fake_ln_like), \
             mock.patch("platon.combined_retriever."
                        "write_param_estimates_file"), \
             mock.patch("platon.combined_retriever.psisloo",
                        return_value=(0, np.zeros(2), np.zeros(2))):
            result = retriever.run_nautilus(
                None, None, None, None, None, None, _FitInfo(),
                num_final_samples=3, verbose=False)

        self.assertIsInstance(result, RetrievalResult)
        self.assertEqual(result.retrieval_type, "nautilus")
        self.assertEqual(_Sampler.init_kwargs["n_live"], 2000)
        self.assertEqual(_Sampler.init_kwargs["n_networks"], 16)
        self.assertEqual(_Sampler.run_kwargs["n_eff"], 10000)
        self.assertTrue(_Sampler.run_kwargs["discard_exploration"])
        self.assertAlmostEqual(np.sum(result.weights), 1)
        self.assertEqual(result.final_logz, -12.5)
        self.assertTrue(result.success)

        plotter = Plotter()
        plotter.plot_retrieval_corner(result)
        plotter.plot_retrieval_TP_profiles(result, num_samples=2)
        plt.close("all")


if __name__ == "__main__":
    unittest.main()
