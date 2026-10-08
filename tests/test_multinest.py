import types
import unittest
from unittest import mock

import numpy as np

from platon.combined_retriever import CombinedRetriever
from platon.fit_info import FitInfo


class TestMultiNest(unittest.TestCase):
    def setUp(self):
        self.retriever = CombinedRetriever()
        self.fit_info = FitInfo({"T": 1000})
        self.fit_info.add_uniform_fit_param("T", 500, 2000)
        self.args = (None, None, None, None, None, None, self.fit_info)

    def test_resumed_posterior_and_iteration_limit(self):
        samples = np.array([[900.], [1000.], [1100.]])
        analyzer = mock.Mock()
        analyzer.get_data.return_value = np.column_stack(
            [np.full(3, 1 / 3), [2, 0, 2], samples])
        analyzer.get_equal_weighted_posterior.return_value = np.column_stack(
            [samples, [-1, 0, -1]])
        # A completed resumed run need not call the likelihood at all.
        solve = mock.Mock(return_value={"logZ": -12.5})
        module = types.SimpleNamespace(
            solve=solve, Analyzer=mock.Mock(return_value=analyzer))
        evaluated = []

        def likelihood(params, *args, ret_best_fit=False,
                       lnlike_per_point=False, **kwargs):
            if ret_best_fit:
                return None, None, None, None
            values = np.array([-0.5 * ((params[0] - 1000) / 100)**2, -1.])
            if lnlike_per_point:
                evaluated.append(params[0])
                self.retriever.params_to_lnlike[tuple(params)] = values
                return values
            return values.sum()

        options = {"resume": True, "outputfiles_basename": "saved_run_"}
        with mock.patch.dict("sys.modules", {"pymultinest": module}), \
             mock.patch.object(self.retriever, "_ln_like", side_effect=likelihood), \
             mock.patch("platon.combined_retriever.write_param_estimates_file"):
            result = self.retriever.run_multinest(
                *self.args, maxiter=17, num_final_samples=3,
                multinest_kwargs=options)

        self.assertEqual(solve.call_args.kwargs["max_iter"], 17)
        self.assertTrue(solve.call_args.kwargs["resume"])
        self.assertNotIn("max_iter", options)
        self.assertCountEqual(evaluated, [900., 1000., 1100.])
        self.assertEqual(np.shape(result.pointwise_lnlikes), (3, 2))
        self.assertTrue(np.isfinite(result.loo_total))
        self.assertEqual(result.final_logz, -12.5)

    def test_rejects_unsupported_call_limit_before_importing_sampler(self):
        with mock.patch.dict("sys.modules", {"pymultinest": None}):
            with self.assertRaisesRegex(ValueError, "does not support maxcall"):
                self.retriever.run_multinest(*self.args, maxcall=10)

    def test_rejects_conflicting_or_invalid_iteration_limit(self):
        with self.assertRaisesRegex(ValueError, "conflicts"):
            self.retriever.run_multinest(
                *self.args, maxiter=10, multinest_kwargs={"max_iter": 20})
        for limit in [-1, 1.5]:
            with self.subTest(limit=limit), self.assertRaisesRegex(
                    ValueError, "non-negative integer"):
                self.retriever.run_multinest(*self.args, maxiter=limit)
