from typing import Any, Dict, Tuple, Set

from fugue_tune.tunable import Tunable
from fugue_tune.tuner import ObjectiveRunner
from fugue_tune.space import StochasticExpression, Rand, Choice
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import numpy as np


class HyperoptRunner(ObjectiveRunner):
    def __init__(self, max_iter: int, seed: int = 0):
        self._max_iter = max_iter
        self._seed = seed

    def run(
        self, tunable: Tunable, kwargs: Dict[str, Any], hp_keys: Set[str]
    ) -> Dict[str, Any]:
        static_params, stochastic_params = self._split(kwargs)
        keys = list(stochastic_params.keys())

        def obj(args) -> Dict[str, Any]:
            params = {k: v for k, v in zip(keys, args)}
            tunable.run(**static_params, **params)
            hp = {k: v for k, v in tunable.hp.items() if k in hp_keys}
            return {
                "loss": tunable.error,
                "status": STATUS_OK,
                "error": tunable.error,
                "hp": hp,
                "metadata": tunable.metadata,
            }

        trials = Trials()
        fmin(
            obj,
            space=list(stochastic_params.values()),
            algo=tpe.suggest,
            max_evals=self._max_iter,
            trials=trials,
            show_progressbar=False,
            rstate=np.random.RandomState(self._seed),
        )

        return {
            "error": trials.best_trial["result"]["error"],
            "hp": trials.best_trial["result"]["hp"],
            "metadata": trials.best_trial["result"]["metadata"],
        }

    def _split(self, kwargs: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        static_params: Dict[str, Any] = {}
        stochastic_params: Dict[str, Any] = {}
        for k, v in kwargs.items():
            if isinstance(v, StochasticExpression):
                if isinstance(v, Rand):
                    stochastic_params[k] = self.convert_rand(k, v)
                elif isinstance(v, Choice):
                    stochastic_params[k] = self.convert_choice(k, v)
                else:
                    raise NotImplementedError(v)  # pragma: no cover
            else:
                static_params[k] = v
        return static_params, stochastic_params

    def convert_rand(self, k: str, v: Rand) -> Any:
        if v.q is None and not v.log and not v.normal:
            return hp.uniform(k, v.start, v.end)
        raise NotImplementedError(k, v)  # pragma: no cover

    def convert_choice(self, k: str, v: Choice) -> Any:
        return hp.choice(k, v.values)
