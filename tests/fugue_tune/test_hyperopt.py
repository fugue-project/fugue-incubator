import pandas as pd

from fugue_tune.convert import tunable
from fugue_tune.hyperopt import HyperoptRunner
from fugue_tune.space import Choice, Rand, Space, Grid

from typing import Dict, Any
from fugue import FugueWorkflow
from fugue_tune.tuner import Tuner


def test_run():
    @tunable()
    def func(df: pd.DataFrame, a: float, b: float, c: int) -> Dict[str, Any]:
        return {"error": a * a + b * b + df.shape[0] + c, "metadata": {"d": 1}}

    pdf = pd.DataFrame([[0]], columns=["a"])
    runner = HyperoptRunner(100, seed=3)

    res = runner.run(
        func, dict(df=pdf, b=Rand(-100, 100), a=10, c=Choice(1, -1)), {"a", "b", "c"}
    )
    assert res["error"] < 103.0
    assert res["hp"]["a"] == 10
    assert abs(res["hp"]["b"]) < 3.0
    assert res["hp"]["c"] == -1
    assert len(res) == 3
    assert res["metadata"] == {"d": 1}


def test_wf():
    @tunable()
    def func(a: float, b: float, c: int) -> float:
        return a * a + b * b + c

    t = Tuner()
    with FugueWorkflow() as dag:
        space = t.space_to_df(
            dag, Space(a=Grid(1, 2), b=Rand(-100, 100), c=Choice(1, -1))
        )
        t.tune(space, func, objective_runner=HyperoptRunner(100, seed=3)).show()
