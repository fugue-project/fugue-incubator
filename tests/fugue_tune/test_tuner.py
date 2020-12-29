import json
from typing import Any, Iterable, List

import pandas as pd
from fugue import ExecutionEngine, FugueWorkflow, IterableDataFrame, WorkflowDataFrame
from pytest import raises

from fugue_tune.exceptions import FugueTuneCompileError
from fugue_tune.space import Grid, Space
from fugue_tune.tuner import Tuner


def test_space_to_df():
    tuner = Tuner()

    with FugueWorkflow() as dag:
        df = tuner.space_to_df(dag, Space(a=Grid(0, 1), b=Grid(2, 3)))
        df.assert_eq(
            dag.df(
                [
                    ['[{"a": 0, "b": 2}]'],
                    ['[{"a": 0, "b": 3}]'],
                    ['[{"a": 1, "b": 2}]'],
                    ['[{"a": 1, "b": 3}]'],
                ],
                "__fmin_params__:str",
            )
        )

    with FugueWorkflow() as dag:
        df = tuner.space_to_df(
            dag, Space(a=Grid(0, 1), b=Grid(2, 3)), batch_size=3, shuffle=False
        )
        df.assert_eq(
            dag.df(
                [
                    ['[{"a": 0, "b": 2}, {"a": 0, "b": 3}, {"a": 1, "b": 2}]'],
                    ['[{"a": 1, "b": 3}]'],
                ],
                "__fmin_params__:str",
            )
        )


def test_tune_simple():
    tuner = Tuner()

    def t1(a: int, b: int) -> float:
        return a + b

    for distributable in [True, False, None]:
        with FugueWorkflow() as dag:
            df = tuner.space_to_df(dag, Space(a=Grid(0, 1), b=Grid(2, 3)))
            tuner.tune(df, t1, distributable=distributable).show()

    def t2(e: ExecutionEngine, a: int, b: int) -> float:
        assert isinstance(e, ExecutionEngine)
        return a + b

    for distributable in [False, None]:
        with FugueWorkflow() as dag:
            df = tuner.space_to_df(dag, Space(a=Grid(0, 1), b=Grid(2, 3)))
            tuner.tune(df, t2, distributable=distributable).show()

    with raises(FugueTuneCompileError):
        with FugueWorkflow() as dag:
            df = tuner.space_to_df(dag, Space(a=Grid(0, 1), b=Grid(2, 3)))
            tuner.tune(df, t2, distributable=True).show()


def test_tune_df(tmpdir):
    tuner = Tuner()

    def t1(a: int, p: pd.DataFrame, b: int) -> float:
        return float(a + b + p["y"].sum())

    for distributable in [True, False, None]:
        with FugueWorkflow() as dag:
            s = tuner.space_to_df(dag, Space(a=Grid(0, 1), b=Grid(2, 3)), batch_size=3)
            t = dag.df([[0, 1], [1, 2], [0, 2]], "x:int,y:int")
            df = tuner.serialize_df(t, "p", str(tmpdir)).cross_join(s.broadcast())
            tuner.tune(df, t1, distributable=distributable).show()
            
