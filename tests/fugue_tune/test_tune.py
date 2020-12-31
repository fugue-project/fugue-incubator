import json
from typing import Any, Iterable, List

import pandas as pd
from fugue import (
    ExecutionEngine,
    FugueWorkflow,
    IterableDataFrame,
    NativeExecutionEngine,
    WorkflowDataFrame,
)
from pytest import raises

from fugue_tune import Grid, Space, serialize_df, space_to_df, tunable, tune
from fugue_tune.exceptions import FugueTuneCompileError


def test_space_to_df():
    with FugueWorkflow() as dag:
        df = space_to_df(dag, Space(a=Grid(0, 1), b=Grid(2, 3)))
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
        df = space_to_df(
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
    def t1(a: int, b: int) -> float:
        return a + b

    for distributable in [True, False, None]:
        with FugueWorkflow() as dag:
            df = space_to_df(dag, Space(a=Grid(0, 1), b=Grid(2, 3)))
            tune(df, t1, distributable=distributable).show()

    @tunable()
    def t2(e: ExecutionEngine, a: int, b: int) -> float:
        assert isinstance(e, ExecutionEngine)
        return a + b

    for distributable in [False, None]:
        with FugueWorkflow() as dag:
            df = space_to_df(dag, Space(a=Grid(0, 1), b=Grid(2, 3)))
            tune(df, t2, distributable=distributable).show()
            
    # equivalent syntax sugar
    with FugueWorkflow() as dag:
        t2.space(a=Grid(0, 1), b=Grid(2, 3)).tune(dag).show()

    with raises(FugueTuneCompileError):
        with FugueWorkflow() as dag:
            df = space_to_df(dag, Space(a=Grid(0, 1), b=Grid(2, 3)))
            tune(df, t2, distributable=True).show()


def test_tune_df(tmpdir):
    @tunable()
    def t1(a: int, df: pd.DataFrame, b: int) -> float:
        return float(a + b + df["y"].sum())

    e = NativeExecutionEngine(conf={"fugue.temp.path": str(tmpdir)})

    for distributable in [True, False, None]:
        with FugueWorkflow(e) as dag:
            s = space_to_df(dag, Space(a=Grid(0, 1), b=Grid(2, 3)), batch_size=3)
            t = dag.df([[0, 1], [1, 2], [0, 2]], "x:int,y:int").partition(by=["x"])
            df = serialize_df(t, "df", str(tmpdir)).cross_join(s.broadcast())
            tune(df, t1, distributable=distributable).show()

    for distributable in [True, False, None]:
        with FugueWorkflow(e) as dag:
            df = dag.df([[0, 1], [1, 2], [0, 2]], "x:int,y:int")
            t1.space(a=Grid(0, 1), b=Grid(2, 3)).tune(df).show()
