import json
from typing import Any, Iterable, List

from fugue import ExecutionEngine, FugueWorkflow, IterableDataFrame, WorkflowDataFrame
from pytest import raises

from fugue_tune.exceptions import FugueTuneCompileError
from fugue_tune.space import Grid, Space
from fugue_tune.tune import grid_tune


def _space_to_df(wf: FugueWorkflow, space: Space) -> WorkflowDataFrame:
    def get_data() -> Iterable[List[Any]]:
        for item in space:
            yield [json.dumps(item)]

    return wf.df(IterableDataFrame(get_data(), "__fmin_params__:str"))


def test_grid_tune():
    def t1(a: int, b: int) -> float:
        return a + b

    for distributable in [True, False, None]:
        with FugueWorkflow() as dag:
            df = _space_to_df(dag, Space(a=Grid(0, 1), b=Grid(2, 3)))
            grid_tune(df, t1, distributable=distributable).show()

    def t2(e: ExecutionEngine, a: int, b: int) -> float:
        assert isinstance(e, ExecutionEngine)
        return a + b

    for distributable in [False, None]:
        with FugueWorkflow() as dag:
            df = _space_to_df(dag, Space(a=Grid(0, 1), b=Grid(2, 3)))
            grid_tune(df, t2, distributable=distributable).show()

    with raises(FugueTuneCompileError):
        with FugueWorkflow() as dag:
            df = _space_to_df(dag, Space(a=Grid(0, 1), b=Grid(2, 3)))
            grid_tune(df, t2, distributable=True).show()
