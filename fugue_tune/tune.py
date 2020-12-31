import copy
import json
import os
import random
from typing import Any, Dict, Iterable, List, Optional, Set, Union
from uuid import uuid4

import pandas as pd
from fugue import (
    ArrayDataFrame,
    DataFrame,
    ExecutionEngine,
    FugueWorkflow,
    IterableDataFrame,
    LocalDataFrame,
    Transformer,
    WorkflowDataFrame,
)
from triad import ParamDict
from triad.utils.convert import get_caller_global_local_vars

import fugue_tune.convert as fc
import fugue_tune.tunable as ft
from fugue_tune.space import Space, decode


class ObjectiveRunner(object):
    def run(
        self, tunable: "ft.Tunable", kwargs: Dict[str, Any], hp_keys: Set[str]
    ) -> Dict[str, Any]:
        tunable.run(**kwargs)
        hp = {k: v for k, v in tunable.hp.items() if k in hp_keys}
        return {"error": tunable.error, "hp": hp, "metadata": tunable.metadata}


class TunableWithSpace(object):
    def __init__(self, tunable: "ft.Tunable", space: Space):
        self._tunable = copy.copy(tunable)
        self._space = space

    @property
    def tunable(self) -> "ft.Tunable":
        return self._tunable

    @property
    def space(self) -> Space:
        return self._space

    def tune(
        self,
        source: Union[WorkflowDataFrame, FugueWorkflow] = None,
        distributable: Optional[bool] = None,
        objective_runner: Optional[ObjectiveRunner] = None,
        df_name: str = "df",
        serialize_path: str = "",
        batch_size: int = 1,
        shuffle: bool = True,
    ) -> WorkflowDataFrame:
        if isinstance(source, WorkflowDataFrame):
            df = source
            data = serialize_df(df, name=df_name, path=serialize_path)
            space_df = space_to_df(
                df.workflow, self.space, batch_size=batch_size, shuffle=shuffle
            )
            return tune(
                data.cross_join(space_df),
                tunable=self.tunable,
                distributable=distributable,
                objective_runner=objective_runner,
            )
        else:
            space_df = space_to_df(
                source, self.space, batch_size=batch_size, shuffle=shuffle
            )
            return tune(
                space_df,
                tunable=self.tunable,
                distributable=distributable,
                objective_runner=objective_runner,
            )


def tune(  # noqa: C901
    params_df: WorkflowDataFrame,
    tunable: Any,
    distributable: Optional[bool] = None,
    objective_runner: Optional[ObjectiveRunner] = None,
) -> WorkflowDataFrame:
    t = fc._to_tunable(  # type: ignore
        tunable, *get_caller_global_local_vars(), distributable
    )
    if distributable is None:
        distributable = t.distributable

    if objective_runner is None:
        objective_runner = ObjectiveRunner()

    # input_has: __fmin_params__:str
    # schema: *,__fmin_value__:double,__fmin_metadata__:str
    def compute_transformer(df: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
        for row in df:
            dfs: Dict[str, Any] = {}
            dfs_keys: Set[str] = set()
            for k, v in row.items():
                if k.startswith("__df_"):
                    key = k[len("__df_") :]
                    if v is not None:
                        dfs[key] = pd.read_parquet(v)
                    dfs_keys.add(key)
            for params in json.loads(row["__fmin_params__"]):
                p = decode(params)
                best = objective_runner.run(  # type: ignore
                    t, dict(**dfs, **p), set(p.keys())
                )
                res = dict(row)
                res["__fmin_params__"] = json.dumps(best["hp"])
                res["__fmin_value__"] = best["error"]
                res["__fmin_metadata__"] = json.dumps(best["metadata"])
                yield res

    # input_has: __fmin_params__:str
    def compute_processor(engine: ExecutionEngine, df: DataFrame) -> DataFrame:
        def get_rows() -> Iterable[Any]:
            keys = list(df.schema.names) + ["__fmin_value__", "__fmin_metadata__"]
            for row in compute_transformer(df.as_dict_iterable()):
                yield [row[k] for k in keys]

        t._execution_engine = engine  # type:ignore
        return ArrayDataFrame(
            get_rows(), df.schema + "__fmin_value__:double,__fmin_metadata__:str"
        )

    if not distributable:
        return params_df.process(compute_processor)
    else:
        return params_df.partition(num="ROWCOUNT", algo="even").transform(
            compute_transformer
        )


def serialize_df(df: WorkflowDataFrame, name: str, path: str = "") -> WorkflowDataFrame:
    pre_partition = df.partition_spec

    def _get_temp_path(p: str, conf: ParamDict) -> str:
        if p is not None and p != "":
            return p
        return conf.get_or_throw("fugue.temp.path", str)  # TODO: remove hard code

    if len(pre_partition.partition_by) == 0:

        def save_single_file(e: ExecutionEngine, input: DataFrame) -> DataFrame:
            p = _get_temp_path(path, e.conf)
            fp = os.path.join(p, str(uuid4()) + ".parquet")
            e.save_df(input, fp, force_single=True)
            return ArrayDataFrame([[fp]], f"__df_{name}:str")

        return df.process(save_single_file)
    else:

        class SavePartition(Transformer):
            def get_output_schema(self, df: DataFrame) -> Any:
                dfn = self.params.get_or_throw("name", str)
                return self.key_schema + f"__df_{dfn}:str"

            def transform(self, df: LocalDataFrame) -> LocalDataFrame:
                p = _get_temp_path(self.params.get("path", ""), self.workflow_conf)
                fp = os.path.join(p, str(uuid4()) + ".parquet")
                df.as_pandas().to_parquet(fp)
                return ArrayDataFrame(
                    [self.cursor.key_value_array + [fp]], self.output_schema
                )

        return df.transform(SavePartition, params={"path": path, "name": name})


def space_to_df(
    wf: FugueWorkflow, space: Space, batch_size: int = 1, shuffle: bool = True
) -> WorkflowDataFrame:
    def get_data() -> Iterable[List[Any]]:
        it = list(space.encode())  # type: ignore
        if shuffle:
            random.seed(0)
            random.shuffle(it)
        res: List[Any] = []
        for a in it:
            res.append(a)
            if batch_size == len(res):
                yield [json.dumps(res)]
                res = []
        if len(res) > 0:
            yield [json.dumps(res)]

    return wf.df(IterableDataFrame(get_data(), "__fmin_params__:str"))
