import json
import os
import random
from typing import Any, Dict, Iterable, List, Optional, Set
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
from triad import assert_or_throw
from triad.utils.convert import get_caller_global_local_vars

from fugue_tune.convert import _to_tunable
from fugue_tune.space import Choice, Grid, Rand, Space


class Tuner(object):
    def tune(  # noqa: C901
        self,
        params_df: WorkflowDataFrame,
        tunable: Any,
        distributable: Optional[bool] = None,
    ) -> WorkflowDataFrame:
        t = _to_tunable(  # type: ignore
            tunable, *get_caller_global_local_vars(), distributable
        )
        if distributable is None:
            distributable = t.distributable

        # input_has: __fmin_params__:str
        # schema: *,__fmin_value__:double,__fmin_metadata__:str
        def compute_transformer(
            df: Iterable[Dict[str, Any]], load: Any = None
        ) -> Iterable[Dict[str, Any]]:
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
                    t.run(**dfs, **params)
                    res = dict(row)
                    res["__fmin_params__"] = json.dumps(
                        {pk: pv for pk, pv in t.hp.items() if pk not in dfs_keys}
                    )
                    res["__fmin_value__"] = t.error
                    res["__fmin_metadata__"] = json.dumps(t.metadata)
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

    def serialize_df(
        self, df: WorkflowDataFrame, name: str, path: str
    ) -> WorkflowDataFrame:
        pre_partition = df.partition_spec
        if len(pre_partition.partition_by) == 0:

            def save_single_file(e: ExecutionEngine, input: DataFrame) -> DataFrame:
                fp = os.path.join(path, str(uuid4()) + ".parquet")
                e.save_df(input, fp, force_single=True)
                return ArrayDataFrame([[fp]], f"__df_{name}:str")

            return df.process(save_single_file)
        else:

            class SavePartition(Transformer):
                def get_output_schema(self, df: DataFrame) -> Any:
                    dfn = self.params.get_or_throw("name", str)
                    return self.key_schema + f"__df_{dfn}:str"

                def transform(self, df: LocalDataFrame) -> LocalDataFrame:
                    fp = os.path.join(
                        self.params.get_or_throw("path", str), str(uuid4()) + ".parquet"
                    )
                    df.as_pandas().to_parquet(fp)
                    return ArrayDataFrame(
                        [self.cursor.key_value_array + [fp]], self.output_schema
                    )

            return df.transform(SavePartition, params={"path": path, "name": name})

    def space_to_df(
        self, wf: FugueWorkflow, space: Space, batch_size: int = 1, shuffle: bool = True
    ) -> WorkflowDataFrame:
        def get_data() -> Iterable[List[Any]]:
            it = list(space)  # type: ignore
            if shuffle:
                random.seed(0)
                random.shuffle(it)
            res: List[Any] = []
            for a in it:
                res.append(self._convert_hp(a))
                if batch_size == len(res):
                    yield [json.dumps(res)]
                    res = []
            if len(res) > 0:
                yield [json.dumps(res)]

        return wf.df(IterableDataFrame(get_data(), "__fmin_params__:str"))

    def _convert_hp(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {k: self._convert_single(v) for k, v in params.items()}

    def _convert_single(self, param: Any) -> Any:
        assert_or_throw(
            not isinstance(param, (Grid, Rand, Choice)), NotImplementedError(param)
        )
        return param
