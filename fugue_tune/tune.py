import json
from typing import Any, Dict, Optional, Iterable

from fugue import WorkflowDataFrame, DataFrame, ExecutionEngine, IterableDataFrame
from triad.utils.convert import get_caller_global_local_vars

from fugue_tune.convert import _to_tunable


def grid_tune(
    params_df: WorkflowDataFrame, tunable: Any, distributable: Optional[bool] = None
) -> WorkflowDataFrame:
    t = _to_tunable(  # type: ignore
        tunable, *get_caller_global_local_vars(), distributable
    )
    if distributable is None:
        distributable = t.distributable

    # input_has: __fmin_params__:str
    # schema: *,__fmin_value__:double,__fmin_metadata__:str
    def compute_transformer(df: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
        for row in df:
            params = json.loads(row["__fmin_params__"])
            t.run(**params)
            row["__fmin_value__"] = t.error
            row["__fmin_metadata__"] = json.dumps(t.metadata)
            yield row

    # input_has: __fmin_params__:str
    def compute_processor(engine: ExecutionEngine, df: DataFrame) -> DataFrame:
        def get_rows() -> Iterable[Any]:
            keys = list(df.schema.names) + ["__fmin_value__", "__fmin_metadata__"]
            for row in compute_transformer(df.as_dict_iterable()):
                yield [row[k] for k in keys]

        t._execution_engine = engine  # type:ignore
        return IterableDataFrame(
            get_rows(), df.schema + "__fmin_value__:double,__fmin_metadata__:str"
        )

    if not distributable:
        return params_df.process(compute_processor)
    else:
        return params_df.partition(num="ROWCOUNT", algo="even").transform(
            compute_transformer
        )
