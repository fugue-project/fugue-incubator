import json
import os
import pickle
from importlib import import_module
from typing import Any, Dict, Iterable, List, Optional, Tuple
from uuid import uuid4

import numpy as np
import pandas as pd
from fugue import (
    DataFrame,
    ExecutionEngine,
    FugueWorkflow,
    NativeExecutionEngine,
    WorkflowDataFrame,
)
from triad import FileSystem, assert_or_throw
from triad.utils.convert import get_full_type_path, to_instance, to_type

from fugue_tune.space import Space
from fugue_tune.tune import (
    ObjectiveRunner,
    TunableWithSpace,
    select_best,
    serialize_df,
    space_to_df,
    tunable,
    tune,
    visualize_top_n as visualize_top,
)
from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.base import is_classifier, is_regressor

_EMPTY_DF = pd.DataFrame()
_EMPTY_LIST: List[str] = []


def suggest_sk_model(
    space: Space,
    train_df: Any,
    scoring: str,
    serialize_path: str,
    cv: int = 5,
    feature_prefix: str = "",
    label_col: str = "label",
    save_model: bool = False,
    partition_keys: List[str] = _EMPTY_LIST,
    top_n: int = 1,
    visualize_top_n: int = 0,
    objective_runner: Optional[ObjectiveRunner] = None,
    distributable: Optional[bool] = None,
    execution_engine: Any = NativeExecutionEngine,
) -> List[Dict[str, Any]]:
    e = to_instance(execution_engine, ExecutionEngine)
    model_path = serialize_path if save_model else ""

    dag = FugueWorkflow()
    df = dag.df(train_df)
    if len(partition_keys) > 0:
        df = df.partition(by=partition_keys)
    skcv = build_sk_cv(
        space=space,
        train_df=df,
        scoring=scoring,
        cv=cv,
        feature_prefix=feature_prefix,
        label_col=label_col,
        save_path=model_path,
    )
    result = skcv.tune(
        objective_runner=objective_runner,
        distributable=distributable,
        serialize_path=serialize_path,
        shuffle=True,
    ).persist()
    best = select_best(result, top=top_n) if top_n > 0 else result
    visualize_top(result, top=visualize_top_n)
    dag.run(e)
    return list(best.result.as_dict_iterable())


def suggest_sk_stacking_model(
    space: Space,
    stack_space: Space,
    train_df: Any,
    scoring: str,
    serialize_path: str,
    cv: int = 5,
    feature_prefix: str = "",
    label_col: str = "label",
    save_model: bool = False,
    partition_keys: List[str] = _EMPTY_LIST,
    top_n: int = 1,
    visualize_top_n: int = 0,
    objective_runner: Optional[ObjectiveRunner] = None,
    distributable: Optional[bool] = None,
    execution_engine: Any = NativeExecutionEngine,
    stack_cv: int = 2,
    stack_method: str = "auto",
    stack_passthrough: bool = False,
) -> List[Dict[str, Any]]:
    e = to_instance(execution_engine, ExecutionEngine)
    model_path = serialize_path if save_model else ""

    dag = FugueWorkflow()
    df = dag.df(train_df)
    if len(partition_keys) > 0:
        df = df.partition(by=partition_keys)
    skcv = build_sk_cv(
        space=space,
        train_df=df,
        scoring=scoring,
        cv=cv,
        feature_prefix=feature_prefix,
        label_col=label_col,
    )
    result = skcv.tune(
        objective_runner=objective_runner,
        distributable=distributable,
        serialize_path=serialize_path,
        shuffle=True,
    ).persist()
    best_models = select_best(result.transform(_extract_model), top=1)
    if top_n > 0:
        best_models = select_best(best_models.drop(["_sk__model"]), top=top_n)
    kwargs = Space(
        _sk__scoring=scoring,
        _sk__cv=cv,
        _sk__feature_prefix=feature_prefix,
        _sk__label_col=label_col,
        _sk__save_path=model_path,
        _sk__stack_cv=stack_cv,
        _sk__method=stack_method,
        _sk__passthrough=stack_passthrough,
    )
    space_df = best_models.process(
        _process_stack_space,
        params=dict(keys=partition_keys, space=stack_space * kwargs),
    )
    data = serialize_df(df, name="_sk__train_df", path=serialize_path)
    if len(partition_keys) > 0:
        data = data.inner_join(space_df.broadcast())
    else:
        data = data.cross_join(space_df.broadcast())
    result = tune(
        data,
        tunable=tunable(_sk_stack_cv),
        distributable=distributable,
        objective_runner=objective_runner,
    )
    best = select_best(result, top=1)
    visualize_top(result, top=visualize_top_n)
    dag.run(e)
    return list(best.result.as_dict_iterable())


def sk_space(model: Any, **kwargs: Any) -> Space:
    return Space(_sk__model=_to_model_str(model), **kwargs)


def build_sk_cv(
    space: Space,
    train_df: WorkflowDataFrame,
    scoring: str,
    cv: int = 5,
    feature_prefix: str = "",
    label_col: str = "label",
    save_path: str = "",
) -> TunableWithSpace:
    kwargs = dict(
        _sk__train_df=train_df,
        _sk__scoring=scoring,
        _sk__cv=cv,
        _sk__feature_prefix=feature_prefix,
        _sk__label_col=label_col,
        _sk__save_path=save_path,
    )
    return tunable(_sk_cv).space(space, **kwargs)  # type:ignore


def _sk_cv(
    _sk__model: str,
    _sk__train_df: pd.DataFrame,
    _sk__scoring: Any,
    _sk__cv: int = 5,
    _sk__feature_prefix: str = "",
    _sk__label_col: str = "label",
    _sk__save_path: str = "",
    **kwargs: Any,
) -> Dict[str, Any]:
    model = _to_model(_sk__model)(**kwargs)
    train_df = _sk__train_df.sample(frac=1, random_state=0).reset_index(drop=True)

    train_x = train_df.drop([_sk__label_col], axis=1)
    cols = [x for x in train_x.columns if x.startswith(_sk__feature_prefix)]
    train_x = train_x[cols]
    train_y = train_df[_sk__label_col]

    s = cross_val_score(model, train_x, train_y, cv=_sk__cv, scoring=_sk__scoring)
    metadata = dict(sk_model=_sk__model, cv_scores=[float(x) for x in s])
    if _sk__save_path != "":
        model.fit(train_x, train_y)
        fp = os.path.join(_sk__save_path, str(uuid4()) + ".pkl")
        with FileSystem().openbin(fp, mode="wb") as f:
            pickle.dump(model, f)
        metadata["model_path"] = fp
    return dict(
        error=-np.mean(s), hp=dict(_sk__model=_sk__model, **kwargs), metadata=metadata
    )


def _sk_stack_cv(
    _sk__model: str,
    _sk__estimators: str,
    _sk__train_df: pd.DataFrame,
    _sk__scoring: Any,
    _sk__stack_cv: int = 2,
    _sk__method: str = "auto",
    _sk__passthrough: bool = False,
    _sk__cv: int = 5,
    _sk__feature_prefix: str = "",
    _sk__label_col: str = "label",
    _sk__save_path: str = "",
    **kwargs: Any,
) -> Dict[str, Any]:
    final_estimator = _to_model(_sk__model)(**kwargs)
    estimators: List[Tuple[str, Any]] = []
    for i, d in enumerate(json.loads(_sk__estimators)):
        key = f"_{i}"
        m = _to_model(d.pop("_sk__model"))
        estimators.append((key, m(**d)))
    if is_classifier(final_estimator):
        model = StackingClassifier(
            estimators,
            final_estimator,
            cv=_sk__stack_cv,
            stack_method=_sk__method,
            passthrough=_sk__passthrough,
            n_jobs=kwargs.get("n_jobs", 1),
        )
    else:
        model = StackingRegressor(
            estimators,
            final_estimator,
            cv=_sk__stack_cv,
            passthrough=_sk__passthrough,
            n_jobs=kwargs.get("n_jobs", 1),
        )
    train_df = _sk__train_df.sample(frac=1, random_state=0).reset_index(drop=True)

    train_x = train_df.drop([_sk__label_col], axis=1)
    cols = [x for x in train_x.columns if x.startswith(_sk__feature_prefix)]
    train_x = train_x[cols]
    train_y = train_df[_sk__label_col]

    s = cross_val_score(model, train_x, train_y, cv=_sk__cv, scoring=_sk__scoring)
    metadata = dict(sk_model=get_full_type_path(model), cv_scores=[float(x) for x in s])
    if _sk__save_path != "":
        model.fit(train_x, train_y)
        fp = os.path.join(_sk__save_path, str(uuid4()) + ".pkl")
        with FileSystem().openbin(fp, mode="wb") as f:
            pickle.dump(model, f)
        metadata["model_path"] = fp
    return dict(
        error=-np.mean(s),
        hp=dict(
            _sk__model=get_full_type_path(model),
            _sk__estimators=dict(
                **{f"_{i}": d for i, d in enumerate(json.loads(_sk__estimators))},
                stacking=dict(_sk__model=_sk__model, **kwargs),
            ),
            _sk__stack_cv=_sk__stack_cv,
            _sk__method=_sk__method,
            _sk__passthrough=_sk__passthrough,
        ),
        metadata=metadata,
    )


def _to_model(obj: Any) -> Any:
    if isinstance(obj, str):
        parts = obj.split(".")
        if len(parts) > 1:
            import_module(".".join(parts[:-1]))
        obj = to_type(obj)
    assert_or_throw(
        is_classifier(obj) or is_regressor(obj),
        TypeError(f"{obj} is neither a sklearn classifier or regressor"),
    )
    return obj


def _to_model_str(model: Any) -> Any:
    if isinstance(model, str):
        model = _to_model(model)
    return get_full_type_path(model)


# schema: *,_sk__model:str
def _extract_model(df: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    for row in df:
        row["_sk__model"] = json.loads(row["__fmin_params__"])["_sk__model"]
        yield row


def _process_stack_space(
    engine: ExecutionEngine, df: DataFrame, keys: List[str], space: Space
) -> DataFrame:
    fe_schema = df.schema.extract(keys) + "__fmin_fe__:str"

    def _merge_space(df: List[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
        p = json.dumps([json.loads(row["__fmin_params__"]) for row in df])
        res = df[0]
        res["__fmin_fe__"] = p
        yield res

    # schema: *-__fmin_fe__
    def _construct_final_space(
        df: Iterable[Dict[str, Any]]
    ) -> Iterable[Dict[str, Any]]:
        for row in df:
            op = json.loads(row["__fmin_params__"])
            for o in op:
                o["_sk__estimators"] = row["__fmin_fe__"]
            row["__fmin_params__"] = json.dumps(op)
            yield row

    with FugueWorkflow(engine) as dag:
        ddf = dag.df(df)
        space_df = space_to_df(dag, space).broadcast()
        if len(keys) == 0:
            fe = ddf.process(_merge_space, schema=fe_schema)
        else:
            fe = ddf.partition(by=keys).transform(_merge_space, schema=fe_schema)
        result = fe.cross_join(space_df).transform(_construct_final_space)

    return result.result
