import os
import pickle
from importlib import import_module
from typing import Any, Dict
from uuid import uuid4

import numpy as np
import pandas as pd
from fugue import WorkflowDataFrame
from triad import FileSystem, assert_or_throw
from triad.utils.convert import get_full_type_path, to_type

from fugue_tune.space import Space
from fugue_tune.tune import TunableWithSpace, tunable
from sklearn.base import is_classifier, is_regressor
from sklearn.model_selection import KFold, cross_val_score

_EMPTY_DF = pd.DataFrame()


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


def sk_space(model: Any, **kwargs: Any) -> Space:
    return Space(_sk__model=_to_model_str(model), **kwargs)


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
    train_x = _sk__train_df.drop([_sk__label_col], axis=1)
    cols = [x for x in train_x.columns if x.startswith(_sk__feature_prefix)]
    train_x = train_x[cols]
    train_y = _sk__train_df[_sk__label_col]

    kf = KFold(n_splits=_sk__cv, random_state=0, shuffle=True)
    s = cross_val_score(model, train_x, train_y, cv=kf, scoring=_sk__scoring)
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
