import pickle

import numpy as np
import pandas as pd
from fugue import FugueWorkflow
from pytest import raises
from sklearn.base import is_classifier, is_regressor
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingRegressor

from fugue_tune import Space
from fugue_tune.sklearn import (
    _process_stack_space,
    _sk_cv,
    _sk_stack_cv,
    _to_model,
    _to_model_str,
    build_sk_cv,
)
from fugue_tune.sklearn import sk_space as ss
from fugue_tune.sklearn import suggest_sk_model, suggest_sk_stacking_model
from fugue_tune.space import Grid


def test_to_model():
    assert is_classifier(_to_model("sklearn.ensemble.RandomForestClassifier"))
    assert is_regressor(_to_model(LinearRegression))
    raises(TypeError, lambda: _to_model("dummy"))
    raises(TypeError, lambda: _to_model("fugue_tune.space.Space"))
    raises(TypeError, lambda: _to_model(Space))


def test_to_model_str():
    assert "sklearn.linear_model._base.LinearRegression" == _to_model_str(
        LinearRegression
    )
    assert "sklearn.linear_model._base.LinearRegression" == _to_model_str(
        "sklearn.linear_model.LinearRegression"
    )
    assert _to_model(_to_model_str(LinearRegression)) is LinearRegression


def test_tunable_sk_cv(tmpdir):
    res = _sk_cv(
        "sklearn.linear_model.LinearRegression",
        _create_mock_data(),
        _sk__scoring="neg_mean_absolute_error",
        _sk__label_col="l",
        _sk__feature_prefix="f_",
        fit_intercept=True,
    )
    assert res["error"] < 0.1
    assert _to_model(res["hp"]["_sk__model"]) is LinearRegression
    assert res["hp"]["fit_intercept"]
    assert isinstance(res["metadata"]["cv_scores"], list)
    assert "model_path" not in res["metadata"]

    res = _sk_cv(
        "sklearn.linear_model.LinearRegression",
        _create_mock_data(),
        _sk__scoring="neg_mean_absolute_error",
        _sk__feature_prefix="f_",
        _sk__label_col="l",
        _sk__save_path=str(tmpdir),
        fit_intercept=True,
    )
    obj = pickle.load(open(res["metadata"]["model_path"], mode="rb"))
    assert isinstance(obj, LinearRegression)


def test_tunable_sk_stack_cv(tmpdir):
    res = _sk_stack_cv(
        "sklearn.linear_model.LinearRegression",
        '[{"_sk__model": "sklearn.linear_model._base.LinearRegression", "normalize": true},'
        '{"_sk__model": "sklearn.linear_model._base.LinearRegression", "normalize": false}]',
        _create_mock_data(),
        _sk__scoring="neg_mean_absolute_error",
        _sk__label_col="l",
        _sk__feature_prefix="f_",
        fit_intercept=True,
        _sk__save_path=str(tmpdir),
    )
    print(res)
    assert res["error"] < 0.1
    assert _to_model(res["hp"]["_sk__model"]) is StackingRegressor
    assert (
        _to_model(res["hp"]["_sk__estimators"]["stacking"]["_sk__model"])
        is LinearRegression
    )
    assert res["hp"]["_sk__estimators"]["stacking"]["fit_intercept"]
    assert isinstance(res["metadata"]["cv_scores"], list)

    obj = pickle.load(open(res["metadata"]["model_path"], mode="rb"))
    assert isinstance(obj, StackingRegressor)


def test_build_sk_cv(tmpdir):
    space = sum(
        [
            ss(LinearRegression, fit_intercept=Grid(True, False)),
            ss(LinearRegression, normalize=Grid(True, False)),
        ]
    )
    dag = FugueWorkflow()
    build_sk_cv(
        space,
        dag.df(_create_mock_data()),
        scoring="neg_mean_absolute_error",
        cv=4,
        label_col="l",
        feature_prefix="f_",
        save_path=str(tmpdir),
    ).tune(distributable=False, serialize_path=str(tmpdir)).show()
    dag.run()


def test_suggest_sk_model(tmpdir):
    space = sum(
        [
            ss(LinearRegression, fit_intercept=Grid(True, False)),
            ss(LinearRegression, normalize=Grid(True, False)),
        ]
    )
    res = suggest_sk_model(
        space,
        _create_mock_data(),
        scoring="neg_mean_absolute_error",
        serialize_path=str(tmpdir),
        label_col="l",
        feature_prefix="f_",
        save_model=True,
        partition_keys=["p"],
        visualize_top_n=2,
    )
    assert len(res) == 4
    print(res)


def test_suggest_sk_stacking_model(tmpdir):
    space = sum(
        [
            ss(LinearRegression, fit_intercept=Grid(True, False)),
            ss(Ridge, alpha=Grid(0.1, 0.2)),
        ]
    )
    space2 = sum(
        [
            ss(LinearRegression, normalize=Grid(True, False)),
        ]
    )
    res = suggest_sk_stacking_model(
        space,
        space2,
        _create_mock_data(),
        scoring="neg_mean_absolute_error",
        serialize_path=str(tmpdir),
        label_col="l",
        feature_prefix="f_",
        save_model=True,
        partition_keys=["p"],
        top_n=2,
    )
    assert len(res) == 4

    space = sum(
        [
            ss(LogisticRegression),
            ss(RandomForestClassifier),
        ]
    )
    space2 = sum(
        [
            ss(LogisticRegression),
        ]
    )
    res = suggest_sk_stacking_model(
        space,
        space2,
        _create_mock_data(regression=False),
        scoring="neg_mean_absolute_error",
        serialize_path=str(tmpdir),
        label_col="l",
        feature_prefix="f_",
        save_model=True,
        top_n=2,
    )
    assert len(res) == 1
    print(res)


def test_process_stack_space(tmpdir):
    space1 = ss(LinearRegression, normalize=Grid(True, False))
    space2 = ss(LinearRegression, fit_intercept=Grid(True, False))
    dag = FugueWorkflow()
    result0 = build_sk_cv(
        space1,
        dag.df(_create_mock_data()),
        scoring="neg_mean_absolute_error",
        cv=2,
        label_col="l",
        feature_prefix="f_",
    ).tune(distributable=False, serialize_path=str(tmpdir))
    res0 = result0.process(_process_stack_space, params=dict(keys=[], space=space2))
    res0.show()

    result1 = build_sk_cv(
        space1,
        dag.df(_create_mock_data()).partition(by=["p"]),
        scoring="neg_mean_absolute_error",
        cv=2,
        label_col="l",
        feature_prefix="f_",
    ).tune(distributable=False, serialize_path=str(tmpdir))
    res1 = result1.process(_process_stack_space, params=dict(keys=["p"], space=space2))
    dag.run()

    assert 2 == len(res0.result.as_array())
    assert 8 == len(res1.result.as_array())


def _create_mock_data(regression=True):
    np.random.seed(0)
    df = pd.DataFrame(np.random.rand(100, 3), columns=["f_a", "f_b", "f_c"])
    df["d"] = "x"
    if regression:
        df["l"] = df["f_a"] * 3 + df["f_b"] * 4 + df["f_c"] * 5 + 100
    else:
        df["l"] = (df["f_a"] * 3 - df["f_b"] * 4 + df["f_c"] * 5) > 0.5
    df["p"] = np.random.randint(low=0, high=4, size=(100, 1))
    return df
