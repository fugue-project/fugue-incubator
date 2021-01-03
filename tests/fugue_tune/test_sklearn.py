import pickle

import numpy as np
import pandas as pd
from fugue import FugueWorkflow
from pytest import raises
from sklearn.base import is_classifier, is_regressor
from sklearn.linear_model import LinearRegression

from fugue_tune import Space
from fugue_tune.sklearn import _sk_cv, _to_model, _to_model_str, build_sk_cv
from fugue_tune.sklearn import sk_space as ss
from fugue_tune.sklearn import suggest_sk_model
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
    )
    assert len(res) == 4
    print(res)


def _create_mock_data():
    np.random.seed(0)
    df = pd.DataFrame(np.random.rand(100, 3), columns=["f_a", "f_b", "f_c"])
    df["d"] = "x"
    df["l"] = df["f_a"] * 3 + df["f_b"] * 4 + df["f_c"] * 5 + 100
    df["p"] = np.random.randint(low=0, high=4, size=(100, 1))
    return df
