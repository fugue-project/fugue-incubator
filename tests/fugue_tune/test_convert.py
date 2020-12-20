from typing import Any, Dict, List, Tuple

from fugue import ExecutionEngine
from pytest import raises

from fugue_tune.convert import _to_tunable, tunable
from fugue_tune.exceptions import FugueTuneCompileError


def test_to_tunable():
    def t1(a: int) -> float:
        pass

    def t2(b: str) -> Tuple[float, Dict[str, Any]]:
        pass

    t22 = _to_tunable(t2)

    def t3() -> Tuple[float, Dict[str, Any]]:
        pass

    def t4(a: int) -> Tuple[float, List[str]]:
        pass

    def t5(e: ExecutionEngine, a: int) -> float:
        pass

    def t6(a: int, e: ExecutionEngine) -> float:
        pass

    assert t1 is _to_tunable(t1)._func
    assert _to_tunable(t1).distributable
    assert _to_tunable(t1, distributable=True).distributable
    assert not _to_tunable(t1, distributable=False).distributable
    assert t2 is _to_tunable(_to_tunable(t2))._func
    assert t2 is _to_tunable("t22")._func
    assert t1 is _to_tunable("t1")._func
    assert t2 is _to_tunable("t2")._func
    assert t5 is _to_tunable(t5)._func
    assert not _to_tunable(t5).distributable
    assert not _to_tunable(t5, distributable=False).distributable
    # with execution engine, distributable can't be true
    raises(FugueTuneCompileError, lambda: _to_tunable(t5, distributable=True))

    # return type must be float or Tuple[float,Dict[str,Any]]
    # input must not be empty
    with raises(FugueTuneCompileError):
        _to_tunable(t3)

    with raises(FugueTuneCompileError):
        _to_tunable("t3")

    with raises(FugueTuneCompileError):
        _to_tunable(t4)

    with raises(FugueTuneCompileError):
        _to_tunable(t6)


def test_deco():
    @tunable()
    def t1(a: int, b: int) -> float:
        return a + b

    @tunable()
    def t2(a: int, b: int) -> Tuple[float, Dict[str, Any]]:
        return a + b, {"x": 1}

    assert 5 == t1(2, 3)
    t1.run(a=3, b=4)
    assert 7 == t1.error
    assert t1.metadata == {}
    assert t1.distributable

    assert t2(2, 3) == (5, {"x": 1})
    t2.run(a=3, b=4)
    assert 7 == t2.error
    assert t2.metadata == {"x": 1}
