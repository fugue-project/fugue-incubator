import copy
import inspect
from typing import Any, Callable, Dict, List, Optional, no_type_check

from fugue import ExecutionEngine
from fugue._utils.interfaceless import (
    FunctionWrapper,
    _ExecutionEngineParam,
    _FuncParam,
    _OtherParam,
    is_class_method,
)
from triad import assert_or_throw
from triad.utils.convert import get_caller_global_local_vars, to_function

from fugue_tune.exceptions import FugueTuneCompileError
from fugue_tune.tunable import SimpleTunable, Tunable


def tunable(
    func: Optional[Callable] = None,
    distributable: Optional[bool] = None,
) -> Callable[[Any], "_FuncAsTunable"]:
    def deco(func: Callable) -> "_FuncAsTunable":
        assert_or_throw(
            not is_class_method(func),
            NotImplementedError("tunable decorator can't be used on class methods"),
        )
        return _FuncAsTunable.from_func(func, distributable=distributable)

    if func is None:
        return deco
    else:
        return deco(func)


def _to_tunable(
    obj: Any,
    global_vars: Optional[Dict[str, Any]] = None,
    local_vars: Optional[Dict[str, Any]] = None,
    distributable: Optional[bool] = None,
) -> Tunable:
    global_vars, local_vars = get_caller_global_local_vars(global_vars, local_vars)

    def get_tunable() -> Tunable:
        if isinstance(obj, Tunable):
            return copy.copy(obj)
        try:
            f = to_function(obj, global_vars=global_vars, local_vars=local_vars)
            # this is for string expression of function with decorator
            if isinstance(f, Tunable):
                return copy.copy(f)
            # this is for functions without decorator
            return _FuncAsTunable.from_func(f, distributable)
        except Exception as e:
            exp = e
        raise FugueTuneCompileError(f"{obj} is not a valid tunable function", exp)

    t = get_tunable()
    if distributable is None:
        distributable = t.distributable
    elif distributable:
        assert_or_throw(
            t.distributable, FugueTuneCompileError(f"{t} is not distributable")
        )
    return t


class _SingleParam(_FuncParam):
    def __init__(self, param: Optional[inspect.Parameter]):
        super().__init__(param, "float", "s")


class _DictParam(_FuncParam):
    def __init__(self, param: Optional[inspect.Parameter]):
        super().__init__(param, "Dict[str,Any]", "d")


class _TunableWrapper(FunctionWrapper):
    def __init__(self, func: Callable):
        super().__init__(func, "^e?[^e]+$", "^[sd]$")

    def _parse_param(
        self,
        annotation: Any,
        param: Optional[inspect.Parameter],
        none_as_other: bool = True,
    ) -> _FuncParam:
        if annotation is float:
            return _SingleParam(param)
        elif annotation is Dict[str, Any]:
            return _DictParam(param)
        elif annotation is ExecutionEngine:
            return _ExecutionEngineParam(param)
        else:
            return _OtherParam(param)

    @property
    def single(self) -> bool:
        return isinstance(self._rt, _SingleParam)

    @property
    def needs_engine(self) -> bool:
        return isinstance(self._params.get_value_by_index(0), _ExecutionEngineParam)


class _FuncAsTunable(SimpleTunable):
    @no_type_check
    def tune(self, **kwargs: Any) -> Dict[str, Any]:
        # pylint: disable=no-member
        args: List[Any] = [self.execution_engine] if self._needs_engine else []
        if self._single:
            return dict(error=self._func(*args, **kwargs))
        else:
            return self._func(*args, **kwargs)

    @no_type_check
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._func(*args, **kwargs)

    @property
    def distributable(self) -> bool:
        return self._distributable  # type: ignore

    @no_type_check
    @staticmethod
    def from_func(
        func: Callable, distributable: Optional[bool] = None
    ) -> "_FuncAsTunable":
        t = _FuncAsTunable()
        tw = _TunableWrapper(func)
        t._func = tw._func
        t._single = tw.single
        t._needs_engine = tw.needs_engine
        if distributable is None:
            t._distributable = not tw.needs_engine
        else:
            if distributable:
                assert_or_throw(
                    not tw.needs_engine,
                    "function with ExecutionEngine can't be distributable",
                )
            t._distributable = distributable

        return t
