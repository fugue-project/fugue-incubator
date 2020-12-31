from typing import Any, Dict

from fugue import ExecutionEngine
from triad import ParamDict

import fugue_tune.tune as ft
from fugue_tune.exceptions import FugueTuneRuntimeError
from fugue_tune.space import Space


class Tunable(object):
    def run(self, **kwargs: Any) -> None:  # pragma: no cover
        raise NotImplementedError

    def report(self, result: Dict[str, Any]) -> None:
        self._error = float(result["error"])
        self._hp = ParamDict(result.get("hp", None))
        self._metadata = ParamDict(result.get("metadata", None))

    @property
    def error(self) -> float:
        try:
            return self._error
        except Exception:
            raise FugueTuneRuntimeError("error is not set")

    @property
    def hp(self) -> ParamDict:
        try:
            return self._hp
        except Exception:
            raise FugueTuneRuntimeError("hp is not set")

    @property
    def metadata(self) -> ParamDict:
        try:
            return self._metadata
        except Exception:
            raise FugueTuneRuntimeError("metadata is not set")

    @property
    def distributable(self) -> bool:  # pragma: no cover
        return True

    @property
    def execution_engine(self) -> ExecutionEngine:
        # pylint: disable=no-member
        try:
            return self._execution_engine  # type: ignore
        except Exception:
            raise FugueTuneRuntimeError("execution_engine is not set")

    def space(self, **kwargs: Any) -> ft.TunableWithSpace:
        return ft.TunableWithSpace(self, Space(**kwargs))


class SimpleTunable(Tunable):
    def tune(self, **kwargs: Any) -> Dict[str, Any]:  # pragma: no cover
        raise NotImplementedError

    def run(self, **kwargs: Any) -> None:
        res = self.tune(**kwargs)
        if "hp" not in res:
            res["hp"] = kwargs
        self.report(res)
