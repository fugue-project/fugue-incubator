from typing import Any, Dict, Tuple

from fugue import ExecutionEngine
from triad import ParamDict

from fugue_tune.exceptions import FugueTuneRuntimeError


class Tunable(object):
    def run(self, **kwargs: Any) -> None:  # pragma: no cover
        raise NotImplementedError

    def report(self, error: Any, metadata: Any = None) -> None:
        self._error = float(error)
        self._metadata = ParamDict(metadata)

    @property
    def error(self) -> float:
        try:
            return self._error
        except Exception:
            raise FugueTuneRuntimeError("error is not set")

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


class SimpleTunable(Tunable):
    def tune(self, **kwargs: Any) -> Tuple[float, Dict[str, Any]]:  # pragma: no cover
        raise NotImplementedError

    def run(self, **kwargs: Any) -> None:
        score, metadata = self.tune(**kwargs)
        self.report(score, metadata)
