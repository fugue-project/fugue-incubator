from typing import Any, Dict, Tuple

import numpy as np
from pytest import raises

from fugue_tune.exceptions import FugueTuneRuntimeError
from fugue_tune.tunable import SimpleTunable


def test_tunable():
    t = _MockTunable()
    t.run(a=1, b=2)
    assert 3.0 == t.error
    assert t.metadata == {}

    t = _MockTunable()
    t.run(a=1, m=2)
    assert 3.0 == t.error
    assert t.metadata == {"m": 2}

    t = _MockTunable()
    raises(FugueTuneRuntimeError, lambda: t.error)
    raises(FugueTuneRuntimeError, lambda: t.metadata)
    raises(FugueTuneRuntimeError, lambda: t.execution_engine)


class _MockTunable(SimpleTunable):
    def tune(self, **kwargs: Any) -> Tuple[float, Dict[str, Any]]:
        error = np.double(sum(kwargs.values()))
        if "m" in kwargs:
            meta = {"m": kwargs["m"]}
        else:
            meta = None
        return error, meta
