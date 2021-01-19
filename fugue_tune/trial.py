import copy
from threading import RLock
from typing import Any, Callable, Dict, List
from uuid import uuid4

from fugue.rpc import RPCHandler


class TrialsTracker(RPCHandler):
    def __init__(self):
        super().__init__()
        self._tt_lock = RLock()
        self._raw_data: Dict[str, List[Dict[str, Any]]] = {}

    def get_raw_data(self) -> Dict[str, List[Dict[str, Any]]]:
        with self._tt_lock:
            return copy.deepcopy(self._raw_data)

    def __call__(self, method: str, **kwargs: Any) -> Any:
        return getattr(self, method)(**kwargs)

    def log_trial(self, trial_id: str, **kwargs: Any) -> None:
        with self._tt_lock:
            if trial_id not in self._raw_data:
                self._raw_data[trial_id] = [dict(kwargs)]
            else:
                self._raw_data[trial_id].append(dict(kwargs))

    def prune(self, trial_id: str) -> bool:
        return False


class TrialCallback(object):
    def __init__(self, callback: Callable):
        self._trial_id = str(uuid4())
        self._callback = callback

    @property
    def trial_id(self) -> str:
        return self._trial_id

    def log_trial(self, **kwargs: Any) -> None:
        self._callback(method="log_trial", trial_id=self.trial_id, **kwargs)

    def __getattr__(self, method: str) -> Callable:
        def _wrapper(**kwargs: Any) -> Any:
            return self._callback(method=method, trial_id=self.trial_id, **kwargs)

        return _wrapper
