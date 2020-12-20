from typing import Any

from fugue.exceptions import FugueWorkflowCompileError, FugueWorkflowRuntimeError


class FugueTuneCompileError(FugueWorkflowCompileError):
    def __init__(self, *args: Any):
        super().__init__(*args)


class FugueTuneRuntimeError(FugueWorkflowRuntimeError):
    def __init__(self, *args: Any):
        super().__init__(*args)
