from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional, Tuple, no_type_check

from triad import assert_or_throw

from fugue_tune.iter import dict_product, product


class Grid(object):
    def __init__(self, *args: Any):
        self._values = list(args)

    def __iter__(self) -> Iterable[Any]:
        yield from self._values


class StochasticExpression(object):
    @property
    def jsondict(self) -> Dict[str, Any]:  # pragma: no cover
        raise NotImplementedError

    def __eq__(self, other: Any):
        return isinstance(other, type(self)) and self.jsondict == other.jsondict


class Choice(StochasticExpression):
    def __init__(self, *args: Any):
        self._values = list(args)

    @property
    def values(self) -> List[Any]:
        return self._values

    @property
    def jsondict(self) -> Dict[str, Any]:
        return dict(_expr_="choice", values=self.values)


class Rand(StochasticExpression):
    def __init__(
        self,
        start: float,
        end: float,
        q: Optional[float] = None,
        log: bool = False,
        normal: bool = False,
    ):
        self.start = start
        self.end = end
        self.q = q
        self.log = log
        self.normal = normal

    @property
    def jsondict(self) -> Dict[str, Any]:
        res = dict(
            _expr_="rand",
            start=self.start,
            end=self.end,
            log=self.log,
            normal=self.normal,
        )
        if self.q is not None:
            res["q"] = self.q
        return res


class RandInt(Rand):
    def __init__(
        self,
        start: int,
        end: int,
        log: bool = False,
        normal: bool = False,
    ):
        super().__init__(start, end, q=1, log=log, normal=normal)

    @property
    def jsondict(self) -> Dict[str, Any]:
        return dict(
            _expr_="randint",
            start=self.start,
            end=self.end,
            log=self.log,
            normal=self.normal,
        )


def decode(value: Any) -> Any:
    if isinstance(value, str):
        return value
    elif isinstance(value, list):
        return [decode(v) for v in value]
    elif isinstance(value, dict):
        if "_expr_" in value:
            e = value.pop("_expr_")
            if e == "choice":  # TODO: embeded rand is not supported
                return Choice(*value["values"])
            if e == "rand":
                return Rand(**value)
            if e == "randint":
                return RandInt(**value)
            raise ValueError(e)  # pragma: no cover
        else:
            return {k: decode(v) for k, v in value.items()}
    else:
        return value


# TODO: make this inherit from iterable?
class Space(object):
    def __init__(self, **kwargs: Any):
        self._value = deepcopy(kwargs)
        self._grid: List[List[Tuple[Any, Any, Any]]] = []
        for k in self._value.keys():
            self._search(self._value, k)

    def __iter__(self) -> Iterable[Dict[str, Any]]:
        for tps in product(self._grid, safe=True, remove_empty=True):  # type: ignore
            for tp in tps:
                tp[0][tp[1]] = tp[2]
            yield deepcopy(self._value)

    def encode(self) -> Iterable[Any]:
        for s in self:  # type: ignore
            yield self._encode_value(s)

    def __mul__(self, other: Any) -> "HorizontalSpace":
        return HorizontalSpace(self, other)

    def __add__(self, other: Any) -> "VerticalSpace":
        return VerticalSpace(self, other)

    def __radd__(self, other: Any) -> "Space":
        assert_or_throw(
            other is None or (isinstance(other, int) and other == 0), ValueError(other)
        )
        return self

    def _search(self, parent: Any, key: Any) -> None:
        node = parent[key]
        if isinstance(node, Grid):
            self._grid.append(self._grid_wrapper(parent, key))
        elif isinstance(node, dict):
            for k in node.keys():
                self._search(node, k)
        elif isinstance(node, list):
            for i in range(len(node)):
                self._search(node, i)

    def _grid_wrapper(self, parent: Any, key: Any) -> List[Tuple[Any, Any, Any]]:
        return [(parent, key, x) for x in parent[key]]

    def _encode_value(self, value: Any) -> Any:
        if isinstance(value, StochasticExpression):
            return value.jsondict
        elif isinstance(value, str):
            return value
        elif isinstance(value, list):
            return [self._encode_value(v) for v in value]
        elif isinstance(value, dict):
            return {k: self._encode_value(v) for k, v in value.items()}
        return value


class HorizontalSpace(Space):
    def __init__(self, *args: Any, **kwargs: Any):
        self._groups: List[VerticalSpace] = []
        for x in args:
            if isinstance(x, HorizontalSpace):
                self._groups.append(VerticalSpace(x))
            elif isinstance(x, VerticalSpace):
                self._groups.append(x)
            elif isinstance(x, Space):
                self._groups.append(VerticalSpace(x))
            elif isinstance(x, dict):
                self._groups.append(VerticalSpace(HorizontalSpace(**x)))
            elif isinstance(x, list):
                self._groups.append(VerticalSpace(*x))
            else:
                raise ValueError(f"{x} is invalid")
        self._dict = {k: _SpaceValue(v) for k, v in kwargs.items()}

    @no_type_check  # TODO: remove this?
    def __iter__(self) -> Iterable[Dict[str, Any]]:
        dicts = list(dict_product(self._dict, safe=True))
        for spaces in product(
            [g.spaces for g in self._groups], safe=True, remove_empty=True
        ):
            for comb in product(list(spaces) + [dicts], safe=True, remove_empty=True):
                res: Dict[str, Any] = {}
                for d in comb:
                    res.update(d)
                yield res


class VerticalSpace(Space):
    def __init__(self, *args: Any):
        self._spaces: List[Space] = []
        for x in args:
            if isinstance(x, Space):
                self._spaces.append(x)
            elif isinstance(x, dict):
                self._spaces.append(Space(**x))
            elif isinstance(x, list):
                self._spaces.append(VerticalSpace(*x))
            else:
                raise ValueError(f"{x} is invalid")

    @property
    def spaces(self) -> List[Space]:
        return self._spaces

    def __iter__(self) -> Iterable[Dict[str, Any]]:
        for space in self._spaces:
            yield from space  # type: ignore


class _SpaceValue(object):
    def __init__(self, value: Any):
        self.value = value

    @no_type_check  # TODO: remove this?
    def __iter__(self) -> Iterable[Any]:
        if isinstance(self.value, (HorizontalSpace, VerticalSpace)):
            yield from self.value
        elif isinstance(self.value, dict):
            yield from dict_product(
                {k: _SpaceValue(v) for k, v in self.value.items()}, safe=True
            )
        elif isinstance(self.value, list):
            yield from product([_SpaceValue(v) for v in self.value], safe=True)
        else:
            yield self.value
