import json

from pytest import raises

from fugue_tune import Choice, Grid, Rand, RandInt, Space, decode
from fugue_tune.space import HorizontalSpace, VerticalSpace


def test_single_space():
    dicts = list(Space(a=1, b=Grid(2, 3, 4)))
    assert 3 == len(dicts)
    assert dict(a=1, b=2) == dicts[0]
    assert dict(a=1, b=3) == dicts[1]

    dicts = list(Space(a=Grid(None, "x"), b=Grid(2, 3)))
    assert 4 == len(dicts)

    dicts = list(Space(a=1, b=[Grid(2, 3), Grid(4, 5)]))
    assert 4 == len(dicts)
    assert dict(a=1, b=[2, 4]) == dicts[0]
    assert dict(a=1, b=[2, 5]) == dicts[1]
    assert dict(a=1, b=[3, 4]) == dicts[2]
    assert dict(a=1, b=[3, 5]) == dicts[3]

    dicts = list(Space(a=1, b=dict(x=Grid(2, 3), y=Grid(4, 5))))
    assert 4 == len(dicts)
    assert dict(a=1, b=dict(x=2, y=4)) == dicts[0]
    assert dict(a=1, b=dict(x=2, y=5)) == dicts[1]
    assert dict(a=1, b=dict(x=3, y=4)) == dicts[2]
    assert dict(a=1, b=dict(x=3, y=5)) == dicts[3]


def test_space_simple_dict():
    spaces = list(HorizontalSpace())
    assert 1 == len(spaces)
    assert {} == spaces[0]

    spaces = list(HorizontalSpace(a=10, b=[1, 2], c=dict(x=1)))
    assert 1 == len(spaces)
    assert dict(a=10, b=[1, 2], c=dict(x=1)) == spaces[0]

    spaces = list(HorizontalSpace(dict(a=10, b=[1, 2], c=dict(x=1))))
    assert 1 == len(spaces)
    assert dict(a=10, b=[1, 2], c=dict(x=1)) == spaces[0]

    spaces = list(HorizontalSpace(dict(a=10), dict(b=[1, 2], c=dict(x=1))))
    assert 1 == len(spaces)
    assert dict(a=10, b=[1, 2], c=dict(x=1)) == spaces[0]

    raises(ValueError, lambda: HorizontalSpace(10))


def test_spaces():
    spaces = list(VerticalSpace())
    assert 0 == len(spaces)

    spaces = list(VerticalSpace(dict(a=10)))
    assert [dict(a=10)] == spaces

    spaces = list(VerticalSpace(dict(a=10), [dict(b=11), dict(c=12)]))
    assert [dict(a=10), dict(b=11), dict(c=12)] == spaces

    spaces = list(VerticalSpace(HorizontalSpace(a=10), dict(b=10)))
    assert [dict(a=10), dict(b=10)] == spaces

    raises(ValueError, lambda: VerticalSpace(10))


def test_space_combo():
    spaces = list(HorizontalSpace(dict(a=10), []))
    assert [dict(a=10)] == spaces

    spaces = list(HorizontalSpace(dict(a=10), [dict(b=20), dict(c=30, a=11)]))
    assert 2 == len(spaces)
    assert dict(a=10, b=20) == spaces[0]
    assert dict(a=11, c=30) == spaces[1]

    spaces = list(
        HorizontalSpace(
            HorizontalSpace(a=10),
            VerticalSpace(dict(b=20), HorizontalSpace(c=30, a=None)),
        )
    )
    assert 2 == len(spaces)
    assert dict(a=10, b=20) == spaces[0]
    assert dict(a=None, c=30) == spaces[1]

    spaces = list(
        HorizontalSpace(
            dict(a=HorizontalSpace(dict(aa=10), VerticalSpace(dict(), dict(cc=12)))),
            VerticalSpace(dict(b=20), HorizontalSpace(c=30)),
        )
    )
    assert 4 == len(spaces)
    assert dict(a=dict(aa=10), b=20) == spaces[0]
    assert dict(a=dict(aa=10, cc=12), b=20) == spaces[1]
    assert dict(a=dict(aa=10), c=30) == spaces[2]
    assert dict(a=dict(aa=10, cc=12), c=30) == spaces[3]

    spaces = list(
        HorizontalSpace(a=VerticalSpace(HorizontalSpace(x=[1, 2]), dict(y=None)))
    )
    assert 2 == len(spaces)
    assert dict(a=dict(x=[1, 2])) == spaces[0]
    assert dict(a=dict(y=None)) == spaces[1]


def test_operators():
    s1 = Space(a=1, b=Grid(2, 3))
    s2 = Space(c=Grid("a", "b"))
    assert [
        dict(a=1, b=2, c="a"),
        dict(a=1, b=2, c="b"),
        dict(a=1, b=3, c="a"),
        dict(a=1, b=3, c="b"),
    ] == list(s1 * s2)

    assert [
        dict(a=1, b=2),
        dict(a=1, b=3),
        dict(c="a"),
        dict(c="b"),
    ] == list(s1 + s2)

    assert [
        dict(a=1, b=2, c="a"),
        dict(a=1, b=3, c="a"),
        dict(a=1, b=2, c="b"),
        dict(a=1, b=3, c="b"),
    ] == list(s1 * [dict(c="a"), dict(c="b")])

    assert [
        dict(a=1, b=2),
        dict(a=1, b=3),
        dict(c="a"),
        dict(c="b"),
    ] == list(s1 + [dict(c="a"), dict(c="b")])


def test_encode_decode():
    s1 = Space(
        a=Grid(1, 2),
        b=Rand(0, 1.0, 0.2, log=True, normal=False),
        c=Choice(1, 2, 3),
        d=[Grid(1, 2), Rand(0, 2.0)],
        e={"x": "xx", "y": Choice("a", "b")},
        f=RandInt(0, 10, log=False, normal=True),
    )
    actual = [decode(x) for x in s1.encode()]
    assert list(s1) == actual
    for x in s1.encode():
        print(json.dumps(x, indent=2))
