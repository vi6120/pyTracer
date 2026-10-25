"""Recorded span inputs and outputs keep their Python types through JSON."""

from __future__ import annotations

import dataclasses
import datetime as dt
import enum
import json
import pathlib
import uuid
from decimal import Decimal
from typing import Any

import pytest
from pydantic import BaseModel

from pytracer_sdk import new_span_id, new_trace_id
from pytracer_sdk.codec import TAG, EncodedPayload, decode_value, encode_value
from pytracer_sdk.schema import SERIALIZED, STORE_ONLY, Span


class Color(enum.Enum):
    RED = "red"


class Level(enum.IntEnum):
    HIGH = 3


class Item(BaseModel):
    name: str
    price: Decimal


@dataclasses.dataclass
class Point:
    x: int
    when: dt.date


SAMPLES: dict[str, Any] = {
    "decimal": Decimal("19.99"),
    "datetime": dt.datetime(2026, 10, 25, 9, 30, 15, 123456),
    "aware datetime": dt.datetime(2026, 10, 25, 9, 30, tzinfo=dt.timezone.utc),
    "date": dt.date(2026, 10, 25),
    "time": dt.time(9, 30, 15),
    "timedelta": dt.timedelta(days=1, seconds=5, microseconds=7),
    "uuid": uuid.UUID(int=1),
    "bytes": b"\x00\xffab",
    "bytearray": bytearray(b"ab"),
    "tuple": (1, "a", Decimal("2.5")),
    "set": {1, 2, 3},
    "frozenset": frozenset({"a", "b"}),
    "int-keyed dict": {1: "one", 2: "two"},
    "enum": Color.RED,
    "int enum": Level.HIGH,
    "pydantic model": Item(name="book", price=Decimal("9.50")),
    "dataclass": Point(x=1, when=dt.date(2026, 1, 1)),
    "path": pathlib.PurePosixPath("/tmp/report.csv"),
    "nested": {"items": [{"price": Decimal("1.10"), "at": dt.date(2026, 1, 2)}], "ids": (1, 2)},
}


def _span(output: Any) -> Span:
    return Span(trace_id=new_trace_id(), span_id=new_span_id(), name="t", start_time_ns=1, output=output)


def _same(a: Any, b: Any) -> bool:
    return type(a) is type(b) and a == b


@pytest.mark.parametrize("name", list(SAMPLES))
def test_recording_round_trip_keeps_the_type(name: str) -> None:
    # The JSONL path (`pytracer record`, `record --from-file`, registry bundles).
    value = SAMPLES[name]
    back = Span.from_json(_span({"v": value}).to_json()).output["v"]
    assert _same(back, value), f"{name}: got {back!r} ({type(back).__name__})"


@pytest.mark.parametrize("name", list(SAMPLES))
def test_gateway_wire_round_trip_keeps_the_type(name: str) -> None:
    # The HTTP transport sends model_dump(mode="json"); the gateway keeps it as
    # received and the store reads it back as serialized data.
    value = SAMPLES[name]
    wire = json.loads(json.dumps(_span({"v": value}).model_dump(mode="json")))
    stored = json.loads(Span.model_validate(wire, context=STORE_ONLY).model_dump_json())
    assert stored == wire
    assert _same(Span.model_validate(stored, context=SERIALIZED).output["v"], value)


def test_nested_values_keep_their_types() -> None:
    back = Span.from_json(_span(SAMPLES["nested"]).to_json()).output
    assert back == SAMPLES["nested"]
    assert type(back["items"][0]["price"]) is Decimal
    assert type(back["ids"]) is tuple


def test_the_tester_case_price_times_qty() -> None:
    back = Span.from_json(_span({"price": Decimal("19.99"), "qty": 2}).to_json()).output
    assert back["price"] * back["qty"] == Decimal("39.98")


def test_plain_json_values_are_written_unchanged() -> None:
    # Recordings of plain data look exactly as they did before tagging existed.
    plain = {"q": "cats", "n": 2, "ok": True, "x": 1.5, "items": [1, "a", None], "m": {"k": "v"}}
    assert TAG not in _span(plain).to_json()
    assert Span.from_json(_span(plain).to_json()).output == plain


def test_a_user_dict_containing_the_reserved_key_round_trips_intact() -> None:
    # Only an object shaped exactly like a tag (the reserved key with "v" and an
    # optional "cls") is read as one; any other dict using the key is kept as data.
    tricky = {TAG: "decimal", "v": "1.0", "note": "user data"}
    assert Span.from_json(_span(tricky).to_json()).output == tricky


def test_an_unimportable_class_falls_back_to_its_plain_data() -> None:
    @dataclasses.dataclass
    class Local:  # defined in a function, so it cannot be imported back
        a: int

    back = Span.from_json(_span(Local(a=1)).to_json()).output
    assert back == {"a": 1}


def test_live_values_are_not_copied() -> None:
    live = {"price": Decimal("1.00"), "ids": [1, 2]}
    assert decode_value(live) is live


def test_encode_leaves_plain_values_alone() -> None:
    for value in ("s", 1, 1.5, True, None, [1, "a"], {"k": "v"}):
        assert encode_value(value) == value


class _Item(BaseModel):
    name: str
    caller: str | None = None
    note: str = "default"


def test_pydantic_model_keeps_which_fields_were_set() -> None:
    # An SDK that builds its next request with model_dump(exclude_unset=True) must
    # see the same dict from a replayed object as from the live one.
    live = _Item(name="weather")
    back = decode_value(json.loads(json.dumps(encode_value(live))))
    assert back == live
    assert back.model_fields_set == live.model_fields_set == {"name"}
    assert back.model_dump(exclude_unset=True) == {"name": "weather"}
    explicit = _Item(name="weather", caller=None)
    again = decode_value(json.loads(json.dumps(encode_value(explicit))))
    assert again.model_fields_set == {"name", "caller"}


# --- a recording cannot instantiate an arbitrary class ---------------------------

GADGET_CALLS: list[object] = []


class Gadget:
    """A class with a side effect in its constructor, as an attacker would pick."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        GADGET_CALLS.append((args, kwargs))


GADGET = f"{Gadget.__module__}:{Gadget.__qualname__}"


@pytest.mark.parametrize(
    "payload",
    [
        {"__pytracer__": "path", "v": ["rm", "-rf", "/"], "cls": GADGET},
        {"__pytracer__": "enum", "v": "x", "cls": GADGET},
        {"__pytracer__": "model", "v": {"a": 1}, "cls": GADGET},
        {"__pytracer__": "dataclass", "v": {"a": 1}, "cls": GADGET},
    ],
    ids=["path", "enum", "model", "dataclass"],
)
def test_a_tag_never_instantiates_a_class_of_the_wrong_kind(payload: dict[str, object]) -> None:
    GADGET_CALLS.clear()
    decoded = decode_value(json.loads(json.dumps(payload)))
    assert GADGET_CALLS == []  # the named class was not called
    assert not isinstance(decoded, Gadget)


def test_a_crafted_popen_payload_decodes_to_plain_data() -> None:
    payload = {"__pytracer__": "enum", "v": ["true"], "cls": "subprocess:Popen"}
    assert decode_value(payload) == ["true"]
    path = {"__pytracer__": "path", "v": ["true"], "cls": "subprocess:Popen"}
    assert isinstance(decode_value(path), pathlib.PurePath)


def test_store_only_keeps_payloads_as_received_and_writes_them_back() -> None:
    # A store-only process (the gateway) decodes nothing and stores the value as
    # the client wrote it; a replay machine later rebuilds it.
    tagged = encode_value({"item": _Item(name="weather"), "price": Decimal("1.5")})
    span = Span.model_validate(
        {**json.loads(_span(None).to_json()), "output": tagged}, context=STORE_ONLY
    )
    assert span.output == EncodedPayload(tagged)
    assert encode_value(span.output) == tagged  # written back unchanged, not re-escaped
    assert json.loads(span.model_dump_json())["output"] == tagged
    assert decode_value(tagged) == {"item": _Item(name="weather"), "price": Decimal("1.5")}
