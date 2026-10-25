"""Only serialized data is decoded, and decoding never imports anything new.

A live value an agent handled is recorded as it is, even when it looks like one of
pyTracer's tags; a recording only rebuilds classes from packages the process has
already imported. Both close ways for data an end user or a client controls to
change what an agent sees or make the process import a module.
"""

from __future__ import annotations

import json
import re
import sys

import pytest
from pytracer_sdk import Resource, SpanKind, Tracer
from pytracer_sdk.codec import decode_value
from pytracer_sdk.schema import SERIALIZED, STORE_ONLY, Span
from pytracer_sdk.transport import InMemoryTransport

# Shaped exactly like an encoded enum (re.IGNORECASE), but it is an end user's data.
TAG_SHAPED = {"__pytracer__": "enum", "v": 2, "cls": "re:RegexFlag"}


def test_a_tag_shaped_live_value_is_recorded_as_the_same_dict() -> None:
    sink = InMemoryTransport()
    tracer = Tracer(sink, resource=Resource(service_name="t", branch="main"))

    @tracer.op(kind=SpanKind.TOOL)
    def echo(payload: dict[str, object]) -> dict[str, object]:
        return payload

    assert echo(dict(TAG_SHAPED)) == TAG_SHAPED
    tracer.flush()
    (span,) = sink.spans
    assert span.input == {"args": [TAG_SHAPED], "kwargs": {}}
    assert type(span.output) is dict and span.output == TAG_SHAPED

    # The same dict comes back from a recording (JSONL) and from a store row.
    assert Span.from_json(span.to_json()).output == TAG_SHAPED
    row = json.loads(span.model_dump_json())
    assert Span.model_validate(row, context=SERIALIZED).output == TAG_SHAPED


def test_assigning_a_tag_shaped_value_keeps_it() -> None:
    span = Span(trace_id="a" * 32, span_id="b" * 16, name="t", start_time_ns=1)
    span.output = dict(TAG_SHAPED)
    assert span.output == TAG_SHAPED


def test_a_recording_never_imports_a_package_the_process_has_not() -> None:
    sys.modules.pop("this", None)  # importing it prints; it is never imported here
    recorded = {"__pytracer__": "enum", "v": "a", "cls": "this:Anything"}
    assert decode_value(recorded) == "a"  # plain data
    assert Span.from_json(
        json.dumps({"trace_id": "a" * 32, "span_id": "b" * 16, "name": "t",
                    "start_time_ns": 1, "output": recorded})
    ).output == "a"
    assert "this" not in sys.modules


def test_a_class_from_an_imported_package_is_still_rebuilt() -> None:
    # A recording (serialized data) of the same shape is an enum, rebuilt.
    assert decode_value(dict(TAG_SHAPED)) is re.IGNORECASE


def test_store_only_and_serialized_contexts_are_read_only() -> None:
    for context in (STORE_ONLY, SERIALIZED):
        with pytest.raises(TypeError):
            context["pytracer_payload"] = "other"  # type: ignore[index]
