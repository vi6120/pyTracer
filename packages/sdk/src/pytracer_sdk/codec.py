"""Type-preserving JSON encoding for recorded span inputs and outputs.

A span's ``input`` and ``output`` hold the real Python values an op received and
returned. Plain JSON cannot carry many of them: a ``Decimal`` price, a ``datetime``,
a ``UUID`` or ``bytes`` would come back as a string, a ``tuple`` or ``set`` as a
list. A replay that served those back would hand the agent a different type than
it got live, so ``price * qty`` on a ``"19.99"`` string silently repeats the text
instead of multiplying.

:func:`encode_value` turns such values into tagged JSON objects
(``{"__pytracer__": "decimal", "v": "19.99"}``) and :func:`decode_value` turns
them back. Plain JSON values (strings, numbers, booleans, null, string-keyed dicts,
lists) are left exactly as they are, so a recording of plain data looks the same as
before. A class-backed value (an enum, a pydantic model, a dataclass) is rebuilt
from its importable class; when the class cannot be imported it falls back to its
plain data.

A recording names the class to rebuild, so decoding is restricted to what the
tag promises: an ``enum`` tag only rebuilds an ``Enum``, ``path`` a ``PurePath``,
``model`` a pydantic model, ``dataclass`` a dataclass. Any other class falls back
to plain data, so a crafted recording cannot instantiate an arbitrary class (such
as ``subprocess.Popen``). A process that only stores spans, like the ingestion
gateway, decodes with ``rebuild_classes=False``: it imports nothing and keeps
class-backed values as their tagged JSON, which it stores unchanged.

The key ``__pytracer__`` is reserved: an object consisting of exactly that key
plus ``v`` (and optionally ``cls``) is read as a tagged value, and is written back
as it is. Any other dict that uses the key is encoded unambiguously and kept as
data.
"""

from __future__ import annotations

import base64
import dataclasses
import datetime as dt
import enum
import importlib
import pathlib
import uuid
from decimal import Decimal
from typing import Any

TAG = "__pytracer__"
_TAGGED_KEYS = frozenset({TAG, "v", "cls"})


def encode_value(value: Any) -> Any:
    """Return a JSON-safe form of ``value`` that :func:`decode_value` can restore."""
    # Enums first: IntEnum / StrEnum are also int / str.
    if isinstance(value, enum.Enum):
        return _tag("enum", encode_value(value.value), cls=_class_path(value))
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        if _is_tagged(value):
            # Already encoded (a class-backed value a store-only process kept as
            # JSON): write it back unchanged.
            return value
        if all(isinstance(k, str) for k in value) and TAG not in value:
            return {k: encode_value(v) for k, v in value.items()}
        # Non-string keys (or a key that would read as our tag) keep their exact
        # keys as pairs, since JSON object keys are always strings.
        return _tag("dict", [[encode_value(k), encode_value(v)] for k, v in value.items()])
    if isinstance(value, list):
        return [encode_value(v) for v in value]
    if isinstance(value, tuple):
        return _tag("tuple", [encode_value(v) for v in value])
    if isinstance(value, (set, frozenset)):
        name = "frozenset" if isinstance(value, frozenset) else "set"
        return _tag(name, [encode_value(v) for v in _stable_order(value)])
    if isinstance(value, Decimal):
        return _tag("decimal", str(value))
    # datetime before date: datetime is a date subclass.
    if isinstance(value, dt.datetime):
        return _tag("datetime", value.isoformat())
    if isinstance(value, dt.date):
        return _tag("date", value.isoformat())
    if isinstance(value, dt.time):
        return _tag("time", value.isoformat())
    if isinstance(value, dt.timedelta):
        return _tag("timedelta", [value.days, value.seconds, value.microseconds])
    if isinstance(value, uuid.UUID):
        return _tag("uuid", str(value))
    if isinstance(value, bytearray):
        return _tag("bytearray", base64.b64encode(bytes(value)).decode("ascii"))
    if isinstance(value, bytes):
        return _tag("bytes", base64.b64encode(value).decode("ascii"))
    if isinstance(value, pathlib.PurePath):
        return _tag("path", str(value), cls=_class_path(value))
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        fields = {f.name: encode_value(getattr(value, f.name)) for f in dataclasses.fields(value)}
        return _tag("dataclass", fields, cls=_class_path(value))
    dump = getattr(value, "model_dump", None)
    if callable(dump) and hasattr(type(value), "model_validate"):
        # Only the fields that were set: rebuilding with model_validate then restores
        # the same values *and* the same fields-set, so code that dumps the object
        # with exclude_unset (as agent SDKs do when building the next request) sees
        # exactly what it saw live.
        return _tag(
            "model", encode_value(dump(mode="python", exclude_unset=True)), cls=_class_path(value)
        )
    # Anything else is left to the caller's serializer, as before.
    return value


def decode_value(value: Any, *, rebuild_classes: bool = True) -> Any:
    """Restore values produced by :func:`encode_value`; other values pass through.

    Containers are only rebuilt when something inside them was decoded, so live
    (never serialized) values keep their identity. With ``rebuild_classes=False``
    no module is imported and class-backed values (enum, path, model, dataclass)
    stay as their tagged JSON; use it where the data is only stored.
    """
    if isinstance(value, dict):
        if _is_tagged(value):
            return _decode_tagged(value[TAG], value, rebuild_classes)
        decoded = {k: decode_value(v, rebuild_classes=rebuild_classes) for k, v in value.items()}
        return value if all(decoded[k] is value[k] for k in value) else decoded
    if isinstance(value, list):
        items = [decode_value(v, rebuild_classes=rebuild_classes) for v in value]
        return value if all(a is b for a, b in zip(items, value)) else items
    return value


def _is_tagged(value: dict[Any, Any]) -> bool:
    return isinstance(value.get(TAG), str) and set(value) <= _TAGGED_KEYS


# Tags whose value is rebuilt from a class the recording names, and the kind of
# class each may rebuild.
_CLASS_TAGS = frozenset({"path", "enum", "model", "dataclass"})


def _decode_tagged(tag: str, value: dict[str, Any], rebuild_classes: bool) -> Any:
    if tag in _CLASS_TAGS and not rebuild_classes:
        return value  # stored as written; nothing is imported
    # Untyped JSON payload; a malformed one falls through to the except below.
    raw: Any = value.get("v")

    def inner(v: Any) -> Any:
        return decode_value(v, rebuild_classes=rebuild_classes)

    try:
        if tag == "dict":
            return {inner(k): inner(v) for k, v in raw}
        if tag == "tuple":
            return tuple(inner(v) for v in raw)
        if tag == "set":
            return {inner(v) for v in raw}
        if tag == "frozenset":
            return frozenset(inner(v) for v in raw)
        if tag == "decimal":
            return Decimal(raw)
        if tag == "datetime":
            return dt.datetime.fromisoformat(raw)
        if tag == "date":
            return dt.date.fromisoformat(raw)
        if tag == "time":
            return dt.time.fromisoformat(raw)
        if tag == "timedelta":
            days, seconds, micros = raw
            return dt.timedelta(days=days, seconds=seconds, microseconds=micros)
        if tag == "uuid":
            return uuid.UUID(raw)
        if tag == "bytes":
            return base64.b64decode(raw)
        if tag == "bytearray":
            return bytearray(base64.b64decode(raw))
        if tag == "path":
            cls = _resolve_class(value.get("cls"), pathlib.PurePath)
            return cls(str(raw)) if cls is not None else pathlib.PurePath(str(raw))
        if tag == "enum":
            cls = _resolve_class(value.get("cls"), enum.Enum)
            data = inner(raw)
            return cls(data) if cls is not None else data
        if tag == "model":
            cls = _resolve_class(value.get("cls"), _pydantic_base())
            data = inner(raw)
            return cls.model_validate(data) if cls is not None else data
        if tag == "dataclass":
            cls = _resolve_class(value.get("cls"), None)
            data = {k: inner(v) for k, v in raw.items()}
            if cls is not None and dataclasses.is_dataclass(cls):
                return _build_dataclass(cls, data)
            return data
    except Exception:  # noqa: BLE001 - a value that will not rebuild falls back, never crashes a load
        return inner(raw)
    # An unknown tag (e.g. written by a newer SDK) is kept as-is rather than lost.
    return value


def _tag(name: str, payload: Any, *, cls: str | None = None) -> dict[str, Any]:
    tagged: dict[str, Any] = {TAG: name, "v": payload}
    if cls is not None:
        tagged["cls"] = cls
    return tagged


def _stable_order(values: set[Any] | frozenset[Any]) -> list[Any]:
    """Sort set members when possible so the same set always encodes the same way."""
    try:
        return sorted(values)
    except TypeError:
        return sorted(values, key=repr)


def _class_path(obj: Any) -> str:
    cls = type(obj)
    return f"{cls.__module__}:{cls.__qualname__}"


def _resolve_class(path: Any, required_base: type[Any] | None) -> type[Any] | None:
    """Import ``module:Qual.Name``; None when it cannot be resolved, or when it is
    not a subclass of ``required_base`` (the kind of class the tag promises)."""
    if not isinstance(path, str) or ":" not in path:
        return None
    module_name, qualname = path.split(":", 1)
    if "<locals>" in qualname:
        return None
    try:
        obj: Any = importlib.import_module(module_name)
        for part in qualname.split("."):
            obj = getattr(obj, part)
    except Exception:  # noqa: BLE001 - not importable here: fall back to plain data
        return None
    if not isinstance(obj, type):
        return None
    if required_base is not None and not issubclass(obj, required_base):
        return None
    return obj


def _pydantic_base() -> type[Any]:
    from pydantic import BaseModel

    return BaseModel


def _build_dataclass(cls: type[Any], data: dict[str, Any]) -> Any:
    """Construct a dataclass from its fields, even when some are ``init=False``."""
    try:
        return cls(**data)
    except TypeError:
        instance = object.__new__(cls)
        for name, val in data.items():
            object.__setattr__(instance, name, val)
        return instance
