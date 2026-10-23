"""HTTPTransport: URL handling and TLS context construction.

The full request path (SDK -> gateway -> store) is covered by the gateway's live
e2e tests; here we cover the transport's own TLS/verification behavior, which is
what the security baseline relies on.
"""

from __future__ import annotations

import json
import ssl
import subprocess
import sys
import textwrap
import urllib.request
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError, create_model
from pydantic_core import PydanticUndefined

from pytracer_sdk import new_span_id, new_trace_id
from pytracer_sdk.schema import Span, SpanStatus
from pytracer_sdk.transport import HTTPTransport, _build_ssl_context


def _previous_gateway_span_model() -> type[BaseModel]:
    """A stand-in for the released 0.1.2 gateway's span contract.

    Rebuilt from the current ``Span`` minus ``error_module`` (the field this line
    added) and with ``extra="forbid"``, which is what the deployed gateway used
    to reject every span carrying the new field.
    """
    fields: dict[str, Any] = {}
    for name, info in Span.model_fields.items():
        if name == "error_module":
            continue
        if info.default_factory is not None:
            default: Any = info.default_factory()  # type: ignore[call-arg]
        elif info.default is not PydanticUndefined:
            default = info.default
        else:
            default = ...
        fields[name] = (info.annotation, default)
    return create_model("PreviousGatewaySpan", __config__=ConfigDict(extra="forbid"), **fields)


def test_spans_are_delivered_on_normal_exit_without_flush(tmp_path: Path) -> None:
    # A script that records a span and exits WITHOUT calling flush()/shutdown()
    # must still deliver the span, via the atexit hook, or short scripts lose
    # their traces silently (the sender is a daemon thread).
    out = tmp_path / "spans.txt"
    script = textwrap.dedent(
        """
        import pathlib, sys
        import pytracer_sdk as pytracer

        out = pathlib.Path(sys.argv[1])

        class FileTransport:
            def send(self, spans):
                with out.open("a", encoding="utf-8") as fh:
                    for span in spans:
                        print(span.span_id, file=fh)

        pytracer.configure(FileTransport(), service_name="exit-test")
        with pytracer.start_span("on-exit"):
            pass
        # deliberately no flush()/shutdown(): the atexit hook must deliver it
        """
    )
    subprocess.run([sys.executable, "-c", script, str(out)], check=True, timeout=30)
    delivered = [line for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(delivered) == 1


def test_wire_payload_drops_unset_optionals_but_keeps_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # V1: the new SDK must not trip a gateway on an older release by sending a
    # field it predates. A span with no error ships without error_module at all;
    # a failed span still carries its real exception module so replay rebuilds it.
    captured: dict[str, Any] = {}

    class _Resp:
        status = 202

        def __enter__(self) -> _Resp:
            return self

        def __exit__(self, *exc: object) -> bool:
            return False

        def read(self) -> bytes:
            return b""

    def _fake_urlopen(request: Any, *args: Any, **kwargs: Any) -> _Resp:
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return _Resp()

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)

    ok = Span(trace_id=new_trace_id(), span_id=new_span_id(), name="ok", start_time_ns=1)
    HTTPTransport("https://gw.example", "k").send([ok])
    sent = captured["body"]["spans"][0]
    assert "error_module" not in sent
    assert "error_type" not in sent

    failed = Span(
        trace_id=new_trace_id(),
        span_id=new_span_id(),
        name="boom",
        start_time_ns=1,
        status=SpanStatus.ERROR,
        error_type="ServiceUnavailable",
        error_module="neo4j.exceptions",
        error_message="db down",
    )
    HTTPTransport("https://gw.example", "k").send([failed])
    sent = captured["body"]["spans"][0]
    assert sent["error_module"] == "neo4j.exceptions"


def test_new_sdk_payload_against_a_previous_gateway_release() -> None:
    # V1 regression (new SDK -> previous gateway release). The 0.1.2 gateway
    # forbids unknown fields and predates error_module, so:
    #   - a no-error span, which now omits error_module, is accepted;
    #   - a failed span still carries error_module and is rejected, which is why
    #     the gateway must be upgraded before the SDKs that send to it.
    previous = _previous_gateway_span_model()

    ok = Span(trace_id=new_trace_id(), span_id=new_span_id(), name="ok", start_time_ns=1)
    ok_wire = ok.model_dump(mode="json", exclude_none=True)
    assert "error_module" not in ok_wire
    previous.model_validate(ok_wire)  # accepted: no field the old gateway did not know

    failed = Span(
        trace_id=new_trace_id(),
        span_id=new_span_id(),
        name="boom",
        start_time_ns=1,
        status=SpanStatus.ERROR,
        error_type="ServiceUnavailable",
        error_module="neo4j.exceptions",
        error_message="db down",
    )
    failed_wire = failed.model_dump(mode="json", exclude_none=True)
    assert failed_wire["error_module"] == "neo4j.exceptions"
    with pytest.raises(ValidationError):
        previous.model_validate(failed_wire)  # upgrade the gateway first


def test_rejections_in_a_202_are_surfaced(caplog: pytest.LogCaptureFixture) -> None:
    # A 202 that still rejected spans must be reported, not counted as fully stored
    # (G1/G5): a running total, a logged warning, and the on_rejection callback.
    seen: list[tuple[int, list]] = []
    t = HTTPTransport("https://gw.example", "k", on_rejection=lambda n, errs: seen.append((n, errs)))
    with caplog.at_level("WARNING", logger="pytracer_sdk"):
        t._note_rejections(b'{"accepted": 3, "rejected": 2, "errors": ["missing branch/commit"]}')
    assert t.rejected_spans == 2
    assert seen == [(2, ["missing branch/commit"])]
    assert "rejected 2 span(s)" in caplog.text


def test_a_clean_202_reports_no_rejections() -> None:
    t = HTTPTransport("https://gw.example", "k")
    t._note_rejections(b'{"accepted": 5, "rejected": 0}')
    assert t.rejected_spans == 0


def test_note_rejections_tolerates_empty_or_bad_body() -> None:
    t = HTTPTransport("https://gw.example", "k")
    t._note_rejections(b"")
    t._note_rejections(b"not json")
    assert t.rejected_spans == 0


def test_base_url_gets_the_endpoint_path() -> None:
    assert HTTPTransport("https://gw.example", "k").url == "https://gw.example/v1/spans"
    assert HTTPTransport("https://gw.example/", "k").url == "https://gw.example/v1/spans"
    # An explicit endpoint is left as-is.
    assert HTTPTransport("https://gw.example/v1/spans", "k").url == "https://gw.example/v1/spans"


def test_default_context_verifies_certs_and_hostname() -> None:
    ctx = _build_ssl_context(verify=True, ca_bundle=None)
    assert ctx.verify_mode == ssl.CERT_REQUIRED
    assert ctx.check_hostname is True


def test_verify_false_disables_verification() -> None:
    ctx = _build_ssl_context(verify=False, ca_bundle=None)
    assert ctx.verify_mode == ssl.CERT_NONE
    assert ctx.check_hostname is False


def test_transport_stores_its_context() -> None:
    verified = HTTPTransport("https://gw.example", "k")
    assert verified._ssl_context.verify_mode == ssl.CERT_REQUIRED
    unverified = HTTPTransport("https://gw.example", "k", verify=False)
    assert unverified._ssl_context.verify_mode == ssl.CERT_NONE


def test_missing_ca_bundle_is_an_error() -> None:
    # A bad CA path must fail loudly rather than silently falling back.
    with pytest.raises((FileNotFoundError, ssl.SSLError, OSError)):
        _build_ssl_context(verify=True, ca_bundle="/no/such/ca-bundle.pem")
