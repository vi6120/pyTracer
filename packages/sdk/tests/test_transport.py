"""HTTPTransport: URL handling and TLS context construction.

The full request path (SDK -> gateway -> store) is covered by the gateway's live
e2e tests; here we cover the transport's own TLS/verification behavior, which is
what the security baseline relies on.
"""

from __future__ import annotations

import ssl
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from pytracer_sdk.transport import HTTPTransport, _build_ssl_context


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
