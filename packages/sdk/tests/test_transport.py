"""HTTPTransport: URL handling and TLS context construction.

The full request path (SDK -> gateway -> store) is covered by the gateway's live
e2e tests; here we cover the transport's own TLS/verification behavior, which is
what the security baseline relies on.
"""

from __future__ import annotations

import ssl

import pytest

from pytracer_sdk.transport import HTTPTransport, _build_ssl_context


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
