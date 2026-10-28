"""Tests for the self-signed certificate pair behind runsecure's TLS port."""

from pathlib import Path

import pytest
from cryptography import x509
from cryptography.hazmat.primitives.serialization import load_pem_private_key

from simulation import selfsigned


@pytest.fixture(autouse=True)
def small_fast_key(monkeypatch):
    """1024-bit keys shave the generation time; fine for RSA round-trip tests."""

    monkeypatch.setattr(selfsigned, "KEY_SIZE", 1024)


@pytest.fixture()
def tmp_certs(tmp_path, monkeypatch):
    """The pair lands in a temporary project dir, not the repo."""
    monkeypatch.setattr(selfsigned, "certificate_paths", lambda base_dir: (
        tmp_path / "certs" / "selfsigned.pem",
        tmp_path / "certs" / "selfsigned.key",
    ))
    monkeypatch.setattr(selfsigned, "device_hosts", lambda: ["localhost", "127.0.0.1", "10.0.0.7"])
    return tmp_path


def test_generated_pair_is_a_valid_cert_and_key(tmp_certs):
    cert_path, key_path = selfsigned.certificate_paths(tmp_certs)

    assert selfsigned.generate_self_signed(cert_path, key_path).exists()
    assert cert_path.read_bytes().startswith(b"-----BEGIN CERTIFICATE-----")
    certificate = x509.load_pem_x509_certificate(cert_path.read_bytes())
    assert certificate.subject.get_attributes_for_oid(x509.NameOID.ORGANIZATION_NAME)[0].value == "SelfDriveSim"
    # PKCS#8, the format Twisted's SSL endpoint consumes.
    key = load_pem_private_key(key_path.read_bytes(), password=None)
    assert key.key_size == 1024


def test_certificate_covers_the_declared_hosts(tmp_certs):
    cert_path, key_path = selfsigned.certificate_paths(tmp_certs)

    selfsigned.generate_self_signed(cert_path, key_path)
    hosts = selfsigned.certificate_hosts(cert_path)

    assert hosts == ["10.0.0.7", "127.0.0.1", "localhost"]


def test_certificate_validity_is_ios_compatible(tmp_certs):
    import datetime

    cert_path, key_path = selfsigned.certificate_paths(tmp_certs)

    selfsigned.generate_self_signed(cert_path, key_path, days=30)
    certificate = x509.load_pem_x509_certificate(cert_path.read_bytes())
    span = certificate.not_valid_after_utc - certificate.not_valid_before_utc

    assert datetime.timedelta(days=31) <= span <= datetime.timedelta(days=32)


def test_ensure_reuses_the_pair_when_it_still_matches(tmp_certs):
    cert_path, key_path = selfsigned.certificate_paths(tmp_certs)
    first = selfsigned.ensure_self_signed(tmp_certs)

    assert first["regenerated"] is True
    before = cert_path.read_bytes()

    second = selfsigned.ensure_self_signed(tmp_certs)

    assert second["regenerated"] is False
    assert second["regeneration_reasons"] == []
    assert cert_path.read_bytes() == before, "reuse must not touch the existing files"


def test_ensure_regenerates_when_the_machine_addresses_change(tmp_certs, monkeypatch):
    selfsigned.ensure_self_signed(tmp_certs)

    # The PC joined another network: the SANs no longer match reality.
    monkeypatch.setattr(
        selfsigned, "device_hosts", lambda: ["localhost", "127.0.0.1", "99.0.0.2"]
    )
    regenerated = selfsigned.ensure_self_signed(tmp_certs)

    assert regenerated["regenerated"] is True
    hosts = selfsigned.certificate_hosts(regenerated["cert_path"])
    assert hosts == ["127.0.0.1", "99.0.0.2", "localhost"]


def test_ensure_regenerates_an_unparseable_certificate(tmp_certs):
    cert_path, key_path = selfsigned.certificate_paths(tmp_certs)
    first = selfsigned.ensure_self_signed(tmp_certs)
    cert_path.write_bytes(b"not a certificate")
    key_path.write_bytes(b"not a key")

    second = selfsigned.ensure_self_signed(tmp_certs)

    assert second["regenerated"] is True
    selfsigned.certificate_hosts(second["cert_path"])  # readable again


def test_invalid_certificate_files_are_refused_cleanly(tmp_certs):
    cert_path, key_path = selfsigned.certificate_paths(tmp_certs)
    cert_path.parent.mkdir(parents=True)
    cert_path.write_bytes(b"garbage")

    with pytest.raises((OSError, ValueError, TypeError, AttributeError, PermissionError)):
        selfsigned.certificate_hosts(cert_path)


def test_certificate_paths_live_in_certs_by_default():
    base = Path("some/project")
    cert_path, key_path = selfsigned.certificate_paths(base)

    assert cert_path.name == "selfsigned.pem"
    assert key_path.name == "selfsigned.key"
    assert cert_path.parent.name == "certs"
