"""A self-signed TLS certificate for serving the dashboard over HTTPS.

Android and iOS only grant camera access on a secure origin
(``https://…``), so the development server needs TLS between the PC and the
phone. ``runsecure`` loads the pair in ``certs/`` and regenerates it
automatically when it is missing, expired, or no longer covers the machine's
LAN addresses in its Subject Alternative Names.

A self-signed certificate gives no trust: the browser shows a warning the
user accepts explicitly (iOS: "Show Details" → "visit this website"; Android
Chrome: "Advanced" → "Proceed"). That one acceptance replaces the whole
Chrome-flag ritual and is the only way camera access works on iOS at all.
"""

import datetime
import ipaddress
import socket
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

# iOS rejects TLS certificates valid for longer than 825 days; stay below it.
DEFAULT_VALIDITY_DAYS = 820
KEY_SIZE = 2048
CERTIFICATE_NAME = "selfsigned"


def certificate_paths(base_dir: Path) -> tuple[Path, Path]:
    """Where `runsecure` naturally stores its pair: <base_dir>/certs/."""
    directory = Path(base_dir) / "certs"
    return directory / f"{CERTIFICATE_NAME}.pem", directory / f"{CERTIFICATE_NAME}.key"


def device_hosts() -> list[str]:
    """Every name under which this PC can be reached, TLS SAN candidates.

    DNS names first (localhost, the machine's own hostname), then IPv4
    addresses (loopback plus the LAN addresses); IPv6 loopback is typical only
    for PC-side browsing, but costs nothing with a SAN of three entries.
    """
    hosts = ["localhost"]
    try:
        hostname = socket.gethostname()
        if hostname:
            hosts.append(hostname)
    except OSError:
        pass
    hosts.extend(["127.0.0.1", "::1", *connect_lan_ips()])
    # Deterministic order so a comparison against a written certificate is stable.
    return sorted(set(hosts))


def connect_lan_ips() -> list[str]:
    """Imported lazily to avoid a module-level dependency loop with connect.py."""
    from simulation.connect import detect_lan_ips

    return detect_lan_ips()


def certificate_hosts(cert_path: Path) -> list[str]:
    """Read the SAN entries of ``cert_path``; raises ``ValueError`` on garbage."""
    cert_path = Path(cert_path)
    with cert_path.open("rb") as handle:
        certificate = x509.load_pem_x509_certificate(handle.read())
    return [str(san.value) for san in certificate.extensions.get_extension_for_class(x509.SubjectAlternativeName).value]


def _san_entry(host: str) -> x509.GeneralName:
    """Map one SAN host onto its GeneralName class."""
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return x509.DNSName(host)
    return x509.IPAddress(address)


def generate_self_signed(
    cert_path: Path,
    key_path: Path,
    *,
    days: int = DEFAULT_VALIDITY_DAYS,
    key_size: int | None = None,
) -> Path:
    """Write one self-signed key + certificate pair covering ``device_hosts()``.

    Both parent directories are created; an existing pair is overwritten. The
    key is unencrypted PKCS#8 PEM, the certificate unencrypted PEM — exactly
    what Twisted's SSL endpoint consumes.
    """
    if days < 1:
        raise ValueError("Certificate validity in days must be positive.")
    if key_size is None:
        # Bound at call time so tests can thin the key out for speed.
        key_size = KEY_SIZE
    hosts = sorted({host for host in device_hosts()})
    key = rsa.generate_private_key(public_exponent=65537, key_size=key_size)
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, "SelfDriveSim (local)"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "SelfDriveSim"),
    ])
    now = datetime.datetime.now(datetime.timezone.utc)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(key.public_key())
        # A random serial number, per RFC 5280; also lets Chrome tell two
        # installed copies apart.
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=days))
        .add_extension(
            x509.BasicConstraints(ca=False, path_length=None),
            critical=True,
        )
        .add_extension(x509.SubjectAlternativeName([_san_entry(h) for h in hosts]), critical=False)
        .add_extension(x509.KeyUsage(
            digital_signature=True,
            key_encipherment=True,
            content_commitment=False,
            data_encipherment=False,
            key_agreement=False,
            key_cert_sign=False,
            crl_sign=False,
            encipher_only=False,
            decipher_only=False,
        ), critical=True)
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
        .sign(key, hashes.SHA256())
    )

    cert_path = Path(cert_path)
    key_path = Path(key_path)
    cert_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    cert_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    return cert_path


def certificate_is_expired(cert_path: Path) -> bool:
    """Whether the certificate is within one day of expiring."""
    with Path(cert_path).open("rb") as handle:
        certificate = x509.load_pem_x509_certificate(handle.read())
    valid_until = certificate.not_valid_after_utc
    return valid_until <= datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=1)


def ensure_self_signed(
    base_dir: Path,
    *,
    days: int = DEFAULT_VALIDITY_DAYS,
    force: bool = False,
) -> dict:
    """Return the paths of a usable pair, regenerating one if needed.

    Regeneration happens when ``force`` is set, either file is missing, the
    pair is near expiry, the current ``device_hosts()`` differ from the ones
    written into the certificate, or the existing files cannot be parsed.
    Includes a ``regenerated`` flag and (for the console) the covered hosts.
    """
    cert_path, key_path = certificate_paths(base_dir)
    reasons: list[str] = []

    if force:
        reasons.append("forced")
    if not (cert_path.exists() and key_path.exists()):
        reasons.append("missing files")
    else:
        try:
            existing = certificate_hosts(cert_path)
            hosts = device_hosts()
            if sorted(existing) != sorted(hosts):
                reasons.append("SAN no longer matches this machine's addresses")
            if certificate_is_expired(cert_path):
                reasons.append("expired or about to expire")
        except (OSError, ValueError, TypeError, AttributeError):
            reasons.append("unparseable")

    if reasons:
        generate_self_signed(cert_path, key_path, days=days)
    return {
        "cert_path": cert_path,
        "key_path": key_path,
        "regenerated": bool(reasons),
        "regeneration_reasons": reasons,
        "hosts": device_hosts(),
        "valid_days": days,
    }
