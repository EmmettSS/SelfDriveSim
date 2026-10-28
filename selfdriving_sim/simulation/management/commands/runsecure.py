"""Development server with one extra HTTPS endpoint for the phone.

``python manage.py runsecure`` starts the same Daphne/ASGI stack as
``manage.py runserver`` (same arguments, autoreloader, static files) and
adds to it an ``ssl`` endpoint on ``--tls-port`` (default 8443) serving a
self-signed certificate from ``certs/``. The point: ``getUserMedia`` refuses
to run on plain http:// without per-browser flags, and on iOS it refuses
altogether; the HTTPS port is the one the dashboard QR code advertises.

The PC-side monitor can stay plain HTTP — the phone connects to the TLS
port, accepts the one-time certificate warning, and camera access works on
Android and iOS without any browser settings.
"""

import datetime
import logging
import os
import socket
import sys
from pathlib import Path

from django.conf import settings
from django.core.management import CommandError

from daphne import __version__ as daphne_version
from daphne.endpoints import build_endpoint_description_strings
from daphne.management.commands.runserver import Command as RunserverCommand

from simulation import connect, selfsigned

logger = logging.getLogger("django.channels.server")

TLS_PORT_DEFAULT = 8443


def tls_endpoint_string(port: int, addr: str, certfile: Path, keyfile: Path) -> str:
    """Twisted ``ssl`` endpoint description for the TLS listener.

    Endpoint strings split on ``:`` and ``=``, and Twisted's tokenizer consumes
    ``\\`` as an escape character — raw Windows backslash paths would be
    mangled. Paths are therefore rendered POSIX-style (``D:/a/b.key``) with
    the drive colon escaped the way daphne's own endpoint helper escapes
    IPv6 addresses.
    """

    def escape(value) -> str:
        return str(value).replace(":", r"\:")

    def escape_path(value) -> str:
        rendered = Path(value).resolve().as_posix()
        return rendered.replace(":", r"\:")

    return (
        f"ssl:port={int(port)}:interface={escape(addr)}"
        f":privateKey={escape_path(keyfile)}:certKey={escape_path(certfile)}"
    )


class Command(RunserverCommand):
    help = (
        "Daphne dev server on HTTP (like runserver) plus an HTTPS port using a "
        "self-signed certificate, so the phone camera works without browser flags."
    )

    def add_arguments(self, parser):
        super().add_arguments(parser)
        parser.add_argument(
            "--no-tls",
            action="store_true",
            help="Skip the HTTPS endpoint entirely (behaves like runserver).",
        )
        parser.add_argument(
            "--tls-port",
            type=int,
            default=None,
            metavar="PORT",
            help=(
                "HTTPS port (default: SIMULATION_TLS_PORT, "
                f"otherwise {TLS_PORT_DEFAULT})."
            ),
        )
        parser.add_argument(
            "--certfile",
            type=Path,
            default=None,
            metavar="FILE",
            help=(
                "TLS certificate PEM (needs --keyfile). Default: the generated "
                "self-signed pair in certs/."
            ),
        )
        parser.add_argument(
            "--keyfile",
            type=Path,
            default=None,
            metavar="FILE",
            help="TLS private key PEM (needs --certfile).",
        )
        parser.add_argument(
            "--tls-days",
            type=int,
            default=selfsigned.DEFAULT_VALIDITY_DAYS,
            metavar="DAYS",
            help=(
                "Validity in days of the auto-generated self-signed certificate "
                f"(default: {selfsigned.DEFAULT_VALIDITY_DAYS}, iOS-compatible)."
            ),
        )
        parser.add_argument(
            "--force-new-cert",
            action="store_true",
            help="Regenerate the self-signed pair even if it looks usable.",
        )

    def handle(self, *args, **options):
        self.no_tls = bool(options.get("no_tls"))
        self.tls_certfile = None
        self.tls_keyfile = None

        # A phone needs to reach this server over the network; runserver's own
        # default, 127.0.0.1, would never accept a connection from outside,
        # and a bare ``8000`` there means "127.0.0.1:8000". Both forms become
        # 0.0.0.0 unless the user spelled a full ``host:port`` pair.
        addrport = (options.get("addrport") or "").strip()
        if not addrport:
            options["addrport"] = "0.0.0.0:8000"
        elif ":" not in addrport:
            options["addrport"] = f"0.0.0.0:{addrport}"

        if not self.no_tls:
            tls_port = options.get("tls_port")
            if tls_port is None:
                try:
                    tls_port = int(os.environ.get(connect.TLS_PORT_ENV, TLS_PORT_DEFAULT))
                except ValueError:
                    tls_port = TLS_PORT_DEFAULT
            self.tls_port = self._pick_free_tls_port(int(tls_port))

            self.certfile, self.keyfile = self._resolve_tls_paths(options)
            self._prepare_certificate(options)

            # Advertised by the QR code; must survive into an autoreloader child.
            os.environ[connect.TLS_PORT_ENV] = str(self.tls_port)
        else:
            os.environ.pop(connect.TLS_PORT_ENV, None)

        super().handle(*args, **options)

    def _pick_free_tls_port(self, requested: int) -> int:
        """The first of ``requested .. requested+49`` that a TCP bind accepts.

        Resources can have the port reserved (Windows/Hyper-V dynamic port
        exclusions) or another process holds it; whether it is 8443 today does
        not generalize, so instead of hard-code-guessing the command probes
        and moves up. The end result stays consistent: the environment the
        views read, the printed URL and the QR code all name the same port.
        """
        skipped = []
        for port in range(requested, requested + 50):
            probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                probe.bind(("0.0.0.0", port))
            except OSError as error:
                skipped.append(f"{port} ({error.strerror})")
                continue
            finally:
                probe.close()
            if skipped:
                self.stdout.write(
                    f"==> TLS port {requested} is not bindable ({'; '.join(skipped)}); "
                    f"using {port} instead. Pass --tls-port to pin it."
                )
            return port
        raise CommandError(
            f"No free TLS port found in {requested}..{requested + 49}. "
            "Free one up or choose --tls-port / --no-tls."
        )

    def _resolve_tls_paths(self, options) -> tuple[Path, Path]:
        """The explicit pair, or the generated pair's natural location."""
        certfile = options.get("certfile")
        keyfile = options.get("keyfile")
        if certfile is not None and keyfile is None:
            raise CommandError("--certfile given without --keyfile.")
        if keyfile is not None and certfile is None:
            raise CommandError("--keyfile given without --certfile.")
        if certfile is not None:
            return Path(certfile), Path(keyfile)
        default_cert, default_key = selfsigned.certificate_paths(Path(settings.BASE_DIR))
        return default_cert, default_key

    def _prepare_certificate(self, options):
        """Create or refresh the self-signed pair, and remember the paths."""
        if self.certfile != selfsigned.certificate_paths(Path(settings.BASE_DIR))[0]:
            # An explicitly given pair is never generated; only validated.
            if not (self.certfile.exists() and self.keyfile.exists()):
                raise CommandError(
                    f"TLS pair missing: {self.certfile} / {self.keyfile}"
                )
            self.tls_certfile, self.tls_keyfile = self.certfile, self.keyfile
            self.stdout.write(
                f"==> HTTPS endpoint uses the given pair: {self.certfile}"
            )
            return

        info = selfsigned.ensure_self_signed(
            Path(settings.BASE_DIR),
            days=int(options.get("tls_days") or selfsigned.DEFAULT_VALIDITY_DAYS),
            force=bool(options.get("force_new_cert")),
        )
        self.tls_certfile = info["cert_path"]
        self.tls_keyfile = info["key_path"]
        if info["regenerated"]:
            self.stdout.write(
                f"==> Generated a self-signed certificate ({info['valid_days']} days): "
                f"{self.tls_certfile}"
            )
        else:
            self.stdout.write(
                f"==> Reusing the self-signed certificate: {self.tls_certfile}"
            )
        self.stdout.write(
            f"    Covers: {', '.join(info['hosts'])}\n"
            "    The phone browser shows a warning once: accept it (iOS: "
            "Show Details -> visit this website). Regenerate with "
            "--force-new-cert."
        )

    def _print_dashboard_info(self):
        """The two URLs, plus what the phone browser asks on the first visit."""
        dashboard = (
            f"https://{connect.detect_lan_ip()}:{self.tls_port}{connect.dashboard_path()}"
        )
        self.stdout.write(
            "\n"
            "    Monitor (this PC, plain HTTP):     http://localhost:%s/\n"
            "    Phone dashboard (scan the QR on    \n"
            "    the monitor page):                 %s\n"
            "\n"
            "    First phone visit: accept the self-signed certificate warning.\n"
            % (self.port, dashboard)
        )

    def inner_run(self, *args, **options):
        """daphne's runserver.inner_run, mirrored, with the TLS endpoint added."""
        if not options.get("use_asgi", True):
            return RunserverCommand.inner_run(self, *args, **options)
        self.stdout.write("Performing system checks...\n\n")
        self.check(display_num_errors=True)
        self.check_migrations()

        quit_command = "CTRL-BREAK" if sys.platform == "win32" else "CONTROL-C"
        now = datetime.datetime.now().strftime("%B %d, %Y - %X")
        self.stdout.write(now)
        self.stdout.write(
            "Django version %(version)s, using settings %(settings)r\n"
            "Starting ASGI/Daphne version %(daphne_version)s development server "
            "at %(protocol)s://%(addr)s:%(port)s/\n"
            f"Quit the server with {quit_command}.\n"
            % dict(
                version=self.get_version(),
                daphne_version=daphne_version,
                settings=settings.SETTINGS_MODULE,
                protocol="http",
                addr=f"[{self.addr}]" if self._raw_ipv6 else self.addr,
                port=self.port,
            )
        )
        if not self.no_tls:
            self.stdout.write(
                f"Serving HTTPS (for phone camera access) on "
                f"{self._raw_ipv6 and '[' + self.addr + ']' or self.addr}:"
                f"{self.tls_port} with {self.tls_certfile}\n"
            )
        self._print_dashboard_info()

        endpoints = build_endpoint_description_strings(host=self.addr, port=self.port)
        if not self.no_tls:
            endpoints.append(
                tls_endpoint_string(
                    self.tls_port, self.addr, self.tls_certfile, self.tls_keyfile
                )
            )

        logger.debug(
            "Daphne running, listening on %s:%s (+TLS %s:%s)",
            self.addr, self.port, self.addr, self.tls_port,
        )
        try:
            self.server_cls(
                application=self.get_application(options),
                endpoints=endpoints,
                signal_handlers=not options["use_reloader"],
                action_logger=self.log_action,
                http_timeout=options.get("http_timeout"),
                root_path=getattr(settings, "FORCE_SCRIPT_NAME", "") or "",
                websocket_handshake_timeout=options.get("websocket_handshake_timeout", 5),
                websocket_max_message_size=options.get("websocket_max_message_size")
                or getattr(settings, "DAPHNE_WEBSOCKET_MAX_MESSAGE_SIZE", 1024 * 1024),
                websocket_max_frame_size=options.get("websocket_max_frame_size")
                or getattr(settings, "DAPHNE_WEBSOCKET_MAX_FRAME_SIZE", 1024 * 1024),
            ).run()
            logger.debug("Daphne exited")
        except KeyboardInterrupt:
            shutdown_message = options.get("shutdown_message", "")
            if shutdown_message:
                self.stdout.write(shutdown_message)
            return
