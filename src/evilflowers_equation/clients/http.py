"""One way to build an HTTP session for the services this worker calls."""

import requests
from pydantic import SecretStr

from evilflowers_equation import __version__

USER_AGENT = f"evilflowers-equation-service/{__version__}"


def session(api_key: SecretStr | None = None) -> requests.Session:
    """A session with our User-Agent and, when given, the bearer key.

    ``requests`` is instrumented by OpenTelemetry once telemetry is on, so every
    call made through it carries ``traceparent``.
    """
    http = requests.Session()
    http.headers["User-Agent"] = USER_AGENT
    if api_key is not None and (secret := api_key.get_secret_value()):
        http.headers["Authorization"] = f"Bearer {secret}"
    return http
