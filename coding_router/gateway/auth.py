"""Local gateway API-key management.

The gateway key is ONLY for client → gateway authentication.
It is NOT a provider API key and must never be used as one.
"""

from __future__ import annotations

import os
import secrets
from pathlib import Path

_DEFAULT_KEY_PATH = Path.home() / ".config" / "coding-router" / "gateway.key"
_ENV_VAR = "CODING_ROUTER_API_KEY"


def _key_path() -> Path:
    """Return the path where the local gateway key is stored."""
    override = os.environ.get("CODING_ROUTER_KEY_PATH")
    return Path(override) if override else _DEFAULT_KEY_PATH


def generate_key() -> str:
    """Generate a new cryptographically-secure gateway key."""
    return f"sk-router-{secrets.token_urlsafe(32)}"


def load_or_create_key() -> str:
    """Load the gateway key from env/file, creating one if it doesn't exist.

    Priority:
      1. ``CODING_ROUTER_API_KEY`` env var
      2. ``~/.config/coding-router/gateway.key`` file
      3. Auto-generate + persist to file
    """
    # 1. Environment variable
    env_key = os.environ.get(_ENV_VAR, "").strip()
    if env_key:
        return env_key

    # 2. Persisted file
    path = _key_path()
    if path.exists():
        stored = path.read_text(encoding="utf-8").strip()
        if stored:
            return stored

    # 3. Generate + save
    key = generate_key()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(key + "\n", encoding="utf-8")
    # Restrict permissions (best-effort on non-POSIX)
    try:
        path.chmod(0o600)
    except OSError:
        pass
    return key


def validate_bearer(authorization: str | None, expected_key: str) -> bool:
    """Validate an Authorization header value against the expected key.

    Accepts both ``Bearer <key>`` and raw ``<key>``.
    """
    if not authorization:
        return False
    token = authorization.removeprefix("Bearer ").strip()
    return secrets.compare_digest(token, expected_key)
