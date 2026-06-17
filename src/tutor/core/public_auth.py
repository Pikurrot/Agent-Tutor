from __future__ import annotations

import os
import secrets


def get_public_credentials() -> tuple[str, str]:
    username = os.environ.get("PUBLIC_APP_USERNAME", "").strip()
    password = os.environ.get("PUBLIC_APP_PASSWORD", "")
    return username, password


def verify_public_login(username: str, password: str) -> bool:
    expected_user, expected_password = get_public_credentials()
    if not expected_user or not expected_password:
        return False
    user_ok = secrets.compare_digest(username.strip(), expected_user)
    pass_ok = secrets.compare_digest(password, expected_password)
    return user_ok and pass_ok


def public_auth_configured() -> bool:
    username, password = get_public_credentials()
    return bool(username and password)
