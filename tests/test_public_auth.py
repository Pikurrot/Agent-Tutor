from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from tutor.core.public_auth import public_auth_configured, verify_public_login


class TestPublicAuth(unittest.TestCase):
    @patch.dict(
        os.environ,
        {"PUBLIC_APP_USERNAME": "deep", "PUBLIC_APP_PASSWORD": "learning26"},
        clear=False,
    )
    def test_valid_credentials(self) -> None:
        self.assertTrue(public_auth_configured())
        self.assertTrue(verify_public_login("deep", "learning26"))

    @patch.dict(
        os.environ,
        {"PUBLIC_APP_USERNAME": "deep", "PUBLIC_APP_PASSWORD": "learning26"},
        clear=False,
    )
    def test_invalid_credentials(self) -> None:
        self.assertFalse(verify_public_login("deep", "wrong"))
        self.assertFalse(verify_public_login("other", "learning26"))

    @patch.dict(os.environ, {"PUBLIC_APP_USERNAME": "", "PUBLIC_APP_PASSWORD": ""}, clear=False)
    def test_missing_config(self) -> None:
        self.assertFalse(public_auth_configured())
        self.assertFalse(verify_public_login("deep", "learning26"))


if __name__ == "__main__":
    unittest.main()
