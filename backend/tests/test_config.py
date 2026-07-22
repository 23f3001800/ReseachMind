"""Settings parsing tests.

ALLOWED_ORIGINS is the setting you are told to configure before exposing the
API publicly, and a parsing failure here kills the process at import — the
worst possible failure mode, since it only appears in the deployment you were
hardening. It gets direct coverage.
"""

import os
import subprocess
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent


def _load_settings_with(env: dict):
    """Import Settings in a clean subprocess with the given environment.

    A subprocess is used because pydantic-settings reads the environment at
    class-instantiation time and config is a module-level singleton.
    """
    code = (
        "import json;"
        "from config import Settings;"
        "s = Settings();"
        "print(json.dumps({'origins': s.allowed_origins_list, 'api_key': s.api_key}))"
    )
    full_env = {**os.environ, "GROQ_API_KEY": "test-key", **env}
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=BACKEND,
        env=full_env,
        capture_output=True,
        text=True,
    )
    return result


class TestAllowedOrigins:
    def test_single_origin_url_does_not_crash(self):
        """A bare URL must parse. As List[str] this raised SettingsError."""
        result = _load_settings_with({"ALLOWED_ORIGINS": "https://my-ui.example.io"})
        assert result.returncode == 0, result.stderr
        assert "https://my-ui.example.io" in result.stdout

    def test_comma_separated_origins(self):
        result = _load_settings_with(
            {"ALLOWED_ORIGINS": "https://a.example.io, https://b.example.io"}
        )
        assert result.returncode == 0, result.stderr
        assert "https://a.example.io" in result.stdout
        assert "https://b.example.io" in result.stdout

    def test_default_is_wildcard(self):
        env = {k: v for k, v in os.environ.items() if k != "ALLOWED_ORIGINS"}
        code = (
            "import json;"
            "from config import Settings;"
            "print(json.dumps(Settings().allowed_origins_list))"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=BACKEND,
            env={**env, "GROQ_API_KEY": "test-key", "ALLOWED_ORIGINS": ""},
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert '["*"]' in result.stdout
