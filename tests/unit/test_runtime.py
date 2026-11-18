from __future__ import annotations

from fliptuned.services.runtime import FORCED_ENV, apply_privacy_env


def test_privacy_env_is_forced_over_the_shell() -> None:
    env = {"HF_HUB_OFFLINE": "0", "HF_HUB_DISABLE_TELEMETRY": "0", "DO_NOT_TRACK": "0"}
    apply_privacy_env(env)
    for key, value in FORCED_ENV.items():
        assert env[key] == value, key
    assert env["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "1"


def test_defaults_do_not_override_the_user() -> None:
    env = {"TRANSFORMERS_NO_ADVISORY_WARNINGS": "0"}
    apply_privacy_env(env)
    assert env["TRANSFORMERS_NO_ADVISORY_WARNINGS"] == "0"
