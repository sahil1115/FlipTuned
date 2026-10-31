from __future__ import annotations

import pytest

from fliptuned.errors import ConfigError
from fliptuned.privacy.scanner import SecretScanner

FAKE = {
    "aws_access_key": "key AKIAIOSFODNN7EXAMPLE here",
    "github_token": "token ghp_FAKEfakeFAKEfakeFAKEfakeFAKEfake1234",
    "slack_token": "xoxb-1234567890-fakefakefake",
    "api_secret_key": "OPENAI_KEY sk-proj-FAKEfakeFAKEfakeFAKEfake",
    "jwt": "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.fakesignaturefake",
    "connection_string": "mongodb+srv://admin:s3cretpw@cluster0.example.net/db",
    "account_key": "DefaultEndpointsProtocol=https;AccountKey=FAKEfakeFAKEfakeFAKEfake0123==;",
    "password_assignment": 'db_password: "hunter2hunter2"',
    "private_key": "-----BEGIN RSA PRIVATE KEY-----\nMIIfake\n-----END RSA PRIVATE KEY-----",
}


@pytest.mark.parametrize("kind", sorted(FAKE))
def test_builtin_patterns_detect_fake_secrets(kind: str) -> None:
    findings = SecretScanner().scan(FAKE[kind], "answer")
    assert [f.type for f in findings] == [kind]
    finding = findings[0]
    assert finding.field == "answer" and 0 <= finding.start < finding.end <= len(FAKE[kind])


@pytest.mark.parametrize(
    "text",
    [
        "password = <your-password>",
        "password: ********",
        "api_key=${API_KEY}",
        "token = {{ token }}",
        "Set password = changeme in the template",
        "Use a password manager; never share passwords.",
        "postgres://localhost:5432/app",
        "max_tokens = 100000",
        "token = os.environ['API_TOKEN']",
        "Count tokens: tokenizer.encode(text)",
    ],
)
def test_placeholders_and_prose_are_not_flagged(text: str) -> None:
    assert SecretScanner().scan(text, "answer") == []


@pytest.mark.parametrize(
    "text",
    [
        "DB_PASSWORD=hunter2hunter2",
        "SECRET_KEY = 'django-insecure-fake'",
        '"apiKey": "fakeKEY123456"',
    ],
)
def test_prefixed_and_suffixed_key_names_are_flagged(text: str) -> None:
    assert [f.type for f in SecretScanner().scan(text, "answer")] == ["password_assignment"]


def test_redacted_form_never_contains_the_value() -> None:
    text = FAKE["github_token"]
    (finding,) = SecretScanner().scan(text, "answer")
    secret = text[finding.start : finding.end]
    assert secret not in finding.redacted and "REDACTED" in finding.redacted
    assert secret not in finding.model_dump_json()


def test_custom_patterns() -> None:
    scanner = SecretScanner([r"\bACME-[0-9]{6}\b"])
    (finding,) = scanner.scan("internal id ACME-123456", "question")
    assert finding.type == "custom_1"
    with pytest.raises(ConfigError, match="not a valid regular expression"):
        SecretScanner(["(unclosed"])
