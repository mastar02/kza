"""
Safety Tests — No Hardcoded Secrets

Scan the source tree for patterns that look like hardcoded credentials:
- Private IP addresses used as runtime defaults (not in comments/examples)
- Bare tokens, passwords, API keys assigned as string literals
- Environment variable references that forgot the ${...} wrapper

These tests act as a CI guardrail: if someone accidentally commits a real
token or a hardcoded host, the build breaks.
"""

import re
from pathlib import Path

import pytest

# Root of source code to scan
SRC_ROOT = Path(__file__).resolve().parent.parent.parent / "src"
CONFIG_ROOT = Path(__file__).resolve().parent.parent.parent / "config"

# Files to skip (tests themselves, __pycache__, etc.)
SKIP_PATTERNS = {"__pycache__", ".pyc", "node_modules", ".git"}


def _python_source_files() -> list[Path]:
    """Collect all .py files under src/."""
    files = []
    for p in SRC_ROOT.rglob("*.py"):
        if not any(skip in str(p) for skip in SKIP_PATTERNS):
            files.append(p)
    return files


def _config_files() -> list[Path]:
    """Collect config YAML files."""
    return list(CONFIG_ROOT.rglob("*.yaml")) + list(CONFIG_ROOT.rglob("*.yml"))


# Regex for private-network IPs
_IP_RE = re.compile(
    r"(?:192\.168|10\.0|172\.(?:1[6-9]|2\d|3[01]))\.\d{1,3}\.\d{1,3}"
)

# Regex for string literals that look like raw tokens/passwords
_SECRET_LITERAL_RE = re.compile(
    r"""(['"])                       # opening quote
        (?:eyJ[A-Za-z0-9_\-]{20,}   # JWT-like
          |ghp_[A-Za-z0-9]{30,}     # GitHub PAT
          |sk-[A-Za-z0-9]{30,}      # OpenAI-style key
        )
        \1                          # matching close quote
    """,
    re.VERBOSE,
)


class TestNoHardcodedSecrets:
    """Verify source files don't contain hardcoded secrets."""

    def test_no_hardcoded_ips_in_source(self):
        """Source code should not contain hardcoded private IPs for runtime use."""
        violations = []
        for path in _python_source_files():
            content = path.read_text(errors="replace")
            for lineno, line in enumerate(content.splitlines(), 1):
                stripped = line.strip()
                # Skip comments and docstrings
                if stripped.startswith("#"):
                    continue
                # Skip lines that are clearly examples/documentation
                if any(kw in stripped.lower() for kw in ["example", "# ", "doc", "192.168.1.x"]):
                    continue
                if _IP_RE.search(stripped):
                    violations.append(f"{path.relative_to(SRC_ROOT)}:{lineno}: {stripped[:120]}")

        assert not violations, (
            "Hardcoded private IPs found in source files — use env vars or config:\n"
            + "\n".join(violations)
        )

    def test_no_hardcoded_ips_in_config(self):
        """Config YAML should reference env vars for hosts, not hardcoded IPs."""
        violations = []
        for path in _config_files():
            content = path.read_text(errors="replace")
            for lineno, line in enumerate(content.splitlines(), 1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if _IP_RE.search(stripped):
                    # Allow if the value is wrapped in ${...} (env var reference)
                    if "${" in stripped:
                        continue
                    violations.append(f"{path.name}:{lineno}: {stripped[:120]}")

        assert not violations, (
            "Hardcoded private IPs found in config — use ${ENV_VAR} references:\n"
            + "\n".join(violations)
        )

    def test_no_jwt_or_token_literals_in_source(self):
        """Source code should not contain string literals that look like real tokens."""
        violations = []
        for path in _python_source_files():
            content = path.read_text(errors="replace")
            for lineno, line in enumerate(content.splitlines(), 1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if _SECRET_LITERAL_RE.search(stripped):
                    violations.append(f"{path.relative_to(SRC_ROOT)}:{lineno}: {stripped[:80]}...")

        assert not violations, (
            "Possible hardcoded secrets found in source files:\n"
            + "\n".join(violations)
        )

    def test_settings_yaml_uses_env_vars_for_secrets(self):
        """settings.yaml should use ${ENV_VAR} for token and secret fields."""
        settings_path = CONFIG_ROOT / "settings.yaml"
        if not settings_path.exists():
            pytest.skip("settings.yaml not found")

        content = settings_path.read_text()
        secret_key_re = re.compile(r"^\s*(token|secret|password|api_key|client_secret)\s*:", re.MULTILINE)

        for match in secret_key_re.finditer(content):
            # Get the full line
            line_start = content.rfind("\n", 0, match.start()) + 1
            line_end = content.find("\n", match.end())
            line = content[line_start:line_end].strip()

            # Value should be an env-var reference or null/empty
            if "${" not in line and "null" not in line and '""' not in line and "''" not in line:
                # Check if the line is a comment
                if not line.lstrip().startswith("#"):
                    pytest.fail(
                        f"settings.yaml contains a secret field without ${{ENV_VAR}} reference: {line}"
                    )

    def test_no_env_vars_with_default_secret_values(self):
        """os.getenv() calls for secrets should not have non-empty defaults."""
        # Pattern: os.getenv("SECRET_KEY", "actual_value")
        getenv_with_default_re = re.compile(
            r'os\.getenv\(\s*["\']'
            r'(HOME_ASSISTANT_TOKEN|SPOTIFY_CLIENT_SECRET|SPOTIFY_CLIENT_ID|API_KEY)'
            r'["\']\s*,\s*["\']([^"\']+)["\']\s*\)'
        )

        violations = []
        for path in _python_source_files():
            content = path.read_text(errors="replace")
            for match in getenv_with_default_re.finditer(content):
                var_name = match.group(1)
                default_val = match.group(2)
                if default_val and not default_val.startswith("${"):
                    violations.append(
                        f"{path.relative_to(SRC_ROOT)}: os.getenv('{var_name}') has default '{default_val[:20]}...'"
                    )

        assert not violations, (
            "os.getenv() for secrets should not have non-empty defaults:\n"
            + "\n".join(violations)
        )
