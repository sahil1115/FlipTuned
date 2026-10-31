"""Write the API's OpenAPI schema for the generated TypeScript client.

    python -m fliptuned.api.openapi_export web/openapi.json [web/src/test/train-schema.json]

The optional second file is the training config's JSON Schema, which the web tests render
(``npm --prefix web run gen:api`` writes both, then runs openapi-typescript).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from fliptuned.api.app import openapi_document
from fliptuned.services.configs import json_schema


def _write(path: str, data: object) -> None:
    text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    Path(path).write_bytes(text.encode("utf-8"))


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) not in (1, 2):
        print(
            "usage: python -m fliptuned.api.openapi_export <out.json> [<train-schema.json>]",
            file=sys.stderr,
        )
        return 2
    _write(args[0], openapi_document())
    if len(args) == 2:
        _write(args[1], json_schema("train"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
