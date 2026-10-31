"""Service layer: the only layer the CLI and the API may call.

Conventions:
- each public service function takes a typed request (Pydantic) and returns a typed result;
- services never print, and never read CLI arguments or HTTP objects;
- failures raise :class:`fliptuned.services.errors.FlipTunedError` subclasses with a
  plain message, a hint and optional details — never record content or secrets.
"""
