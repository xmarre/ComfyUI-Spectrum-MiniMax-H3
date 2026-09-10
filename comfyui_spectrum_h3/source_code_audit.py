"""Stable code fingerprints derived from reviewed on-disk Python source.

Compatibility bridges already gate accepted modules by exact Git blob. This helper
adds a second, independent proof for nested callables: compile the audited file
without executing it, locate the expected lexical code path, and compare its
behavioral code structure with the live callable. The comparison does not trust
mutable bindings in ``sys.modules``.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import sys
import types
from typing import Any


def _constant_semantics(value: Any) -> Any:
    if isinstance(value, types.CodeType):
        return ("code", _code_semantics(value))
    if isinstance(value, tuple):
        return ("tuple", tuple(_constant_semantics(item) for item in value))
    if isinstance(value, frozenset):
        frozen = tuple(sorted((_constant_semantics(item) for item in value), key=repr))
        return ("frozenset", frozen)
    if value is None or isinstance(value, (bool, int, float, complex, str, bytes)):
        return (type(value).__name__, value)
    if value is Ellipsis:
        return ("ellipsis",)
    return (type(value).__module__, type(value).__qualname__, repr(value))


def _code_semantics(code: types.CodeType) -> tuple[Any, ...]:
    return (
        code.co_name,
        code.co_qualname,
        code.co_argcount,
        getattr(code, "co_posonlyargcount", 0),
        code.co_kwonlyargcount,
        code.co_nlocals,
        code.co_stacksize,
        code.co_flags,
        code.co_code,
        tuple(_constant_semantics(value) for value in code.co_consts),
        code.co_names,
        code.co_varnames,
        code.co_freevars,
        code.co_cellvars,
        getattr(code, "co_exceptiontable", b""),
    )


def _direct_nested_code(parent: types.CodeType, name: str) -> types.CodeType | None:
    matches = [
        value
        for value in parent.co_consts
        if isinstance(value, types.CodeType) and value.co_name == name
    ]
    if len(matches) != 1:
        return None
    return matches[0]


@lru_cache(maxsize=32)
def _reference_semantics(
    path: str,
    mtime_ns: int,
    size: int,
    lexical_path: tuple[str, ...],
    optimize: int,
) -> tuple[Any, ...] | None:
    try:
        data = Path(path).read_bytes()
    except OSError:
        return None
    if len(data) != size:
        return None
    try:
        current = compile(
            data,
            path,
            "exec",
            dont_inherit=True,
            optimize=optimize,
        )
    except (SyntaxError, ValueError, TypeError):
        return None
    for name in lexical_path:
        current = _direct_nested_code(current, name)
        if current is None:
            return None
    return _code_semantics(current)


def matches_nested_source_code(
    function: Any,
    source_path: Path,
    lexical_path: tuple[str, ...],
) -> bool:
    """Compare a live callable with the nested code compiled from audited source."""
    base = getattr(function, "__func__", function)
    live_code = getattr(base, "__code__", None)
    if not isinstance(live_code, types.CodeType) or not lexical_path:
        return False
    try:
        resolved = source_path.resolve()
        stat = resolved.stat()
    except (OSError, RuntimeError):
        return False
    reference = _reference_semantics(
        str(resolved),
        int(stat.st_mtime_ns),
        int(stat.st_size),
        tuple(lexical_path),
        int(sys.flags.optimize),
    )
    return reference is not None and _code_semantics(live_code) == reference
