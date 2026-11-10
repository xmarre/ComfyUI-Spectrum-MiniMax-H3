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
        getattr(code, "co_qualname", code.co_name),
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


_UNREVIEWABLE_DEFAULT = object()


def _default_spec(node: Any) -> tuple[str, Any] | None:
    """Describe a reviewable default expression: a literal or a dotted name."""
    import ast

    if node is None:
        return None
    if isinstance(node, ast.Constant):
        return ("const", node.value)
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, (ast.USub, ast.UAdd))
        and isinstance(node.operand, ast.Constant)
        and type(node.operand.value) in (int, float, complex)
    ):
        value = node.operand.value
        return ("const", -value if isinstance(node.op, ast.USub) else value)
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
        return ("name", tuple(reversed(parts)))
    return ("unreviewable", _UNREVIEWABLE_DEFAULT)


@lru_cache(maxsize=64)
def _reference_defaults(
    path: str,
    mtime_ns: int,
    size: int,
    lexical_path: tuple[str, ...],
) -> tuple[tuple[Any, ...], tuple[tuple[str, Any], ...]] | None:
    import ast

    try:
        data = Path(path).read_bytes()
        tree = ast.parse(data, filename=path)
    except (OSError, SyntaxError, ValueError):
        return None
    if len(data) != size:
        return None
    body = tree.body
    target = None
    for name in lexical_path:
        matches = [
            node
            for node in body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
        ]
        if len(matches) != 1:
            return None
        target = matches[0]
        body = target.body
    if not isinstance(target, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None
    positional = tuple(_default_spec(node) for node in target.args.defaults)
    keyword = tuple(
        (arg.arg, _default_spec(node))
        for arg, node in zip(target.args.kwonlyargs, target.args.kw_defaults, strict=True)
        if node is not None
    )
    return positional, keyword


def _resolve_default(spec: tuple[str, Any], namespace: dict[str, Any]) -> Any:
    kind, value = spec
    if kind == "const":
        return value
    if kind != "name":
        return _UNREVIEWABLE_DEFAULT
    head, *rest = value
    if head not in namespace:
        return _UNREVIEWABLE_DEFAULT
    resolved = namespace[head]
    for attribute in rest:
        try:
            resolved = getattr(resolved, attribute)
        except AttributeError:
            return _UNREVIEWABLE_DEFAULT
    return resolved


def _same_default(live: Any, spec: tuple[str, Any], namespace: dict[str, Any], immutable: tuple[type, ...]) -> bool:
    expected = _resolve_default(spec, namespace)
    if expected is _UNREVIEWABLE_DEFAULT:
        return False
    if spec[0] == "const":
        # Literal defaults are immutable; require the exact type so 1 != True etc.
        return type(live) is type(expected) and live == expected
    # A resolved name must be the identical, immutable object.
    return live is expected and isinstance(live, immutable)


def matches_source_defaults(
    function: Any,
    source_path: Path,
    lexical_path: tuple[str, ...],
    *,
    immutable_types: tuple[type, ...] = (),
) -> bool:
    """Compare live parameter defaults with those written in audited source.

    Defaults are executable behaviour that bytecode comparison does not cover.
    Each audited default must be a literal or a dotted name resolved in the
    function's own globals to an immutable object of ``immutable_types``.
    """
    base = getattr(function, "__func__", function)
    if not lexical_path or not hasattr(base, "__defaults__"):
        return False
    try:
        resolved = source_path.resolve()
        stat = resolved.stat()
    except (OSError, RuntimeError):
        return False
    reference = _reference_defaults(
        str(resolved),
        int(stat.st_mtime_ns),
        int(stat.st_size),
        tuple(lexical_path),
    )
    if reference is None:
        return False
    positional, keyword = reference
    namespace = getattr(base, "__globals__", None)
    if not isinstance(namespace, dict):
        return False
    live_positional = base.__defaults__ or ()
    if len(live_positional) != len(positional):
        return False
    if not all(
        spec is not None and _same_default(live, spec, namespace, immutable_types)
        for live, spec in zip(live_positional, positional, strict=True)
    ):
        return False
    live_keyword = dict(base.__kwdefaults__ or {})
    if set(live_keyword) != {name for name, _spec in keyword}:
        return False
    return all(
        _same_default(live_keyword[name], spec, namespace, immutable_types)
        for name, spec in keyword
    )


_DESCRIPTOR_KINDS = {
    "staticmethod": "staticmethod",
    "classmethod": "classmethod",
    "property": "property",
}


@lru_cache(maxsize=32)
def _reference_class_inventory(
    path: str,
    mtime_ns: int,
    size: int,
    class_name: str,
) -> tuple[tuple[str, str], ...] | None:
    """Return (name, binding kind) for every callable a class body defines."""
    import ast

    try:
        data = Path(path).read_bytes()
        tree = ast.parse(data, filename=path)
    except (OSError, SyntaxError, ValueError):
        return None
    if len(data) != size:
        return None
    classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    ]
    if len(classes) != 1:
        return None
    inventory: dict[str, str] = {}
    for index, node in enumerate(classes[0].body):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            decorators = node.decorator_list
            if not decorators:
                kind = "function"
            elif (
                len(decorators) == 1
                and isinstance(decorators[0], ast.Name)
                and decorators[0].id in _DESCRIPTOR_KINDS
            ):
                kind = _DESCRIPTOR_KINDS[decorators[0].id]
            else:
                kind = "unsupported"
            # A later definition rebinds the name, exactly as at class creation.
            inventory[node.name] = kind
        elif isinstance(node, ast.Pass) or (
            index == 0
            and isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            continue
        else:
            inventory["<class-body statement>"] = "unsupported"
    return tuple(sorted(inventory.items()))


def _live_binding_kind(value: Any) -> str | None:
    # A descriptor subclass can override __get__ while retaining the reviewed
    # __func__. Only the built-in descriptor has the source's binding semantics.
    if type(value) is staticmethod:
        return "staticmethod" if isinstance(value.__func__, types.FunctionType) else None
    if type(value) is classmethod:
        return "classmethod" if isinstance(value.__func__, types.FunctionType) else None
    if type(value) is property:
        return "property"
    if isinstance(value, types.FunctionType):
        return "function"
    return None


def class_callable_inventory_reason(
    cls: type,
    source_path: Path,
    class_name: str,
) -> str | None:
    """Compare a live class's callables and binding kinds with its audited source.

    Every callable the class body defines must still be present with the same
    binding kind (instance function, staticmethod, classmethod or property), and
    the live class may not carry callables or descriptors the source lacks.
    """
    try:
        resolved = source_path.resolve()
        stat = resolved.stat()
    except (OSError, RuntimeError):
        return f"{class_name} audited source is unavailable"
    inventory = _reference_class_inventory(
        str(resolved),
        int(stat.st_mtime_ns),
        int(stat.st_size),
        class_name,
    )
    if inventory is None:
        return f"{class_name} audited source inventory is unavailable"
    expected = dict(inventory)
    unsupported = sorted(name for name, kind in expected.items() if kind == "unsupported")
    if unsupported:
        return f"{class_name} audited source declares unreviewable members: {unsupported}"
    live = vars(cls)
    for name, kind in inventory:
        actual = _live_binding_kind(live.get(name))
        if actual != kind:
            return (
                f"{class_name}.{name} is missing or not the reviewed {kind} "
                f"(found {type(live.get(name)).__name__})"
            )
    for name, value in live.items():
        if name in expected:
            continue
        # These storage descriptors are created by Python, not by the class
        # body. An arbitrary descriptor must not pass as a noncallable value.
        if (
            name in {"__dict__", "__weakref__"}
            and type(value) is types.GetSetDescriptorType
            and value.__objclass__ is cls
        ):
            continue
        if callable(value) or any(hasattr(type(value), hook) for hook in ("__get__", "__set__", "__delete__")):
            return f"{class_name}.{name} is an unreviewed callable attribute"
    return None
