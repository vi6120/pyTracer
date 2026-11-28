"""Mechanical mutants of the support agent's glue (``benchmarks/verdicts/support_agent/glue.py``).

Each mutant is one small edit made by a fixed rule, the kind mutation-testing tools
make: a comparison or a boolean flipped, a condition negated, a string emptied, a
statement removed, an exception type or a method swapped, an entry dropped from a
tuple, a dict or a returned list. Only the code between the model and the tools is
mutated (the four glue functions and the two constants they use); docstrings are
left alone.

The edit replaces only the source text of the node it changes, so the rest of the
file, comments included, stays as written. The list is the same every time: the
order follows the file.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

FUNCTIONS = ("prepare_args", "call_tool", "history_for_model", "run_record")
CONSTANTS = ("ORDER_WRITES", "ORDER_ID")

_CMP_SWAP: dict[type, type] = {
    ast.In: ast.NotIn, ast.NotIn: ast.In, ast.Eq: ast.NotEq, ast.NotEq: ast.Eq,
    ast.Is: ast.IsNot, ast.IsNot: ast.Is,
}
_EXC_SWAP = {"TimeoutError": "ValueError", "Exception": "TimeoutError"}
_METHOD_SWAP = {"startswith": "endswith", "fullmatch": "match", "lstrip": "rstrip"}


@dataclass(frozen=True)
class Mutant:
    id: str  # M001, M002, ... in file order
    operator: str
    where: str  # the glue function or constant it is in
    line: int
    description: str
    source: str  # the whole mutated glue.py
    cosmetic: str  # the same mutant with a comment-only line added above the edit


def _replace(source: str, node: ast.AST, text: str) -> str:
    """The source with ``node``'s own text replaced by ``text``."""
    lines = source.splitlines(keepends=True)
    starts = [0]
    for line in lines:
        starts.append(starts[-1] + len(line))
    begin = starts[node.lineno - 1] + len(lines[node.lineno - 1].encode()[: node.col_offset].decode())
    end_line = lines[node.end_lineno - 1]  # type: ignore[operator]
    end = starts[node.end_lineno - 1] + len(end_line.encode()[: node.end_col_offset].decode())  # type: ignore[operator]
    return source[:begin] + text + source[end:]


def _comment_above(source: str, line: int) -> str:
    """The source with a comment-only line above ``line``: nothing that runs changes."""
    lines = source.splitlines(keepends=True)
    target = lines[line - 1]
    indent = target[: len(target) - len(target.lstrip())]
    return "".join([*lines[: line - 1], f"{indent}# checked\n", *lines[line - 1:]])


def _docstring(node: ast.AST, parent: Any) -> bool:
    body: Any = getattr(parent, "body", None)
    return bool(body) and body[0] is node and isinstance(node, ast.Expr) and isinstance(
        getattr(node, "value", None), ast.Constant) and isinstance(node.value.value, str)  # type: ignore[attr-defined]


def _edits(scope: ast.AST) -> Iterator[tuple[str, ast.AST, str, str]]:
    """(operator, node, replacement text, description) for every mutation in ``scope``."""
    parents = {child: parent for parent in ast.walk(scope) for child in ast.iter_child_nodes(parent)}
    new: Any
    for node in ast.walk(scope):
        parent: Any = parents.get(node)
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in _CMP_SWAP:
            new = ast.Compare(node.left, [_CMP_SWAP[type(node.ops[0])]()], node.comparators)
            yield "flip comparison", node, ast.unparse(new), f"`{ast.unparse(node)}` -> `{ast.unparse(new)}`"
        elif isinstance(node, ast.BoolOp):
            new_op = ast.Or() if isinstance(node.op, ast.And) else ast.And()
            new = ast.BoolOp(new_op, node.values)
            yield "swap and/or", node, ast.unparse(new), f"`{ast.unparse(node)}` -> `{ast.unparse(new)}`"
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            yield "drop not", node, ast.unparse(node.operand), f"`{ast.unparse(node)}` -> `{ast.unparse(node.operand)}`"
        elif isinstance(node, (ast.If, ast.IfExp)) and not isinstance(node.test, ast.UnaryOp):
            new = f"not ({ast.unparse(node.test)})"
            yield "negate condition", node.test, new, f"`{ast.unparse(node.test)}` -> `{new}`"
        elif isinstance(node, ast.comprehension):
            for test in node.ifs:
                if not isinstance(test, ast.UnaryOp):
                    new = f"not ({ast.unparse(test)})"
                    yield "negate condition", test, new, f"`{ast.unparse(test)}` -> `{new}`"
        elif (isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value
              and not isinstance(parent, (ast.JoinedStr, ast.Expr))):  # not f-string parts or docstrings
            yield "empty string", node, '""', f"`{ast.unparse(node)}` -> `\"\"`"
        elif isinstance(node, ast.Name) and node.id in _EXC_SWAP and isinstance(parent, ast.ExceptHandler):
            new = _EXC_SWAP[node.id]
            yield "swap exception", node, new, f"`except {node.id}` -> `except {new}`"
        elif isinstance(node, ast.Attribute) and node.attr in _METHOD_SWAP:
            new = f"{ast.unparse(node.value)}.{_METHOD_SWAP[node.attr]}"
            yield "swap method", node, new, f"`.{node.attr}` -> `.{_METHOD_SWAP[node.attr]}`"
        elif isinstance(node, ast.Return) and isinstance(node.value, ast.Call) and isinstance(
                node.value.func, ast.Name) and node.value.func.id in ("list", "dict"):
            value = ast.unparse(node.value)
            if node.value.func.id == "list":
                for cut, said in (("[1:]", "the first"), ("[:-1]", "the last")):
                    yield "drop element", node.value, value + cut, f"`{value}` -> `{value + cut}` (drops {said})"
            else:
                yield "drop element", node.value, "{}", f"`{value}` -> `{{}}` (drops every entry)"
        elif isinstance(node, ast.Continue):
            yield "continue to break", node, "break", "`continue` -> `break`"
        elif isinstance(node, ast.Dict) and len(node.keys) > 1:
            for i, key in enumerate(node.keys):
                new = ast.Dict([k for j, k in enumerate(node.keys) if j != i],
                               [v for j, v in enumerate(node.values) if j != i])
                yield "drop dict entry", node, ast.unparse(new), f"drop `{ast.unparse(key)}` from a dict"  # type: ignore[arg-type]
        elif isinstance(node, ast.Tuple) and isinstance(parent, ast.Assign) and len(node.elts) > 1:
            for i, elt in enumerate(node.elts):
                new = ast.Tuple([e for j, e in enumerate(node.elts) if j != i])
                yield "drop tuple entry", node, f"({', '.join(ast.unparse(e) for e in new.elts)},)", \
                    f"drop `{ast.unparse(elt)}` from the tuple"
        if isinstance(node, ast.stmt) and not isinstance(node, ast.FunctionDef) and not _docstring(node, parent):
            if isinstance(parent, (ast.FunctionDef, ast.If, ast.For, ast.Try, ast.ExceptHandler)):
                yield "remove statement", node, "pass", f"remove `{ast.unparse(node).splitlines()[0]}`"


def mutants(source: str) -> list[Mutant]:
    tree = ast.parse(source)
    found: list[Mutant] = []
    seen: set[str] = set()
    scopes = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name in FUNCTIONS)
              or (isinstance(n, ast.Assign) and any(getattr(t, "id", "") in CONSTANTS for t in n.targets))]
    for scope in scopes:
        where = scope.name if isinstance(scope, ast.FunctionDef) else ast.unparse(scope.targets[0])
        nodes: list[ast.AST] = [scope.value] if isinstance(scope, ast.Assign) else [scope]
        for root in nodes:
            for operator, node, text, description in _edits(root):
                mutated = _replace(source, node, text)
                if mutated == source or mutated in seen:
                    continue
                try:
                    compile(mutated, "glue.py", "exec")
                except SyntaxError:
                    continue
                seen.add(mutated)
                found.append(Mutant(
                    id="", operator=operator, where=where, line=node.lineno, description=description,
                    source=mutated, cosmetic=_comment_above(mutated, node.lineno),
                ))
    found.sort(key=lambda m: (m.line, m.operator, m.description))
    return [Mutant(**{**m.__dict__, "id": f"M{i + 1:03d}"}) for i, m in enumerate(found)]
