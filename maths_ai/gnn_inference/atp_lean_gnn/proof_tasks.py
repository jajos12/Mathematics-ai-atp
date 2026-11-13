"""Trusted theorem boundaries and original Lean context for prover tasks."""

import re


NAME = r"(?:«[^»\r\n]+»|[^\W\d][\w'′]*)(?:\.(?:«[^»\r\n]+»|[^\W\d][\w'′]*))*"


def masked(source):
    chars, position, depth = list(source), 0, 0
    while position < len(source):
        if depth:
            if source.startswith("/-", position):
                depth += 1
                chars[position:position + 2] = "  "
                position += 2
            elif source.startswith("-/", position):
                depth -= 1
                chars[position:position + 2] = "  "
                position += 2
            else:
                if source[position] != "\n":
                    chars[position] = " "
                position += 1
        elif source.startswith("/-", position):
            depth = 1
            chars[position:position + 2] = "  "
            position += 2
        elif source.startswith("--", position):
            while position < len(source) and source[position] != "\n":
                chars[position] = " "
                position += 1
        elif source[position] == '"':
            chars[position] = " "
            position += 1
            while position < len(source):
                char = source[position]
                if char != "\n":
                    chars[position] = " "
                position += 1
                if char == "\\" and position < len(source):
                    chars[position] = " "
                    position += 1
                elif char == '"':
                    break
        else:
            position += 1
    if depth:
        raise ValueError("unterminated source comment")
    return "".join(chars)


def split_prefix(prefix):
    """Extract imports while preserving all other trusted context byte offsets."""
    view, imports, spans = masked(prefix), [], []
    for match in re.finditer(r"^[ \t]*import[ \t]+([^\n]+)", view, re.MULTILINE):
        modules = match.group(1).strip().split()
        if not modules or any(not re.fullmatch(r"[A-Za-z_][\w.]*", module) for module in modules):
            raise ValueError("unsupported import declaration")
        imports.extend(modules)
        spans.append((match.start(), match.end()))
    chars = list(prefix)
    for start, end in spans:
        chars[start:end] = " " * (end - start)
    context = "".join(chars)
    if re.search(r"^\s*(?:namespace|section|set_option)\b[^\n]*\bin\s*$", masked(context), re.MULTILINE):
        raise ValueError("scoped command prefixes require a complete task context")
    return tuple(dict.fromkeys(imports)), context


def qualified_name(name, prefix):
    if name.startswith("_root_."):
        return name[7:]
    stack = []
    for line in masked(prefix).splitlines():
        match = re.fullmatch(rf"\s*(namespace|section|end)(?:\s+({NAME}))?\s*", line)
        if not match:
            continue
        kind, value = match.groups()
        if kind == "end":
            if not stack:
                raise ValueError("unbalanced source context")
            stack.pop()
        else:
            if kind == "namespace" and not value:
                raise ValueError("namespace name required")
            stack.append((kind, value))
    return ".".join([value for kind, value in stack if kind == "namespace"] + [name])


def checked_header(header, *, prefix=""):
    header = header.strip()
    if re.match(r"example\b", header):
        header = re.sub(r"^example\b", "theorem GnnSearchChecked", header, count=1)
    match = re.match(rf"(?:theorem|lemma)\s+({NAME})(?=\s|[:({{\[])", header)
    if not match:
        raise ValueError("named theorem/lemma or example header required")
    if not header or re.search(r"^[ \t]*(?:import|namespace|axiom|end)\b", masked(header), re.MULTILINE):
        raise ValueError("header must contain one declaration")
    return header, qualified_name(match.group(1), prefix)


def parse_problem(name, source):
    """Same conservative miniF2F source boundary policy as baseline cohort."""
    view = masked(source)
    declaration = re.search(r"^[ \t]*theorem\s+" + re.escape(name) + r"(?=\s|\(|\{|:)", view, re.MULTILINE)
    if declaration is None:
        raise ValueError("expected standalone named theorem not found")
    prefix = source[:declaration.start()]
    for line in masked(prefix).splitlines():
        line = line.strip()
        if line and not (line in {"import Mathlib", "import Aesop"} or line.startswith("open ") or line.startswith("set_option ")):
            raise ValueError("unsupported pre-theorem context; no dependency guessing")
    depth = 0
    for index in range(declaration.start(), len(view) - 1):
        if view[index] in "([{":
            depth += 1
        elif view[index] in ")]}":
            depth -= 1
        elif depth == 0 and view[index:index + 2] == ":=":
            return {"prefix": prefix, "header": source[declaration.start():index].rstrip()}
    raise ValueError("top-level proof assignment not found")
