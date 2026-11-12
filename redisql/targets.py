import re
from array import array
from collections import defaultdict, deque

import sqlglot
from sqlglot import exp
from sqlglot.dialects.sqlite import SQLite
from sqlglot.optimizer.scope import Scope, traverse_scope

from .sql import normalize_sql, parse_query, qualify_query, sqlite_string_columns


class _SpanGenerator(SQLite.Generator):
    def __init__(self, prefix):
        super().__init__(dialect="sqlite", comments=False, pretty=False, normalize=False)
        self.prefix = prefix

    def sql(self, expression, key=None, comment=True):
        value = super().sql(expression, key, comment)
        if isinstance(expression, exp.Expression) and key is None:
            return f"{self.prefix}[{id(expression)}:s]{value}{self.prefix}[{id(expression)}:e]"
        return value


def _canonical(sql, required=False):
    try:
        text = normalize_sql(sql)
        return text, parse_query(text)
    except Exception:
        if required:
            raise ValueError("Repair SQL must be a single parsable read-only query") from None
        return sql, None


def _spans(tree, text):
    nodes = list(tree.walk(bfs=False))
    prefix = "\ue000"
    while prefix in text:
        prefix += "\ue000"
    marked = _SpanGenerator(prefix).generate(tree, copy=False)
    pattern = re.compile(re.escape(prefix) + r"\[(\d+):(s|e)\]")
    pieces = []
    starts = defaultdict(list)
    spans = {}
    cursor = 0
    offset = 0
    for match in pattern.finditer(marked):
        piece = marked[cursor:match.start()]
        pieces.append(piece)
        offset += len(piece)
        key = int(match.group(1))
        if match.group(2) == "s":
            starts[key].append(offset)
        elif starts[key]:
            start = starts[key].pop()
            previous = spans.get(key)
            spans[key] = (min(previous[0], start), max(previous[1], offset)) if previous else (start, offset)
        cursor = match.end()
    pieces.append(marked[cursor:])
    if "".join(pieces) != text:
        spans = _fragment_spans(tree, text)
    spans[id(tree)] = (0, len(text))
    for node in reversed(nodes):
        if id(node) not in spans:
            children = [spans[id(child)] for child in node.iter_expressions() if id(child) in spans]
            if children:
                spans[id(node)] = (min(start for start, end in children), max(end for start, end in children))
    return nodes, spans


def _fragment_spans(tree, text):
    tokens = sqlglot.Dialect.get_or_raise("sqlite").tokenize(text)
    starts = {token.start for token in tokens}
    ends = {token.end + 1 for token in tokens}
    spans = {id(tree): (0, len(text))}
    used = defaultdict(set)
    for node in tree.walk(bfs=False):
        if node is tree:
            continue
        parent = node.parent
        while parent is not None and id(parent) not in spans:
            parent = parent.parent
        left, right = spans.get(id(parent), (0, len(text)))
        fragment = node.sql(dialect="sqlite", pretty=False, comments=False, normalize=False)
        position = text.find(fragment, left, right)
        while position >= 0:
            end = position + len(fragment)
            key = (id(parent), fragment)
            if position in starts and end in ends and position not in used[key]:
                spans[id(node)] = (position, end)
                used[key].add(position)
                break
            position = text.find(fragment, position + 1, right)
    return spans


def _resolved_spans(tree, text, schema):
    nodes, spans = _spans(tree, text)
    if schema is not None:
        for column in sqlite_string_columns(tree, schema):
            literal = exp.Literal.string(column.name)
            if id(column) in spans:
                spans[id(literal)] = spans[id(column)]
            column.replace(literal)
        nodes = list(tree.walk(bfs=False))
    return nodes, spans


def _lex(text):
    return [
        {"type": token.token_type.name, "text": token.text, "start": token.start, "end": token.end + 1}
        for token in sqlglot.Dialect.get_or_raise("sqlite").tokenize(text)
    ]


def _owners(tokens, nodes, spans):
    result = []
    for token in tokens:
        candidates = [
            (spans[id(node)][1] - spans[id(node)][0], index)
            for index, node in enumerate(nodes)
            if id(node) in spans and spans[id(node)][0] <= token["start"] and token["end"] <= spans[id(node)][1]
        ]
        result.append(min(candidates)[1] if candidates else 0)
    return result


def _name_key(node):
    if isinstance(node, (exp.Table, exp.Column)):
        parts = node.parts
    elif isinstance(node, exp.Identifier):
        parts = [node]
    else:
        parts = []
    return tuple(str(part.this) if part.args.get("quoted") else str(part.this).casefold() for part in parts)


def _match_key(node):
    if isinstance(node, (exp.Table, exp.Column, exp.Identifier)):
        value = _name_key(node)
    elif isinstance(node, exp.Literal):
        value = ("string" if node.is_string else "number",)
    elif isinstance(node, exp.Null):
        value = ("null",)
    elif isinstance(node, exp.Boolean):
        value = ("boolean",)
    else:
        value = ()
    return type(node).__name__, value, node.arg_key if node.parent is not None else "root"


def _node_pairs(draft_nodes, repair_nodes):
    available = defaultdict(deque)
    for index, node in enumerate(draft_nodes):
        available[_match_key(node)].append(index)
    pairs = {}
    for index, node in enumerate(repair_nodes):
        candidates = available[_match_key(node)]
        if candidates:
            pairs[index] = candidates.popleft()
    return pairs


def _lcs(left, right, equal):
    height, width = len(left), len(right)
    lengths = [array("I", [0]) * (width + 1) for _ in range(height + 1)]
    for i in range(height - 1, -1, -1):
        row, following = lengths[i], lengths[i + 1]
        for j in range(width - 1, -1, -1):
            best = max(following[j], row[j + 1])
            if equal(i, j):
                best = max(best, following[j + 1] + 1)
            row[j] = best
    pairs = []
    i = j = 0
    while i < height and j < width:
        best = lengths[i][j]
        if lengths[i + 1][j] == best:
            i += 1
        elif equal(i, j) and lengths[i + 1][j + 1] + 1 == best:
            pairs.append((i, j))
            i += 1
            j += 1
        else:
            j += 1
    return pairs


def _node_label(node):
    groups = (
        (exp.Select, exp.Union, exp.Intersect, exp.Except, exp.Subquery),
        (exp.From, exp.Where, exp.Group, exp.Having, exp.Order, exp.Limit, exp.Offset, exp.With, exp.CTE),
        (exp.Table, exp.Join),
        (exp.Predicate, exp.And, exp.Or, exp.Not),
        (exp.AggFunc,),
        (exp.Func, exp.Binary, exp.Unary),
        (exp.Identifier, exp.Column, exp.Star),
        (exp.Literal, exp.Null, exp.Boolean),
    )
    return next((index for index, classes in enumerate(groups) if isinstance(node, classes)), 8)


def _edge_label(parent, child):
    key = child.arg_key
    if isinstance(parent, exp.Select) and key == "expressions":
        return 0
    if key in {"from", "from_", "joins", "with", "with_"}:
        return 1
    if key in {"where", "having", "on", "condition"}:
        return 2
    if isinstance(parent, exp.Binary) and key == "this":
        return 3
    if isinstance(parent, exp.Binary) and key == "expression":
        return 4
    return 5


def _overlap_positions(offsets, start, end):
    return [index for index, (left, right) in enumerate(offsets) if left < end and start < right and left < right]


def _schema_positives(tree, schema):
    inferred = schema is None
    if schema is None:
        qualified = tree
        tables = {
            source.name.casefold(): {"name": source.name, "columns": None}
            for scope in traverse_scope(tree) or []
            for source in scope.sources.values()
            if isinstance(source, exp.Table)
        }
    else:
        tables = {table["name"].casefold(): table for table in schema["tables"]}
        qualified = qualify_query(tree, schema)
    scopes = list(traverse_scope(qualified) or [])
    values = set()

    def physical_column(table, name):
        item = tables.get(table.name.casefold())
        if item:
            if item["columns"] is None:
                return {f"{item['name']}.{name}"}
            for column in item["columns"]:
                if column["name"].casefold() == name.casefold():
                    return {f"{item['name']}.{column['name']}"}
        return set()

    def output(scope, name, seen):
        for projection in scope.expression.selects:
            if isinstance(projection, exp.Star) or isinstance(projection, exp.Column) and projection.is_star:
                resolved = resolve(scope, name, projection.table if isinstance(projection, exp.Column) else "", seen)
                if resolved:
                    return resolved
            if projection.alias_or_name.casefold() == name.casefold():
                resolved = set()
                for column in projection.find_all(exp.Column):
                    resolved.update(resolve(scope, column.name, column.table, seen))
                return resolved
        return set()

    def resolve(scope, name, qualifier, seen):
        key = (id(scope), name.casefold(), qualifier.casefold())
        if key in seen:
            return set()
        seen = seen | {key}
        resolved = set()
        sources = [(alias, source) for alias, source in scope.sources.items() if not qualifier or alias.casefold() == qualifier.casefold()]
        if inferred and not qualifier and len(sources) > 1:
            return set()
        for alias, source in sources:
            if isinstance(source, exp.Table):
                resolved.update(physical_column(source, name))
            elif isinstance(source, Scope):
                resolved.update(output(source, name, seen))
        if not resolved and scope.parent is not None:
            resolved.update(resolve(scope.parent, name, qualifier, seen))
        return resolved

    for scope in scopes:
        for source in scope.sources.values():
            if isinstance(source, exp.Table) and source.name.casefold() in tables:
                values.add(tables[source.name.casefold()]["name"])
        for column in scope.columns:
            if column.name != "*":
                values.update(resolve(scope, column.name, column.table, set()))
    return sorted(values)


def build_targets(draft, repair, tokenizer, schema=None, max_length=256):
    repair_sql, repair_tree = _canonical(repair, required=True)
    draft_sql, draft_tree = _canonical(draft)
    encoded = tokenizer(repair_sql, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
    input_ids = list(encoded["input_ids"])
    offsets = [tuple(offset) for offset in encoded["offset_mapping"]]
    if len(input_ids) != len(offsets) or any(len(offset) != 2 for offset in offsets):
        raise ValueError("Tokenizer must provide one character offset per token")
    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("Tokenizer must define eos_token_id")
    input_ids.append(int(eos))
    offsets.append((0, 0))
    if max_length is not None and len(input_ids) > max_length:
        raise ValueError("Repair target exceeds max_length")
    repair_nodes, repair_spans = _resolved_spans(repair_tree, repair_sql, schema)
    mode = "token"
    try:
        draft_tokens, repair_tokens = _lex(draft_sql), _lex(repair_sql)
        if draft_tree is not None:
            draft_nodes, draft_spans = _resolved_spans(draft_tree, draft_sql, schema)
            draft_owners = _owners(draft_tokens, draft_nodes, draft_spans)
            repair_owners = _owners(repair_tokens, repair_nodes, repair_spans)
            node_pairs = _node_pairs(draft_nodes, repair_nodes)
            equal = lambda i, j: draft_tokens[i]["type"] == repair_tokens[j]["type"] and draft_tokens[i]["text"] == repair_tokens[j]["text"] and node_pairs.get(repair_owners[j]) == draft_owners[i]
            mode = "ast"
        else:
            equal = lambda i, j: draft_tokens[i]["type"] == repair_tokens[j]["type"] and draft_tokens[i]["text"] == repair_tokens[j]["text"]
        matches = _lcs(draft_tokens, repair_tokens, equal)
    except sqlglot.errors.TokenError:
        draft_tokens = [{"text": value, "start": index, "end": index + 1} for index, value in enumerate(draft_sql)]
        repair_tokens = [{"text": value, "start": index, "end": index + 1} for index, value in enumerate(repair_sql)]
        matches = _lcs(draft_tokens, repair_tokens, lambda i, j: draft_tokens[i]["text"] == repair_tokens[j]["text"])
        mode = "character"
    changed = set()
    boundaries = set()
    previous_left = previous_right = -1
    retained = {right for left, right in matches}
    for left, right in matches + [(len(draft_tokens), len(repair_tokens))]:
        left_gap = left - previous_left - 1
        right_gap = right - previous_right - 1
        if right_gap:
            for index in range(previous_right + 1, right):
                token = repair_tokens[index]
                changed.update(_overlap_positions(offsets, token["start"], token["end"]))
        elif left_gap:
            next_index = right
            if mode == "character":
                next_index = next((index for index in range(right, len(repair_tokens)) if index in retained and repair_tokens[index]["text"].strip()), len(repair_tokens))
            if next_index < len(repair_tokens):
                token = repair_tokens[next_index]
                positions = _overlap_positions(offsets, token["start"], token["end"])
                if positions:
                    boundaries.add(positions[0])
            else:
                boundaries.add(len(input_ids) - 1)
        previous_left, previous_right = left, right
    identifier_positions = set()
    for node in repair_nodes:
        if isinstance(node, exp.Identifier) and isinstance(node.parent, (exp.Table, exp.Column)) and id(node) in repair_spans:
            identifier_positions.update(_overlap_positions(offsets, *repair_spans[id(node)]))
    ast_nodes = []
    node_indices = {}
    for node in repair_nodes:
        span = repair_spans.get(id(node))
        positions = _overlap_positions(offsets, *span) if span else []
        if positions:
            node_indices[id(node)] = len(ast_nodes)
            ast_nodes.append({"positions": positions, "label": _node_label(node)})
    ast_edges = []
    for node in repair_nodes:
        if node.parent is not None and id(node) in node_indices and id(node.parent) in node_indices:
            ast_edges.append({"parent": node_indices[id(node.parent)], "child": node_indices[id(node)], "label": _edge_label(node.parent, node)})
    if not ast_nodes:
        raise ValueError("Repair SQL has no target tokens")
    return {
        "repair_sql": repair_sql,
        "input_ids": input_ids,
        "token_offsets": [list(offset) for offset in offsets],
        "edit_mask": [int(index in changed or index in boundaries) for index in range(len(input_ids))],
        "identifier_mask": [int(index in changed and index in identifier_positions) for index in range(len(input_ids))],
        "changed_mask": [int(index in changed) for index in range(len(input_ids))],
        "deletion_mask": [int(index in boundaries) for index in range(len(input_ids))],
        "ast_nodes": ast_nodes,
        "ast_edges": ast_edges,
        "schema_positives": _schema_positives(repair_tree, schema),
        "alignment_mode": mode,
    }
