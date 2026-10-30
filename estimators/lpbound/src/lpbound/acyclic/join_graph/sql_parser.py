import sqlparse
import re
from sqlparse.sql import IdentifierList, Identifier, Where, Comparison, Parenthesis
from sqlparse.tokens import Whitespace, Comment, Operator

from datetime import datetime

# Import our classes
from lpbound.acyclic.join_graph.join_graph import JoinGraph
from lpbound.acyclic.join_graph.vertex import Vertex
from lpbound.acyclic.join_graph.edge import Edge
from lpbound.acyclic.join_graph.predicate import EqualityPredicate, InequalityPredicate
from lpbound.config.benchmark_schema import BenchmarkSchema


def parse_sql_query_to_join_graph(sql_str: str, schema_json: BenchmarkSchema) -> JoinGraph:
    """
    schema_json is the entire JSON loaded from 'schema.json'.
    We'll use schema_json["relations"] to confirm table/column types if needed.
    """

    # schema information
    relations = list(schema_json["relations"].keys())

    parsed = sqlparse.parse(sql_str)
    if not parsed:
        raise ValueError("Could not parse the SQL query.")

    statement = parsed[0]
    if statement.get_type() != "SELECT":
        raise ValueError("Only SELECT statements are supported by this simple parser.")

    # --- NEW: Extract groupby_vars from SELECT DISTINCT(...) ---
    groupby_vars_map = {}  # alias -> list of columns
    distinct_tokens = [t for t in statement.tokens if t.ttype not in (Whitespace, Comment)]
    # Find SELECT DISTINCT(...)
    for i, token in enumerate(distinct_tokens):
        if token.is_keyword and token.value.upper().startswith("SELECT"):
            # Look for DISTINCT as next token
            if i + 1 < len(distinct_tokens):
                next_token = distinct_tokens[i + 1]
                if next_token.is_keyword and next_token.value.upper().startswith("DISTINCT"):
                    # The next token after DISTINCT should be a parenthesis group
                    if i + 2 < len(distinct_tokens):
                        distinct_token = distinct_tokens[i + 2]
                        # It could be a Parenthesis or IdentifierList
                        # We'll use sqlparse to extract identifiers inside the parenthesis
                        # e.g. DISTINCT(mc.company_id, mk.keyword_id)
                        if hasattr(distinct_token, "tokens"):
                            # Find all identifiers inside
                            for subtoken in distinct_token.tokens:
                                if isinstance(subtoken, IdentifierList):
                                    for identifier in subtoken.get_identifiers():
                                        if "." in identifier.value:
                                            alias, col = identifier.value.split(".", 1)
                                            alias = alias.strip().upper()
                                            col = col.strip().upper()
                                            groupby_vars_map.setdefault(alias, []).append(col)
                                elif isinstance(subtoken, Identifier):
                                    if "." in subtoken.value:
                                        alias, col = subtoken.value.split(".", 1)
                                        alias = alias.strip().upper()
                                        col = col.strip().upper()
                                        groupby_vars_map.setdefault(alias, []).append(col)
                        # If not, fallback: try to parse the value as a string
                        elif distinct_token.value.startswith("(") and distinct_token.value.endswith(")"):
                            inner = distinct_token.value[1:-1]
                            for var in inner.split(","):
                                var = var.strip()
                                if "." in var:
                                    alias, col = var.split(".", 1)
                                    alias = alias.strip().upper()
                                    col = col.strip().upper()
                                    groupby_vars_map.setdefault(alias, []).append(col)
                    break  # Only process the first SELECT DISTINCT
                # NEW: Handle Function token structure (DISTINCT(...))
                elif hasattr(next_token, "tokens") and len(next_token.tokens) >= 2:
                    # Check if this is a DISTINCT function
                    if (
                        isinstance(next_token.tokens[0], Identifier)
                        and next_token.tokens[0].value.upper() == "DISTINCT"
                        and isinstance(next_token.tokens[1], Parenthesis)
                    ):

                        # Extract from the Parenthesis token
                        paren_token = next_token.tokens[1]
                        if len(paren_token.tokens) >= 3:  # Should have (, content, )
                            content_token = paren_token.tokens[1]

                            if isinstance(content_token, IdentifierList):
                                # Multiple variables: mc.company_id, mk.keyword_id
                                for identifier in content_token.get_identifiers():
                                    if "." in identifier.value:
                                        alias, col = identifier.value.split(".", 1)
                                        alias = alias.strip().upper()
                                        col = col.strip().upper()
                                        groupby_vars_map.setdefault(alias, []).append(col)
                            elif isinstance(content_token, Identifier):
                                # Single variable: t.kind_id
                                if "." in content_token.value:
                                    alias, col = content_token.value.split(".", 1)
                                    alias = alias.strip().upper()
                                    col = col.strip().upper()
                                    groupby_vars_map.setdefault(alias, []).append(col)
                    break  # Only process the first SELECT DISTINCT

    # We'll store alias -> base_table
    alias_to_table = {}

    # 1) Extract FROM.  The benchmark SQL uses a flat comma-separated FROM
    # list.  Parsing that text directly is more reliable than sqlparse's
    # IdentifierList here because ``at`` (the standard AKA_TITLE alias) is a
    # SQL keyword and sqlparse otherwise detaches it from its table.
    from_match = re.search(
        r"\bFROM\s+(.*?)(?:\s+WHERE\b|;|$)",
        sql_str,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if from_match is None:
        raise ValueError("No FROM clause found in the query.")
    for table_reference in from_match.group(1).split(","):
        parts = table_reference.strip().split()
        if not parts:
            continue
        table_name = parts[0]
        alias = parts[-1] if len(parts) > 1 else table_name
        if len(parts) >= 2 and parts[-2].upper() == "AS":
            alias = parts[-1]
        alias_to_table[alias.upper()] = table_name.upper()

    where_token = None
    # tokens = list(statement.tokens)
    tokens = [t for t in statement.tokens if t.ttype not in (Whitespace, Comment)]

    for i, token in enumerate(tokens):
        # Detect WHERE block
        if isinstance(token, Where):
            where_token = token
    if not alias_to_table:
        raise ValueError("No valid table references found after FROM.")

    # 2) Create the JoinGraph and add vertices
    jg = JoinGraph(is_groupby=len(groupby_vars_map) > 0)

    for alias, table_name in alias_to_table.items():
        # We confirm table_name is in schema_json["relations"], else error
        if table_name not in schema_json["relations"]:
            raise ValueError(f"Table '{table_name}' not found in schema JSON.")
        relation_id = relations.index(table_name)
        groupby_vars = groupby_vars_map.get(alias, [])
        v = Vertex(alias=alias, relation_id=relation_id, relation_name=table_name, groupby_vars=groupby_vars)
        jg.add_vertex(v)

    # 3) Parse WHERE conditions
    if where_token:
        # We'll gather all Comparisons
        comparisons = extract_comparisons(where_token)

        for comp in comparisons:
            left_expr, operator, right_expr = extract_comparison_parts(comp)
            right_is_column = isinstance(comp.right, Identifier)

            # if left_expr = "t.id", we split on ".", get alias="t", col="id"
            # same for right_expr. If right_expr doesn't have ".", treat as literal
            if "." not in left_expr:
                raise ValueError(f"Unsupported left expression: {left_expr}")
            left_alias, left_col = left_expr.split(".")
            left_alias = left_alias.upper()
            left_col = left_col.upper()

            # sqlparse distinguishes a qualified column (Identifier) from a
            # string or numeric literal.  Looking for a dot in the text is not
            # sufficient: IMDB predicates such as ``info > '8.5'`` contain a
            # dot but are local string predicates, not joins.
            if right_is_column:
                # It's a join
                right_alias, right_col = right_expr.split(".")
                right_alias = right_alias.upper()
                right_col = right_col.upper()

                assert operator == "=", f"Only equality joins are supported for now. {operator} is not supported."
                # Create an Edge between left_alias and right_alias
                cond_str = f"{left_expr} {operator} {right_expr}"
                e = Edge(
                    alias_left=left_alias, alias_right=right_alias, join_var_left=left_col, join_var_right=right_col
                )
                # Possibly annotate PK-FK if in fk_pk_joins_dict
                # We'll do that in a later step (post-processing)
                jg.add_edge(e)
            else:
                # Local predicate
                # We guess if it's an equality or inequality
                # We'll store col_name = left_col
                # The data type is found from schema_json["relations"][table_name][col_name], if you want
                table_name = alias_to_table[left_alias]
                if left_col not in schema_json["relations"][table_name]:
                    raise ValueError(f"Column '{left_col}' not found in table '{table_name}'")
                col_type = schema_json["relations"][table_name][left_col]

                # Build the appropriate predicate
                pred = build_predicate(left_col.upper(), col_type, operator, right_expr)
                if isinstance(pred, EqualityPredicate):
                    jg.vertices[left_alias].add_equality_predicate(pred)
                else:
                    jg.vertices[left_alias].add_inequality_predicate(pred)

    # 4) Post-processing: Build transitive closure and join pools
    jg.build_transitive_closure_and_join_pools()

    # 5) Post-processing: set pk_fk_info if it matches fk_pk_joins_dict
    #    For each edge, check if base_table is in fk_pk_joins_dict
    #    e.g. if we see "MOVIE_INFO_IDX" joined to "TITLE" on "movie_id = id",
    #    we find an entry in fk_pk_joins_dict.
    annotate_pk_fk(jg, schema_json)

    return jg


# ------------------------------
# HELPER FUNCTIONS
# ------------------------------


def parse_table_identifier(identifier) -> tuple[str, str]:
    """
    For "title AS t" or "MOVIE_INFO mi", return (table_name, alias).
    """
    tbl_name = identifier.get_real_name()  # e.g. "title"
    alias = identifier.get_alias()  # e.g. "t"
    if not alias:
        alias = tbl_name
    return (tbl_name, alias)


def extract_comparisons(where_token: Where):
    """Recursively collect comparisons from a WHERE clause.

    Some benchmark generators wrap every predicate in parentheses, e.g.
    ``WHERE (a.id = b.id) AND (a.x < 10)``.  sqlparse represents those
    comparisons below ``Parenthesis`` nodes, so inspecting only the direct
    children silently drops both joins and local predicates.
    """
    comps = []

    def visit(token):
        if isinstance(token, Comparison):
            comps.append(token)
            return
        if getattr(token, "is_group", False):
            for child in token.tokens:
                visit(child)

    visit(where_token)
    return comps


def extract_comparison_parts(comp: Comparison):
    # Typically comp has structure: [Identifier('t.id'), Operator('='), Identifier('mi.movie_id')]
    left_expr_str = comp.left.value
    # the operator string should be the comparison token in comp.tokens
    operator_strs = [t.value for t in comp.tokens if t.ttype == Operator.Comparison]
    assert len(operator_strs) == 1, f"Expected exactly one operator in comparison, but got: {operator_strs} for {comp}"
    operator_str = operator_strs[0]

    right_expr_str = comp.right.value
    return (left_expr_str.strip(), operator_str.strip(), right_expr_str.strip())


def build_predicate(col_name: str, col_type: str, op: str, literal: str) -> EqualityPredicate | InequalityPredicate:
    """
    Build an EqualityPredicate or InequalityPredicate from the operator.
    col_type is 'int', 'str', etc.
    If 'str', we keep it as string. If 'int', parse as int if possible.
    """
    # Convert to python type if col_type is int and literal is numeric, etc.
    val = literal.strip()
    # Remove quotes if present
    if (val.startswith("'") and val.endswith("'")) or (val.startswith('"') and val.endswith('"')):
        val = val[1:-1]

    if col_type == "int":
        # Attempt parse
        try:
            val = int(val)
        except ValueError:
            # fallback, keep as string
            pass
    elif col_type == "float":
        try:
            val = float(val)
        except ValueError:
            pass

    # If col_type is "str", we just keep val as is

    # handle datetime
    # we convert datetime to unix timestamp
    # e.g. "2010-08-23 16:21:10"
    if col_type == "date":
        if not isinstance(val, str):
            val = str(val)
        val = datetime.strptime(val, "%Y-%m-%d %H:%M:%S").timestamp()
    # If col_type is "datetime", or anything else, also keep as is

    if op == "=":
        return EqualityPredicate(col_name, val, col_type)
    elif op == ">":
        return InequalityPredicate(col_name, col_type, left_value=val, left_inclusive=False)
    elif op == ">=":
        return InequalityPredicate(col_name, col_type, left_value=val, left_inclusive=True)
    elif op == "<":
        return InequalityPredicate(col_name, col_type, right_value=val, right_inclusive=False)
    elif op == "<=":
        return InequalityPredicate(col_name, col_type, right_value=val, right_inclusive=True)
    else:
        raise ValueError(f"Operator {op} not supported in this example.")


def annotate_pk_fk(jg: JoinGraph, schema_json: BenchmarkSchema):
    """
    Go through edges. If the base_table of one side is in fk_pk_joins_dict
    and it says it joins to the other side's base_table, annotate edge.pk_fk_info.
    For example,
      "MOVIE_INFO_IDX" -> [ { "pk_relation": "TITLE", "fk": "MOVIE_ID", "pk": "ID" } ]
    If we see an edge: "mi_idx.movie_id = t.id",
    then we set e.pk_fk_info = {"pk_table":"TITLE","fk_table":"MOVIE_INFO_IDX","fk_col":"MOVIE_ID","pk_col":"ID"}.
    """
    fk_pk = schema_json.get("fk_pk_joins_dict", {})

    # Build a small reverse lookup: table alias -> table name
    alias_to_table = {}
    for alias, vtx in jg.vertices.items():
        alias_to_table[alias] = vtx.relation_name  # e.g. "MOVIE_INFO_IDX"

    for edge in jg.edges:
        left_table = alias_to_table[edge.alias_left]  # e.g. "MOVIE_INFO_IDX"
        right_table = alias_to_table[edge.alias_right]  # e.g. "TITLE"

        left_col = edge.join_var_left
        right_col = edge.join_var_right

        # Now we see if left_table is in fk_pk
        if left_table in fk_pk:
            fk_items = fk_pk[left_table]  # e.g. [ {"pk_relation": "TITLE", "fk":"MOVIE_ID","pk":"ID"} ]
            for item in fk_items:
                if item["pk_relation"] == right_table:
                    pk_table = item["pk_relation"]  # e.g. "TITLE"
                    fk_col = item["fk"]  # e.g. "MOVIE_ID"
                    pk_col = item["pk"]  # e.g. "ID"

                    if pk_table == right_table and left_col.upper() == fk_col and right_col.upper() == pk_col:
                        # => left_table is FK, referencing right_table as PK
                        fk_alias = edge.alias_left
                        # Add pk_fk_info
                        jg.vertices[fk_alias].add_pk_fk_info((edge.alias_right, fk_col, pk_col))

        # We also do the reverse check if right_table is in fk_pk
        # and they reference left_table as pk_relation
        if right_table in fk_pk:
            fk_items = fk_pk[right_table]
            for item in fk_items:
                if item["pk_relation"] == left_table:
                    pk_table = item["pk_relation"]  # e.g. "TITLE"
                    fk_col = item["fk"]  # e.g. "MOVIE_ID"
                    pk_col = item["pk"]  # e.g. "ID"
                    if pk_table == left_table and right_col.upper() == fk_col and left_col.upper() == pk_col:
                        # => right_table is FK, referencing left_table as PK
                        fk_alias = edge.alias_right
                        # Add pk_fk_info
                        jg.vertices[fk_alias].add_pk_fk_info((edge.alias_left, fk_col, pk_col))
