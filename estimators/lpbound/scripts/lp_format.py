"""Losslessly read OR-Tools LP exports for C++ timing."""
from pathlib import Path
import re

def parse_exported_lp(
    program_path: Path,
) -> tuple[list[dict[str, object]], dict[int, str], dict[int, float]]:
    """Parse the exact OR-Tools LP exported by the Python estimator.

    The repository's former C++ input generator reconstructed a stronger,
    optimized Shannon LP from relation variables.  That can change the
    objective for some STATS join graphs.  Timing must solve the exact dumped
    LP instead, so this parser preserves every exported variable, coefficient,
    inequality direction, and right-hand side.
    """
    lines = program_path.read_text().splitlines()
    section = ""
    variable_names: list[str] = []
    objective_expression = ""
    constraint_expressions: list[tuple[str, str, float]] = []

    def parse_terms(expression: str) -> dict[str, float]:
        tokens = expression.split()
        if len(tokens) % 2:
            raise ValueError(f"Malformed LP expression in {program_path}: {expression}")
        terms: dict[str, float] = {}
        for index in range(0, len(tokens), 2):
            coefficient = float(tokens[index])
            name = tokens[index + 1]
            terms[name] = terms.get(name, 0.0) + coefficient
        return terms

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("\\"):
            continue
        if line == "Maximize":
            section = "objective"
            continue
        if line == "Subject to":
            section = "constraints"
            continue
        if line == "Bounds":
            section = "bounds"
            continue
        if line == "End":
            break

        if section == "objective":
            if ":" not in line:
                raise ValueError(f"Malformed objective in {program_path}: {line}")
            objective_expression = line.split(":", 1)[1].strip()
        elif section == "constraints":
            if ":" not in line:
                raise ValueError(f"Malformed constraint in {program_path}: {line}")
            expression = line.split(":", 1)[1].strip()
            match = re.fullmatch(r"(.+?)\s*(<=|>=|=)\s*([-+0-9.eE]+)", expression)
            if match is None:
                raise ValueError(f"Malformed constraint in {program_path}: {line}")
            constraint_expressions.append(
                (match.group(1).strip(), match.group(2), float(match.group(3)))
            )
        elif section == "bounds":
            match = re.fullmatch(r"([-+0-9.eE]+)\s*<=\s*(\S+)", line)
            if match is None or float(match.group(1)) != 0.0:
                raise ValueError(f"Unsupported variable bound in {program_path}: {line}")
            variable_names.append(match.group(2))

    if not variable_names or not objective_expression:
        raise ValueError(f"Incomplete LP export: {program_path}")
    variable_ids = {name: index for index, name in enumerate(variable_names)}

    objective: dict[int, float] = {}
    for name, coefficient in parse_terms(objective_expression).items():
        objective[variable_ids[name]] = coefficient

    sense_names = {"<=": "L", ">=": "G", "=": "E"}
    parsed_constraints: list[dict[str, object]] = []
    for expression, sense, rhs in constraint_expressions:
        coefficients = {
            variable_ids[name]: coefficient
            for name, coefficient in parse_terms(expression).items()
        }
        parsed_constraints.append(
            {"sense": sense_names[sense], "rhs": rhs, "variables": coefficients}
        )

    variables = {variable_id: name for name, variable_id in variable_ids.items()}
    return parsed_constraints, variables, objective


