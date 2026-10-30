import json
from typing import TypedDict

from lpbound.config.lpbound_config import LpBoundConfig
from lpbound.config.paths import LpBoundPaths


# Define types for the foreign key to primary key join structure
class FkPkJoin(TypedDict):
    pk_relation: str
    fk: str
    pk: str


# Define the main schema type with more generic typing
class BenchmarkSchema(TypedDict):
    name: str
    relations: dict[str, dict[str, str]]  # Relation name -> (Column name -> Type)
    join_variables: dict[str, list[str]]  # Relation name -> List of join columns
    groupby_variables: dict[str, list[list[str]]]  # Relation name -> List of groupby column combinations
    equality_predicate_variables: dict[str, list[str]]  # Relation name -> List of equality predicate columns
    range_predicate_variables: dict[str, list[str]]  # Relation name -> List of range predicate columns
    fk_pk_joins_dict: dict[str, list[FkPkJoin]]  # FK relation -> List of FK-PK join specifications


def load_benchmark_schema(lpbound_config: LpBoundConfig) -> BenchmarkSchema:
    """
    Loads the JSON file that defines the schema (relations, join_variables, etc.).
    Works with any schema following the expected structure.
    """
    path = lpbound_config.schema_path or (LpBoundPaths.SCHEMAS_DIR / f"{lpbound_config.benchmark_name}_schema.json")
    with open(path, "r") as f:
        data: BenchmarkSchema = json.load(f)
    return data
