import json
import re
from typing import List, Dict, Tuple, Set

class SQLQuery:
    def __init__(self, tables: List[str], alias: Dict[str, str], join_conditions: List[Tuple[str, str, str, str]], predicates: List[Tuple[str, str, str, str]]) -> None:
        self._tables = tables
        self._alias = alias
        self._join_conditions = join_conditions
        self._predicates = predicates

    @staticmethod
    def from_sql(sql: str) -> "SQLQuery":
        tables: List[str] = []
        alias: Dict[str, str] = {}
        join_conditions: List[Tuple[str, str, str, str]] = []
        predicates: List[Tuple[str, str, str, str]] = []
        alias_to_name: Dict[str, str] = {}

        from_index = sql.lower().find(" from ")
        where_index = sql.lower().find(" where ")
        if where_index == -1:
            from_str = sql[from_index + 6:].strip()
            from_str = from_str[:-1].strip() if from_str[-1] == ";" else from_str
            where_str = ""
        else:
            from_str = sql[from_index + 6: where_index].strip()
            where_str = sql[where_index + 7:].strip()
            where_str = where_str[:-1].strip() if where_str[-1] == ";" else where_str

        for alias_str in from_str.strip().split(','):
            table_alias =  alias_str.strip().split()
            assert(table_alias[0] not in alias)
            if len(table_alias) == 1:
                tables.append(table_alias[0])
                alias[table_alias[0]] = table_alias[0]
                alias_to_name[table_alias[0]] = table_alias[0]
            elif len(table_alias) == 2:
                tables.append(table_alias[0])
                alias[table_alias[0]] = table_alias[1]
                alias_to_name[table_alias[1]] = table_alias[0]
            else:
                tables.append(table_alias[0])
                alias[table_alias[0]] = table_alias[2]
                alias_to_name[table_alias[2]] = table_alias[0]

        if where_str != "":
            for predicate_str in SQLQuery._split_predicate(where_str):
                predicate_str = predicate_str.strip()
                if predicate_str[0] == '(' and predicate_str[-1] == ')':
                    predicate_str = predicate_str[1:-1].strip()
                
                first_dot_index = predicate_str.find('.')
                
                opertor_index = -1
                for i, c in enumerate(predicate_str):
                    if c in "<=>":
                        opertor_index = i
                        break
                
                left_table = alias_to_name[predicate_str[:first_dot_index].strip()]
                left_column = predicate_str[first_dot_index + 1:opertor_index].strip()
                operator = predicate_str[opertor_index]

                right_operand_index = opertor_index + 1
                if opertor_index + 1 < len(predicate_str) and predicate_str[opertor_index + 1] in "<=>":
                    right_operand_index = opertor_index + 2
                    operator = predicate_str[opertor_index: opertor_index + 2]

                right_str = predicate_str[right_operand_index:].strip()
                if re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z0-9_]*$", right_str):
                    right_table, right_column = right_str.split('.')
                    right_table = alias_to_name[right_table]
                    if (left_table, left_column) < (right_table, right_column):
                        join_conditions.append((left_table, left_column, right_table, right_column))
                    else:
                        join_conditions.append((right_table, right_column, left_table, left_column))
                else:
                    predicates.append((left_table, left_column, operator, right_str))

        return SQLQuery(tables, alias, join_conditions, predicates)

    @staticmethod
    def get_possible_joins_and_predicates(sqls: List[str], result_file_path: str = None) -> None:
        tables = set()
        alias = {}
        column_operators: Dict[Tuple[str, str], Set[str]] = {}
        join_conditions: Set[Tuple[str, str, str, str]] = set()
        max_predicate_num_per_table: Dict[str, int] = {}

        for sql in sqls:
            query = SQLQuery.from_sql(sql)

            alias.update(query._alias)

            for table, column, right_table, right_column in query._join_conditions:
                join_conditions.add((table, column, right_table, right_column))
                tables.add(table)
                tables.add(right_table)

            predicate_column_per_table = {}

            for table, column, operator, _ in query._predicates:
                if (table, column) not in column_operators:
                    column_operators[(table, column)] = set()
                column_operators[(table, column)].add(operator)
                
                if table not in predicate_column_per_table:
                    predicate_column_per_table[table] = set()
                predicate_column_per_table[table].add(column)
            
            for table, columns in predicate_column_per_table.items():
                if table not in max_predicate_num_per_table:
                    max_predicate_num_per_table[table] = 0
                max_predicate_num_per_table[table] = max(max_predicate_num_per_table[table], len(columns))

        print(f"Tables: {len(tables)}")
        print(sorted(list(tables)))
        print(', '.join(sorted(list(tables))))
        print(alias)
        print("Possible Join Conditions: ")
        for table, column, right_table, right_column in sorted(join_conditions):
            print(f"\t{table}.{column} = {right_table}.{right_column}")

        print("Possible Predicates: ")
        for (table, column), operators in sorted(column_operators.items()):
            print(f"\t{table}.{column} {operators}")

        print("Maximum Predicates per Table: ")
        for table, max_num in sorted(max_predicate_num_per_table.items()):
            print(f"\t{table}: {max_num}")
        print("Maximum Predicates Num: ", max(max_predicate_num_per_table.values()))

        # Save to JSON if result_file_path is provided
        if result_file_path is not None:
            result = {
                "tables": sorted(list(tables)),
                "join_conditions": [
                    f"{t1}.{c1}={t2}.{c2}"
                    for t1, c1, t2, c2 in sorted(join_conditions)
                ],
                "predicates": {
                    f"{t}.{c}": sorted(list(ops))
                    for (t, c), ops in sorted(column_operators.items())
                },
            }
            with open(result_file_path, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\nResult saved to {result_file_path}")

    @staticmethod
    def _split_predicate(predicate_str: str) -> List[str]:
        n = len(predicate_str)
        result = []
        in_quote = False    
        i, j = 0, 0
        while j < n:
            if predicate_str[j] == "'":
                if in_quote:
                    if (j + 1 < n and predicate_str[j+1] == "'"):
                        j += 2
                    else:
                        in_quote = False
                        j += 1
                else:
                    in_quote = True
                    j += 1
            elif in_quote:
                j += 1
            else:
                if predicate_str[j: j + 5].lower() == " and ":    
                    result.append(predicate_str[i:j].strip())
                    i = j + 5
                    j = i
                else:
                    j += 1
        result.append(predicate_str[i:].strip())      
        return result      

if __name__ == "__main__":
    workload_file_path = "workloads/job_large_ours/job_large_ours.sql"
    result_file_path = "workloads/job_large_ours/predicates.json"

    column_operators: Dict[Tuple[str, str], Set[str]] = {}

    with open(workload_file_path, 'r') as f:
        sqls = [l.split("||")[0] for l in f.readlines() if "||0||None" not in l]
    print(sqls)

    SQLQuery.get_possible_joins_and_predicates(sqls, result_file_path)
