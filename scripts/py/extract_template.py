import os
import sys
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(parent_dir)

from query.sql_query import SQLQuery, SQLTemplate
from schema.schemas import stats_schema, imdb_schema, ergastf1_schema, imdb_m_schema


def extract_templates_from_file(schema, query_file_path, keep_filter_table, keep_column, keep_operator):
    templates = []
    templates_id_dict = {}
    sql_group_by_templates = {}
    
    with open(query_file_path, 'r') as f:
        for sql in f:
            query = SQLQuery.parse(schema, sql.strip()).canonicalize_equal_join_keys()
            template = SQLTemplate.from_query(query, keep_filter_table=keep_filter_table, \
                keep_column=keep_column, keep_operator=keep_operator)
            assert(template.is_acyclic())
            assert(SQLQuery.parse(schema, sql.strip()).canonicalize()) # assert missed pk table not occurred in original query.
            if query.contains_fk_fk_joins():
                print("FK-FK join detected")
            template_str = template.sql()
            if template_str in templates_id_dict:
                template_id = templates_id_dict[template_str]
                sql_group_by_templates[template_id].append(sql.strip())
            else:
                template_id = len(templates)
                templates_id_dict[template_str] = template_id
                templates.append(template)
                sql_group_by_templates[template_id] = [sql.strip()]
    
    return templates, sql_group_by_templates


def write_templates_to_file(templates, output_file_location):
    with open(output_file_location, 'w') as f:
        for template in templates:
            f.write(template.sql() + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract Query Templates.')

    parser.add_argument('--query_set', type=str, choices=['job_light', 'stats_ceb', 'ergastf1', 'job_m_ours'], help='Workload name', required=True)
    parser.add_argument('--keep_filter_table', type=bool, default=True, help='Consider which tables are used in filter predicates')
    parser.add_argument('--keep_column', type=bool, default=True, help='Consider which columns are used in filter predicates')
    parser.add_argument('--keep_operator', type=bool, default=True, help='Consider the operators used in filtered columns')
    parser.add_argument('--output_file_location', type=str, help='Output file location for templates', required=True)

    args = parser.parse_args()
    if args.query_set == 'job_light':
        query_file_path = "workloads/job_light/job_light.sql"
        schema = imdb_schema(username="pilotscope", password="", database_name="imdb", host="localhost", port=5432)
    elif args.query_set == 'stats_ceb':
        query_file_path = "workloads/stats_ceb/stats_ceb.sql"
        schema = stats_schema(username="pilotscope", password="", database_name="stats", host="localhost", port=5432)
    elif args.query_set == 'ergastf1':
        query_file_path = "workloads/ergastf1/ergastf1_queries.sql"
        schema = ergastf1_schema(username="pilotscope", password="", database_name="ergastf1", host="localhost", port=5432)
    elif args.query_set == 'job_m_ours':
        query_file_path = "workloads/job_m_ours/job_m_ours.sql"
        schema = imdb_m_schema(username="pilotscope", password="", database_name="imdb", host="localhost", port=5432)
    
    templates, sql_group_by_templates = extract_templates_from_file(schema,
                                                                    query_file_path, 
                                                                    args.keep_filter_table, 
                                                                    args.keep_column,
                                                                    args.keep_operator,
                                                                    )

    write_templates_to_file(templates, args.output_file_location)
    
    for tid, template in enumerate(templates):
        print(f"Template {tid}: {template.sql()}\n")
        for sql in sql_group_by_templates[tid]:
            print(sql)
        print()
    