import psycopg2
from plan_generator.connection_args import DB_CONNECTION_ARGS
from typing import List, Dict, Any

def get_db_indexes_info(db: str) -> Dict[str, Dict[str, List[str]]]:
    user, password, host, port = DB_CONNECTION_ARGS["user"], DB_CONNECTION_ARGS["password"], DB_CONNECTION_ARGS["host"], DB_CONNECTION_ARGS["port"]
    conn = psycopg2.connect(
        dbname=db,
        user=user,
        password=password,
        host=host,
        port=port
    )
    result_dict = {}
    cursor = conn.cursor()
    sql = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' AND table_type = 'BASE TABLE';"
    cursor.execute(sql)
    tables = cursor.fetchall()
    for table in tables:
        table_name = table[0]
        sql = f"SELECT indexname, indexdef FROM pg_indexes WHERE tablename = '{table_name}';"
        cursor.execute(sql)
        indexes = cursor.fetchall()
        index_name_to_columns = {}
        for row in indexes:
            index_name: str = row[0]
            index_def: str = row[1]
            index_columns = index_def[index_def.index("(") + 1:index_def.index(")")].replace(" ", "").split(",")
            index_name_to_columns[index_name] = index_columns
        result_dict[table_name] = index_name_to_columns
    cursor.close()
    conn.close()
    return result_dict
