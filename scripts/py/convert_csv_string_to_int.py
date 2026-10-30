import polars as pl
import pickle
from pathlib import Path

table_path_template = Path("./datasets/imdb/{}.csv")

imdb_m_str_columns = [
    "cast_info.note",
    "comp_cast_type.kind",
    "company_name.country_code",
    "company_name.name",
    "company_type.kind",
    "info_type.info",
    "keyword.keyword",
    "kind_type.kind",
    "movie_info.info",
    "movie_info_idx.info",
    "title.title"
]

imdb_large_str_columns = [
    "aka_name.name_pcode_cf",
    "aka_title.md5sum",
    "aka_title.note",
    "aka_title.phonetic_code",
    "aka_title.title",
    "cast_info.note",
    "char_name.name_pcode_nf",
    "comp_cast_type.kind",
    "company_name.country_code",
    "company_name.md5sum",
    "company_name.name",
    "company_name.name_pcode_nf",
    "company_name.name_pcode_sf",
    "company_type.kind",
    "info_type.info",
    "keyword.phonetic_code",
    "movie_companies.note",
    "movie_info.info",
    "movie_info_idx.info",
    "name.gender",
    "title.md5sum",
    "title.phonetic_code",
    "title.title"
]

def encode_str_columns(is_imdb_m: bool):
    columns = imdb_m_str_columns if is_imdb_m else imdb_large_str_columns
    table_columns = {}
    for column in columns:
        table_name, column_name = column.split(".")
        table_columns.setdefault(table_name, []).append(column_name)

    encoded_dict = {}
    for table, cols in table_columns.items():
        table_path = table_path_template.with_name(f"{table}.csv")
        new_table_path = table_path_template.with_name(f"{table}_4_imdb_m.csv" if is_imdb_m else f"{table}_4_imdb_large.csv")

        df = pl.read_csv(table_path, separator="|", null_values=["", "NaN"])

        for col in cols:
            uniques = df[col].drop_nulls().unique().sort()

            value_2_id = {v: i for i, v in enumerate(uniques.to_list())}
            encoded_dict[f"{table}.{col}"] = value_2_id

            df = df.with_columns(
                df[col].replace(value_2_id, default=None).alias(col)
            )

        df.write_csv(new_table_path, separator="|")

    rows = []
    for key, mapping in encoded_dict.items():
        table, col = key.split(".")
        for value, idx in mapping.items():
            rows.append((table, col, value, idx))

    df = pl.DataFrame(rows, schema=["table", "column", "value", "id"])
    df.write_parquet("./datasets/imdb/encode_dict_4_m.parquet" if is_imdb_m else "./datasets/imdb/encode_dict_4_large.parquet")

if __name__ == "__main__":
    encode_str_columns(True)
    encode_str_columns(False)
