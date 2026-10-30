import psycopg2


from typing import Optional, Dict
from schema.comparison_operator import ColumnType
from schema.sql_schema import SQLSchema


def get_connection(database: str, user: str, password: str, host: str = "localhost", port: Optional[int] = None) -> psycopg2.extensions.connection:
    if port is None:
        port = 5432
    return psycopg2.connect(host=host, database=database, user=user, password=password, port=port)


def imdb_schema(username: str, password: str, database_name: str, host: str = "localhost", port: Optional[int] = None, sample_size: Optional[int] = None) -> SQLSchema:
    connection: psycopg2.extensions.connection = get_connection(database_name, username, password, host, port=port)
    
    mask={"title", "movie_info", "movie_info_idx", "movie_companies", "movie_keyword", "cast_info"}
    
    alias = {
        "title": "t",
        "movie_info": "mi",
        "movie_info_idx": "mi_idx",
        "movie_companies": "mc",
        "movie_keyword": "mk",
        "cast_info": "ci"
    }

    table_column_types: Dict[str, Dict[str, ColumnType]] = {
        "title": {
            "kind_id": ColumnType.CATEGORICAL,
            "production_year": ColumnType.NUMERICAL,
        },
        "cast_info": {
            "role_id": ColumnType.CATEGORICAL,
        },
        "movie_companies": {
            "company_id": ColumnType.CATEGORICAL,
            "company_type_id": ColumnType.CATEGORICAL,
        },
        "movie_info": {
            "info_type_id": ColumnType.CATEGORICAL,
        },
        "movie_info_idx": {
            "info_type_id": ColumnType.CATEGORICAL,
        },
        "movie_keyword": {
            "keyword_id": ColumnType.CATEGORICAL,
        }
    }
    
    schema = SQLSchema.from_connection("imdb", connection, table_column_types, mask, alias=alias, sample_size=sample_size)
    return schema

def imdb_m_schema(username: str, password: str, database_name: str, host: str = "localhost", port: Optional[int] = None, sample_size: Optional[int] = None) -> SQLSchema:
    """
        An IMDB schema contains 16 tables to support JOB-M queries.
    """
    connection: psycopg2.extensions.connection = get_connection(database_name, username, password, host, port=port)
    
    mask = {'aka_title', 'cast_info', 'comp_cast_type', 'company_name', 'company_type', 'complete_cast',
            'info_type', 'keyword', 'kind_type', 'link_type', 'movie_companies', 'movie_info', 
            'movie_info_idx', 'movie_keyword', 'movie_link', 'title'}
    
    alias = {'movie_companies': 'mc', 
             'info_type': 'it', 
             'title': 't', 
             'movie_info_idx': 'mi_idx', 
             'company_type': 'ct', 
             'cast_info': 'ci', 
             'movie_keyword': 'mk', 
             'keyword': 'k', 
             'company_name': 'cn', 
             'movie_link': 'ml', 
             'link_type': 'lt', 
             'movie_info': 'mi', 
             'aka_title': 'at', 
             'complete_cast': 'cc', 
             'comp_cast_type': 'cct', 
             'kind_type': 'kt'}

    table_column_types: Dict[str, Dict[str, ColumnType]] = {
        "title": {
            "title": ColumnType.CATEGORICAL,
            "production_year": ColumnType.NUMERICAL,
            "episode_nr": ColumnType.NUMERICAL,
        },
        
        "cast_info": {
            "note": ColumnType.CATEGORICAL,
        },
        "comp_cast_type": {
            "kind": ColumnType.CATEGORICAL,
        },
        "company_name": {
            "country_code": ColumnType.CATEGORICAL,
            "name": ColumnType.CATEGORICAL, 
        },
        "company_type": {
            "kind": ColumnType.CATEGORICAL,
        },
        "info_type": {
            "info": ColumnType.CATEGORICAL,    
        },
        "keyword": {
            "keyword": ColumnType.CATEGORICAL,
        },
        "kind_type": {
            "kind": ColumnType.CATEGORICAL,
        },
        "movie_info": {
            "info": ColumnType.CATEGORICAL,
        },
        "movie_info_idx": {
            "info": ColumnType.STRING_WITH_CMP,
            "movie_id": ColumnType.CATEGORICAL,
        },
    }
    
    schema = SQLSchema.from_connection("imdb_m", connection, table_column_types, mask, alias=alias, sample_size=sample_size)
    return schema

def imdb_large_schema(username: str, password: str, database_name: str, host: str = "localhost", port: Optional[int] = None, sample_size: Optional[int] = None) -> SQLSchema:
    """
        Full IMDB schema contains 21 tables to support JOB-Large queries.
    """
    connection: psycopg2.extensions.connection = get_connection(database_name, username, password, host, port=port)
    
    mask = {'aka_name', 'aka_title', 'cast_info', 'char_name', 'comp_cast_type', 'company_name', 'company_type', 
            'complete_cast', 'info_type', 'keyword', 'kind_type', 'link_type', 'movie_companies', 'movie_info', 
            'movie_info_idx', 'movie_keyword', 'movie_link', 'name', 'person_info', 'role_type', 'title'}
    
    alias = {'aka_title': 'at', 
             'movie_info': 'mi', 
             'title': 't', 
             'company_name': 'cn', 
             'movie_companies': 'mc', 
             'comp_cast_type': 'cct', 
             'company_type': 'ct', 
             'complete_cast': 'cc', 
             'keyword': 'k', 
             'kind_type': 'kt', 
             'movie_info_idx': 'mi_idx', 
             'movie_keyword': 'mk', 
             'cast_info': 'ci', 
             'char_name': 'ch_n', 
             'info_type': 'it', 
             'aka_name': 'an', 
             'name': 'n', 
             'role_type': 'rt', 
             'person_info': 'pi', 
             'link_type': 'lt', 
             'movie_link': 'ml'}

    table_column_types: Dict[str, Dict[str, ColumnType]] = {
        "title": {
            "title": ColumnType.CATEGORICAL,
            "production_year": ColumnType.NUMERICAL,
            "phonetic_code": ColumnType.CATEGORICAL,
            "md5sum": ColumnType.CATEGORICAL,
            "kind_id": ColumnType.NUMERICAL,
            "id": ColumnType.NUMERICAL,
        },
        
        "cast_info": {
            "id": ColumnType.NUMERICAL,
            "movie_id": ColumnType.NUMERICAL,
            "note": ColumnType.CATEGORICAL,
            "nr_order": ColumnType.NUMERICAL,
            "person_id": ColumnType.NUMERICAL,
            "person_role_id": ColumnType.NUMERICAL,
            "role_id": ColumnType.NUMERICAL,
        },
        "comp_cast_type": {
            "id": ColumnType.NUMERICAL,
            "kind": ColumnType.CATEGORICAL,
        },
        "company_name": {
            "id": ColumnType.NUMERICAL,
            "country_code": ColumnType.CATEGORICAL,
            "name": ColumnType.CATEGORICAL, 
            "md5sum": ColumnType.CATEGORICAL,
            "name_pcode_nf": ColumnType.CATEGORICAL,
            "name_pcode_sf": ColumnType.CATEGORICAL,
        },
        "company_type": {
            "id": ColumnType.NUMERICAL,
            "kind": ColumnType.CATEGORICAL,
        },
        "info_type": {
            "id": ColumnType.NUMERICAL,
            "info": ColumnType.CATEGORICAL,    
        },
        "keyword": {
            "id": ColumnType.NUMERICAL,
            "phonetic_code": ColumnType.CATEGORICAL,
        },
        "kind_type": {
            "id": ColumnType.NUMERICAL,
        },
        "movie_info": {
            "id": ColumnType.NUMERICAL,
            "info": ColumnType.CATEGORICAL,
            "info_type_id": ColumnType.NUMERICAL,
            "movie_id": ColumnType.NUMERICAL,
        },
        "movie_info_idx": {
            "id": ColumnType.NUMERICAL,
            "info": ColumnType.CATEGORICAL,
            "info_type_id": ColumnType.NUMERICAL,
            "movie_id": ColumnType.NUMERICAL,
        },
        'name': {
            'id': ColumnType.NUMERICAL,
            'gender': ColumnType.CATEGORICAL,
        },
        'person_info': {
            'id': ColumnType.NUMERICAL,
            'info_type_id': ColumnType.NUMERICAL,
            'person_id': ColumnType.NUMERICAL,
        }, 
        'role_type': {
            'id': ColumnType.NUMERICAL,
        }, 
        'char_name': {
            'id': ColumnType.NUMERICAL,
            'name_pcode_nf': ColumnType.CATEGORICAL,
        }, 
        'aka_name': {
            'id': ColumnType.NUMERICAL,
            'name_pcode_cf': ColumnType.CATEGORICAL,
            'person_id': ColumnType.NUMERICAL,
        },
        'aka_title': {
            'id': ColumnType.NUMERICAL,
            'kind_id': ColumnType.NUMERICAL,
            'md5sum': ColumnType.CATEGORICAL,
            'movie_id': ColumnType.NUMERICAL,
            'note': ColumnType.CATEGORICAL,
            'phonetic_code': ColumnType.CATEGORICAL,
            'production_year': ColumnType.NUMERICAL,
            'title': ColumnType.CATEGORICAL,
        },
        'complete_cast': {
            'id': ColumnType.NUMERICAL,
            'movie_id': ColumnType.NUMERICAL,
            'status_id': ColumnType.NUMERICAL,
            'subject_id': ColumnType.NUMERICAL,
        },
        "movie_companies": {
            "company_id": ColumnType.NUMERICAL,
            "company_type_id": ColumnType.NUMERICAL,
            "id": ColumnType.NUMERICAL,
            "movie_id": ColumnType.NUMERICAL,
            "note": ColumnType.CATEGORICAL,
        },
        "movie_keyword": {
            "id": ColumnType.NUMERICAL,
            "keyword_id": ColumnType.NUMERICAL,
            "movie_id": ColumnType.NUMERICAL,
        }
    }
    
    schema = SQLSchema.from_connection("imdb_large", connection, table_column_types, mask, alias=alias, sample_size=sample_size)
    return schema


def stats_schema(username: str, password: str, database_name: str, host: str = "localhost", port: Optional[int] = None, sample_size: Optional[int] = None) -> SQLSchema:
    connection: psycopg2.extensions.connection = get_connection(database_name, username, password, host, port=port)

    alias = {
        "badges": "b",
        "comments": "c",
        "posthistory": "ph",
        "postlinks": "pl",
        "posts": "p",
        "tags": "t",
        "users": "u",
        "votes": "v"
    }

    table_column_types = {
        "badges": {
            "date": ColumnType.NUMERICAL,
        },
        "comments": {
            "creationdate": ColumnType.NUMERICAL,
            "score": ColumnType.CATEGORICAL,
        },
        "posthistory": {
            "creationdate": ColumnType.NUMERICAL,
            "posthistorytypeid": ColumnType.CATEGORICAL
        },
        "postlinks": {
            "creationdate": ColumnType.NUMERICAL,
            "linktypeid": ColumnType.CATEGORICAL
        },
        "posts": {
            "answercount": ColumnType.NUMERICAL,
            "commentcount": ColumnType.NUMERICAL,
            "creationdate": ColumnType.NUMERICAL,
            "favoritecount": ColumnType.NUMERICAL,
            "posttypeid": ColumnType.CATEGORICAL,
            "score": ColumnType.NUMERICAL,
            "viewcount": ColumnType.NUMERICAL
        },
        "tags": {

        },
        "users": {
            "creationdate": ColumnType.NUMERICAL,
            "downvotes": ColumnType.NUMERICAL,
            "reputation": ColumnType.NUMERICAL,
            "upvotes": ColumnType.NUMERICAL,
            "views": ColumnType.NUMERICAL
        },
        "votes": {
            "bountyamount": ColumnType.NUMERICAL,
            "creationdate": ColumnType.NUMERICAL,
            "votetypeid": ColumnType.CATEGORICAL
        }
    }


    return SQLSchema.from_connection(database_name, connection, table_column_types, alias=alias, sample_size=sample_size)

def ergastf1_schema(username: str, password: str, database_name: str, host: str = "localhost", port: Optional[int] = None, sample_size: Optional[int] = None) -> SQLSchema:
    connection: psycopg2.extensions.connection = get_connection(database_name, username, password, host, port=port)

    alias = {
        "circuits": "erg_cir",
        "constructors": "erg_con",
        "drivers": "erg_driv",
        "seasons": "erg_sea",
        "races": "erg_rac",
        "constructorresults": "erg_conres",
        "constructorstandings": "erg_consta",
        "driverstandings": "erg_drivsta",
        "laptimes": "erg_lap",
        "pitstops": "erg_pit",
        "qualifying": "erg_qua",
        "status": "erg_sta",
        "results": "erg_res",
        "target": "erg_tar",
    }


    table_column_types = {
        "circuits": {

        },
        "constructorresults": {
            "constructorid": ColumnType.NUMERICAL,
            "constructorresultsid": ColumnType.NUMERICAL,
            "points": ColumnType.CATEGORICAL,
            "raceid": ColumnType.NUMERICAL,
        },
        "constructors": {

        },
        "constructorstandings": {
            "constructorid": ColumnType.NUMERICAL,
            "constructorstandingsid": ColumnType.NUMERICAL,
            "points": ColumnType.NUMERICAL,
            "position": ColumnType.CATEGORICAL,
            "positiontext": ColumnType.CATEGORICAL,
            "raceid": ColumnType.NUMERICAL,
            "wins": ColumnType.CATEGORICAL
        },
        "drivers": {
            "nationality": ColumnType.CATEGORICAL
        },
        "driverstandings": {
            "driverid": ColumnType.NUMERICAL,
            "driverstandingsid": ColumnType.NUMERICAL,
            "points": ColumnType.NUMERICAL,
            "position": ColumnType.NUMERICAL,
            "raceid": ColumnType.NUMERICAL,
            "wins": ColumnType.CATEGORICAL
        },
        "laptimes": {
            "driverid": ColumnType.NUMERICAL,
            "lap": ColumnType.CATEGORICAL,
            "milliseconds": ColumnType.NUMERICAL,
            "position": ColumnType.CATEGORICAL,
            "raceid": ColumnType.NUMERICAL
        },
        "pitstops": {
            "lap": ColumnType.CATEGORICAL,
            "milliseconds": ColumnType.NUMERICAL,
            "raceid": ColumnType.NUMERICAL,
            "stop": ColumnType.CATEGORICAL
        },
        "qualifying": {
            "constructorid": ColumnType.CATEGORICAL,
            "driverid": ColumnType.NUMERICAL,
            "number": ColumnType.CATEGORICAL,
            "position": ColumnType.CATEGORICAL,
            "qualifyid": ColumnType.NUMERICAL,
            "raceid": ColumnType.NUMERICAL
        },
        "races": {
            "circuitid": ColumnType.CATEGORICAL,
            "name": ColumnType.CATEGORICAL,
            "round": ColumnType.CATEGORICAL
        },
        "results": {
            "constructorid": ColumnType.NUMERICAL,
            "driverid": ColumnType.NUMERICAL,
            "fastestlap": ColumnType.CATEGORICAL,
            "grid": ColumnType.CATEGORICAL,
            "laps": ColumnType.NUMERICAL,
            "milliseconds": ColumnType.NUMERICAL,
            "number": ColumnType.NUMERICAL,
            "points": ColumnType.CATEGORICAL,
            "position": ColumnType.CATEGORICAL,
            "positionorder": ColumnType.CATEGORICAL,
            "positiontext": ColumnType.CATEGORICAL,
            "raceid": ColumnType.NUMERICAL,
            "rank": ColumnType.CATEGORICAL,
            "resultid": ColumnType.NUMERICAL,
            "statusid": ColumnType.NUMERICAL
        },
        "seasons": {
        
        },
        "status": {
        
        },
        "target": {
            "driverid": ColumnType.NUMERICAL,
            "raceid": ColumnType.NUMERICAL,
            "targetid": ColumnType.NUMERICAL,
            "win": ColumnType.CATEGORICAL
        }
    }

    return SQLSchema.from_connection("ergastf1", connection, table_column_types, alias=alias, sample_size=sample_size)


def get_schema(schema_name: str, user: str, password: str, host: str = "localhost", port: int = 5432, sample_size: Optional[int] = None) -> SQLSchema:
    if schema_name == "imdb":
        schema = imdb_schema(username=user, password=password, database_name="imdb", host=host, port=port, sample_size=sample_size)
    elif schema_name == "imdb_m":
        schema = imdb_m_schema(username=user, password=password, database_name="imdb", host=host, port=port, sample_size=sample_size)
    elif schema_name == "imdb_large":
        schema = imdb_large_schema(username=user, password=password, database_name="imdb", host=host, port=port, sample_size=sample_size)
    elif schema_name == "stats":
        schema = stats_schema(username=user, password=password, database_name=schema_name, host=host, port=port, sample_size=sample_size)
    elif schema_name == "ergastf1":
        schema = ergastf1_schema(username=user, password=password, database_name=schema_name, host=host, port=port, sample_size=sample_size)

    return schema