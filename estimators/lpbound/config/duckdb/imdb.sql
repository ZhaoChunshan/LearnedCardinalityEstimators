-- Adapted from the parent project datasets/imdb/postgres_create_imdb.sql
CREATE TABLE aka_name (
id integer not null,
    person_id integer not null,
    name character varying,
    imdb_index varchar(3),
    name_pcode_cf varchar(11),
    name_pcode_nf varchar(11),
    surname_pcode varchar(11),
    md5sum varchar(65)
);

CREATE TABLE aka_title (
id integer not null,
    movie_id integer,
    title character varying,
    imdb_index varchar(4),
    kind_id integer not null,
    production_year integer,
    phonetic_code varchar(5),
    episode_of_id integer,
    season_nr integer,
    episode_nr integer,
    note varchar(72),
    md5sum varchar(32)
);

CREATE TABLE cast_info (
id integer not null,
    person_id integer not null,
    movie_id integer not null,
    person_role_id integer,
    note character varying,
    nr_order integer,
    role_id integer not null
);

CREATE TABLE char_name (
id integer not null,
    name character varying,
    imdb_index varchar(2),
    imdb_id integer,
    name_pcode_nf varchar(5),
    surname_pcode varchar(5),
    md5sum varchar(32)
);

CREATE TABLE comp_cast_type (
id integer not null,
    kind varchar(32) not null
);

CREATE TABLE company_name (
id integer not null,
    name character varying,
    country_code varchar(6),
    imdb_id integer,
    name_pcode_nf varchar(5),
    name_pcode_sf varchar(5),
    md5sum varchar(32)
);

CREATE TABLE company_type (
id integer not null,
    kind varchar(32)
);

CREATE TABLE complete_cast (
id integer not null,
    movie_id integer,
    subject_id integer not null,
    status_id integer not null
);

CREATE TABLE info_type (
id integer not null,
    info varchar(32) not null
);

CREATE TABLE keyword (
id integer not null,
    keyword character varying not null,
    phonetic_code varchar(5)
);

CREATE TABLE kind_type (
id integer not null,
    kind varchar(15)
);

CREATE TABLE link_type (
id integer not null,
    link varchar(32) not null
);

CREATE TABLE movie_companies (
id integer not null,
    movie_id integer not null,
    company_id integer not null,
    company_type_id integer not null,
    note character varying
);

CREATE TABLE movie_info_idx (
id integer not null,
    movie_id integer not null,
    info_type_id integer not null,
    info character varying not null,
    note varchar(1)
);

CREATE TABLE movie_keyword (
id integer not null,
    movie_id integer not null,
    keyword_id integer not null
);

CREATE TABLE movie_link (
id integer not null,
    movie_id integer not null,
    linked_movie_id integer not null,
    link_type_id integer not null
);

CREATE TABLE name (
id integer not null,
    name character varying not null,
    imdb_index varchar(9),
    imdb_id integer,
    gender varchar(1),
    name_pcode_cf varchar(5),
    name_pcode_nf varchar(5),
    surname_pcode varchar(5),
    md5sum varchar(32)
);

CREATE TABLE role_type (
id integer not null,
    role varchar(32) not null
);

CREATE TABLE title (
id integer not null,
    title character varying,
    imdb_index varchar(5),
    kind_id integer not null,
    production_year integer,
    imdb_id integer,
    phonetic_code varchar(5),
    episode_of_id integer,
    season_nr integer,
    episode_nr integer,
    series_years varchar(49),
    md5sum varchar(32)
);

CREATE TABLE movie_info (
id integer not null,
    movie_id integer not null,
    info_type_id integer not null,
    info character varying,
    note character varying
);

CREATE TABLE person_info (
id integer not null,
    person_id integer not null,
    info_type_id integer not null,
    info character varying,
    note character varying
);
