-- Table: aka_name
CREATE STATISTICS stats_aka_name_id_person_id ON id, person_id FROM aka_name;
ANALYZE aka_name;

-- Table: aka_title
CREATE STATISTICS stats_aka_title_id_movie_id ON id, movie_id FROM aka_title;
CREATE STATISTICS stats_aka_title_kind_id_movie_id ON kind_id, movie_id FROM aka_title;
CREATE STATISTICS stats_aka_title_id_kind_id ON id, kind_id FROM aka_title;
CREATE STATISTICS stats_aka_title_id_note ON id, note FROM aka_title;
CREATE STATISTICS stats_aka_title_kind_id_production_year ON kind_id, production_year FROM aka_title;
CREATE STATISTICS stats_aka_title_note_production_year ON note, production_year FROM aka_title;
CREATE STATISTICS stats_aka_title_id_kind_id_movie_id ON id, kind_id, movie_id FROM aka_title;
CREATE STATISTICS stats_aka_title_id_movie_id_note ON id, movie_id, note FROM aka_title;
CREATE STATISTICS stats_aka_title_id_kind_id_note ON id, kind_id, note FROM aka_title;
CREATE STATISTICS stats_aka_title_id_kind_id_production_year ON id, kind_id, production_year FROM aka_title;
CREATE STATISTICS stats_aka_title_kind_id_movie_id_production_year ON kind_id, movie_id, production_year FROM aka_title;
CREATE STATISTICS stats_aka_title_id_note_production_year ON id, note, production_year FROM aka_title;
CREATE STATISTICS stats_aka_title_kind_id_note_production_year ON kind_id, note, production_year FROM aka_title;
CREATE STATISTICS stats_aka_title_id_kind_id_movie_id_note ON id, kind_id, movie_id, note FROM aka_title;
CREATE STATISTICS stats_aka_title_id_kind_id_movie_id_production_year ON id, kind_id, movie_id, production_year FROM aka_title;
CREATE STATISTICS stats_aka_title_id_kind_id_note_production_year ON id, kind_id, note, production_year FROM aka_title;
CREATE STATISTICS stats_aka_title_id_movie_id_note_production_year ON id, movie_id, note, production_year FROM aka_title;
CREATE STATISTICS stats_aka_title_kind_id_movie_id_note_production_year ON kind_id, movie_id, note, production_year FROM aka_title;
CREATE STATISTICS stats_aka_title_id_kind_id_movie_id_note_production_year ON id, kind_id, movie_id, note, production_year FROM aka_title;
ANALYZE aka_title;

-- Table: cast_info
CREATE STATISTICS stats_cast_info_id_role_id ON id, role_id FROM cast_info;
CREATE STATISTICS stats_cast_info_id_person_id ON id, person_id FROM cast_info;
CREATE STATISTICS stats_cast_info_person_role_id_role_id ON person_role_id, role_id FROM cast_info;
CREATE STATISTICS stats_cast_info_id_person_role_id ON id, person_role_id FROM cast_info;
CREATE STATISTICS stats_cast_info_person_id_role_id ON person_id, role_id FROM cast_info;
CREATE STATISTICS stats_cast_info_note_role_id ON note, role_id FROM cast_info;
CREATE STATISTICS stats_cast_info_person_id_person_role_id ON person_id, person_role_id FROM cast_info;
CREATE STATISTICS stats_cast_info_id_note ON id, note FROM cast_info;
CREATE STATISTICS stats_cast_info_nr_order_person_role_id ON nr_order, person_role_id FROM cast_info;
CREATE STATISTICS stats_cast_info_nr_order_role_id ON nr_order, role_id FROM cast_info;
CREATE STATISTICS stats_cast_info_id_nr_order ON id, nr_order FROM cast_info;
CREATE STATISTICS stats_cast_info_note_person_role_id ON note, person_role_id FROM cast_info;
CREATE STATISTICS stats_cast_info_nr_order_person_id ON nr_order, person_id FROM cast_info;
CREATE STATISTICS stats_cast_info_id_person_role_id_role_id ON id, person_role_id, role_id FROM cast_info;
CREATE STATISTICS stats_cast_info_id_person_id_role_id ON id, person_id, role_id FROM cast_info;
CREATE STATISTICS stats_cast_info_id_person_id_person_role_id ON id, person_id, person_role_id FROM cast_info;
CREATE STATISTICS stats_cast_info_id_note_role_id ON id, note, role_id FROM cast_info;
CREATE STATISTICS stats_cast_info_person_id_person_role_id_role_id ON person_id, person_role_id, role_id FROM cast_info;
CREATE STATISTICS stats_cast_info_id_nr_order_role_id ON id, nr_order, role_id FROM cast_info;
CREATE STATISTICS stats_cast_info_nr_order_person_role_id_role_id ON nr_order, person_role_id, role_id FROM cast_info;
CREATE STATISTICS stats_cast_info_note_person_role_id_role_id ON note, person_role_id, role_id FROM cast_info;
CREATE STATISTICS stats_cast_info_id_nr_order_person_role_id ON id, nr_order, person_role_id FROM cast_info;
CREATE STATISTICS stats_cast_info_id_note_person_role_id ON id, note, person_role_id FROM cast_info;
CREATE STATISTICS stats_cast_info_id_nr_order_person_id ON id, nr_order, person_id FROM cast_info;
CREATE STATISTICS stats_cast_info_id_note_person_id ON id, note, person_id FROM cast_info;
CREATE STATISTICS stats_cast_info_note_person_id_role_id ON note, person_id, role_id FROM cast_info;
CREATE STATISTICS stats_cast_info_nr_order_person_id_role_id ON nr_order, person_id, role_id FROM cast_info;
CREATE STATISTICS stats_cast_info_nr_order_person_id_person_role_id ON nr_order, person_id, person_role_id FROM cast_info;
CREATE STATISTICS stats_cast_info_note_nr_order_role_id ON note, nr_order, role_id FROM cast_info;
CREATE STATISTICS stats_cast_info_id_note_nr_order ON id, note, nr_order FROM cast_info;
CREATE STATISTICS stats_cast_info_note_person_id_person_role_id ON note, person_id, person_role_id FROM cast_info;
CREATE STATISTICS stats_cast_info_note_nr_order_person_role_id ON note, nr_order, person_role_id FROM cast_info;
CREATE STATISTICS stats_cast_info_id_person_id_person_role_id_role_id ON id, person_id, person_role_id, role_id FROM cast_info;
CREATE STATISTICS stats_cast_info_id_note_person_role_id_role_id ON id, note, person_role_id, role_id FROM cast_info;
CREATE STATISTICS stats_cast_info_id_nr_order_person_role_id_role_id ON id, nr_order, person_role_id, role_id FROM cast_info;
CREATE STATISTICS stats_cast_info_id_note_person_id_role_id ON id, note, person_id, role_id FROM cast_info;
CREATE STATISTICS stats_cast_info_id_nr_order_person_id_role_id ON id, nr_order, person_id, role_id FROM cast_info;
CREATE STATISTICS stats_cast_info_id_nr_order_person_id_person_role_id ON id, nr_order, person_id, person_role_id FROM cast_info;
CREATE STATISTICS stats_cast_info_nr_order_person_id_person_role_id_role_id ON nr_order, person_id, person_role_id, role_id FROM cast_info;
CREATE STATISTICS stats_cast_info_note_person_id_person_role_id_role_id ON note, person_id, person_role_id, role_id FROM cast_info;
CREATE STATISTICS stats_cast_info_id_note_nr_order_role_id ON id, note, nr_order, role_id FROM cast_info;
CREATE STATISTICS stats_cast_info_id_note_person_id_person_role_id ON id, note, person_id, person_role_id FROM cast_info;
CREATE STATISTICS stats_cast_info_note_nr_order_person_role_id_role_id ON note, nr_order, person_role_id, role_id FROM cast_info;
CREATE STATISTICS stats_cast_info_id_note_nr_order_person_role_id ON id, note, nr_order, person_role_id FROM cast_info;
CREATE STATISTICS stats_cast_info_note_nr_order_person_id_role_id ON note, nr_order, person_id, role_id FROM cast_info;
CREATE STATISTICS stats_cast_info_id_note_nr_order_person_id ON id, note, nr_order, person_id FROM cast_info;
CREATE STATISTICS stats_cast_info_note_nr_order_person_id_person_role_id ON note, nr_order, person_id, person_role_id FROM cast_info;
CREATE STATISTICS stats_cast_info_id_nr_order_person_id_person_role_id_role_id ON id, nr_order, person_id, person_role_id, role_id FROM cast_info;
CREATE STATISTICS stats_cast_info_id_note_person_id_person_role_id_role_id ON id, note, person_id, person_role_id, role_id FROM cast_info;
CREATE STATISTICS stats_cast_info_id_note_nr_order_person_role_id_role_id ON id, note, nr_order, person_role_id, role_id FROM cast_info;
CREATE STATISTICS stats_cast_info_id_note_nr_order_person_id_role_id ON id, note, nr_order, person_id, role_id FROM cast_info;
CREATE STATISTICS stats_cast_info_note_nr_order_person_id_person_role_id_role_id ON note, nr_order, person_id, person_role_id, role_id FROM cast_info;
CREATE STATISTICS stats_cast_info_id_note_nr_order_person_id_person_role_id ON id, note, nr_order, person_id, person_role_id FROM cast_info;
CREATE STATISTICS stats_cast_info_group_54 ON id, note, nr_order, person_id, person_role_id, role_id FROM cast_info;
ANALYZE cast_info;

-- Table: comp_cast_type
CREATE STATISTICS stats_comp_cast_type_id_kind ON id, kind FROM comp_cast_type;
ANALYZE comp_cast_type;

-- Table: company_name
CREATE STATISTICS stats_company_name_name_pcode_nf_name_pcode_sf ON name_pcode_nf, name_pcode_sf FROM company_name;
ANALYZE company_name;

-- Table: company_type
CREATE STATISTICS stats_company_type_id_kind ON id, kind FROM company_type;
ANALYZE company_type;

-- Table: complete_cast
CREATE STATISTICS stats_complete_cast_id_subject_id ON id, subject_id FROM complete_cast;
CREATE STATISTICS stats_complete_cast_id_movie_id ON id, movie_id FROM complete_cast;
CREATE STATISTICS stats_complete_cast_movie_id_status_id ON movie_id, status_id FROM complete_cast;
CREATE STATISTICS stats_complete_cast_id_movie_id_subject_id ON id, movie_id, subject_id FROM complete_cast;
CREATE STATISTICS stats_complete_cast_id_movie_id_status_id ON id, movie_id, status_id FROM complete_cast;
CREATE STATISTICS stats_complete_cast_id_movie_id_status_id_subject_id ON id, movie_id, status_id, subject_id FROM complete_cast;
ANALYZE complete_cast;

-- Table: info_type
CREATE STATISTICS stats_info_type_id_info ON id, info FROM info_type;
ANALYZE info_type;

-- Table: movie_companies
CREATE STATISTICS stats_movie_companies_id_movie_id ON id, movie_id FROM movie_companies;
CREATE STATISTICS stats_movie_companies_company_type_id_id ON company_type_id, id FROM movie_companies;
CREATE STATISTICS stats_movie_companies_company_id_id ON company_id, id FROM movie_companies;
CREATE STATISTICS stats_movie_companies_company_id_company_type_id ON company_id, company_type_id FROM movie_companies;
CREATE STATISTICS stats_movie_companies_company_id_movie_id ON company_id, movie_id FROM movie_companies;
CREATE STATISTICS stats_movie_companies_company_id_company_type_id_id ON company_id, company_type_id, id FROM movie_companies;
CREATE STATISTICS stats_movie_companies_company_id_id_movie_id ON company_id, id, movie_id FROM movie_companies;
CREATE STATISTICS stats_movie_companies_company_type_id_id_movie_id ON company_type_id, id, movie_id FROM movie_companies;
CREATE STATISTICS stats_movie_companies_company_id_company_type_id_movie_id ON company_id, company_type_id, movie_id FROM movie_companies;
CREATE STATISTICS stats_movie_companies_company_id_company_type_id_id_movie_id ON company_id, company_type_id, id, movie_id FROM movie_companies;
ANALYZE movie_companies;

-- Table: movie_info
CREATE STATISTICS stats_movie_info_id_info_type_id ON id, info_type_id FROM movie_info;
CREATE STATISTICS stats_movie_info_info_info_type_id ON info, info_type_id FROM movie_info;
CREATE STATISTICS stats_movie_info_id_info ON id, info FROM movie_info;
CREATE STATISTICS stats_movie_info_id_movie_id ON id, movie_id FROM movie_info;
CREATE STATISTICS stats_movie_info_id_info_info_type_id ON id, info, info_type_id FROM movie_info;
CREATE STATISTICS stats_movie_info_id_info_type_id_movie_id ON id, info_type_id, movie_id FROM movie_info;
CREATE STATISTICS stats_movie_info_id_info_movie_id ON id, info, movie_id FROM movie_info;
CREATE STATISTICS stats_movie_info_id_info_info_type_id_movie_id ON id, info, info_type_id, movie_id FROM movie_info;
ANALYZE movie_info;

-- Table: movie_info_idx
CREATE STATISTICS stats_movie_info_idx_id_movie_id ON id, movie_id FROM movie_info_idx;
CREATE STATISTICS stats_movie_info_idx_info_info_type_id ON info, info_type_id FROM movie_info_idx;
ANALYZE movie_info_idx;

-- Table: movie_keyword
CREATE STATISTICS stats_movie_keyword_id_movie_id ON id, movie_id FROM movie_keyword;
ANALYZE movie_keyword;

-- Table: name
CREATE STATISTICS stats_name_gender_id ON gender, id FROM name;
ANALYZE name;

-- Table: person_info
CREATE STATISTICS stats_person_info_id_person_id ON id, person_id FROM person_info;
ANALYZE person_info;

-- Table: title
CREATE STATISTICS stats_title_id_kind_id ON id, kind_id FROM title;
CREATE STATISTICS stats_title_kind_id_production_year ON kind_id, production_year FROM title;
CREATE STATISTICS stats_title_id_kind_id_production_year ON id, kind_id, production_year FROM title;
ANALYZE title;
