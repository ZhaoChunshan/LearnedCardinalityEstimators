-- Table: movie_companies
CREATE STATISTICS stats_movie_companies_company_id_company_type_id ON company_id, company_type_id FROM movie_companies;
ANALYZE movie_companies;

-- Table: title
CREATE STATISTICS stats_title_kind_id_production_year ON kind_id, production_year FROM title;
ANALYZE title;
