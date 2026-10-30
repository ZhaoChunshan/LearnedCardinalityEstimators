DO $$
DECLARE
    r RECORD;
BEGIN
    FOR r IN
        SELECT n.nspname AS schema_name,
               s.stxname AS statistics_name
        FROM pg_statistic_ext s
        JOIN pg_namespace n
          ON n.oid = s.stxnamespace
    LOOP
        EXECUTE format(
            'DROP STATISTICS %I.%I',
            r.schema_name,
            r.statistics_name
        );
    END LOOP;
END $$;
