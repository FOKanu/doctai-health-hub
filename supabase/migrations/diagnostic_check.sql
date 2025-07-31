-- Detailed Diagnostic Check
-- Run this to see exactly what's missing from your table setup

-- =============================================================================
-- STEP 1: Check which tables actually exist
-- =============================================================================

SELECT
    'MISSING TABLES' as issue_type,
    table_name,
    '❌ NOT FOUND' as status
FROM (
    SELECT 'image_metadata' as table_name
    UNION ALL SELECT 'patient_timelines'
    UNION ALL SELECT 'health_metrics_timeseries'
    UNION ALL SELECT 'scan_sequences'
    UNION ALL SELECT 'risk_progressions'
) expected_tables
WHERE table_name NOT IN (
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema = 'public'
);

-- =============================================================================
-- STEP 2: Check which tables have RLS enabled
-- =============================================================================

SELECT
    'RLS DISABLED' as issue_type,
    tablename as table_name,
    '❌ RLS NOT ENABLED' as status
FROM pg_tables
WHERE schemaname = 'public'
    AND tablename IN (
        'image_metadata',
        'patient_timelines',
        'health_metrics_timeseries',
        'scan_sequences',
        'risk_progressions'
    )
    AND rowsecurity = false;

-- =============================================================================
-- STEP 3: Check which tables are missing RLS policies
-- =============================================================================

SELECT
    'MISSING POLICIES' as issue_type,
    tablename as table_name,
    COUNT(*) as policy_count,
    CASE
        WHEN COUNT(*) = 0 THEN '❌ NO POLICIES'
        WHEN COUNT(*) < 4 THEN '❌ INCOMPLETE POLICIES'
        ELSE '✅ ALL POLICIES'
    END as status
FROM pg_policies
WHERE schemaname = 'public'
    AND tablename IN (
        'image_metadata',
        'patient_timelines',
        'health_metrics_timeseries',
        'scan_sequences',
        'risk_progressions'
    )
GROUP BY tablename
HAVING COUNT(*) < 4

UNION ALL

SELECT
    'MISSING POLICIES' as issue_type,
    expected_table as table_name,
    0 as policy_count,
    '❌ NO POLICIES' as status
FROM (
    SELECT 'image_metadata' as expected_table
    UNION ALL SELECT 'patient_timelines'
    UNION ALL SELECT 'health_metrics_timeseries'
    UNION ALL SELECT 'scan_sequences'
    UNION ALL SELECT 'risk_progressions'
) expected_tables
WHERE expected_table NOT IN (
    SELECT DISTINCT tablename
    FROM pg_policies
    WHERE schemaname = 'public'
);

-- =============================================================================
-- STEP 4: Check primary keys
-- =============================================================================

SELECT
    'MISSING PRIMARY KEY' as issue_type,
    t.table_name,
    '❌ NO PRIMARY KEY' as status
FROM information_schema.tables t
WHERE t.table_schema = 'public'
    AND t.table_name IN (
        'image_metadata',
        'patient_timelines',
        'health_metrics_timeseries',
        'scan_sequences',
        'risk_progressions'
    )
    AND NOT EXISTS (
        SELECT 1
        FROM information_schema.table_constraints tc
        WHERE tc.table_name = t.table_name
        AND tc.constraint_type = 'PRIMARY KEY'
    );

-- =============================================================================
-- STEP 5: Summary of what needs to be fixed
-- =============================================================================

SELECT
    'SUMMARY' as issue_type,
    'Total Issues Found' as description,
    (
        (SELECT COUNT(*) FROM information_schema.tables
         WHERE table_schema = 'public'
         AND table_name IN ('image_metadata','patient_timelines','health_metrics_timeseries','scan_sequences','risk_progressions'))
        +
        (SELECT COUNT(*) FROM pg_tables
         WHERE schemaname = 'public'
         AND tablename IN ('image_metadata','patient_timelines','health_metrics_timeseries','scan_sequences','risk_progressions')
         AND rowsecurity = false)
        +
        (SELECT COUNT(*) FROM (
            SELECT tablename, COUNT(*) as policy_count
            FROM pg_policies
            WHERE schemaname = 'public'
            AND tablename IN ('image_metadata','patient_timelines','health_metrics_timeseries','scan_sequences','risk_progressions')
            GROUP BY tablename
            HAVING COUNT(*) < 4
        ) missing_policies)
    ) as issue_count,
    'Run apply_rls_policies.sql to fix' as action_needed;
