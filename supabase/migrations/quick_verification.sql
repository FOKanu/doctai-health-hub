-- Quick Table Setup Verification
-- Run this script to quickly check if your tables are set up correctly
-- Compare results with table_verification_checklist.csv

-- Check 1: Table Existence
SELECT
    'Table Existence' as check_type,
    table_name,
    '✅ EXISTS' as status,
    'Table exists in database' as description
FROM information_schema.tables
WHERE table_schema = 'public'
    AND table_name IN (
        'image_metadata',
        'patient_timelines',
        'health_metrics_timeseries',
        'scan_sequences',
        'risk_progressions'
    )
ORDER BY table_name;

-- Check 2: Primary Keys
SELECT
    'Primary Key' as check_type,
    t.table_name,
    CASE
        WHEN c.column_name = 'id' AND c.data_type = 'uuid' THEN '✅ CORRECT'
        ELSE '❌ INCORRECT'
    END as status,
    'id (UUID)' as expected_value
FROM information_schema.tables t
JOIN information_schema.columns c ON t.table_name = c.table_name
JOIN information_schema.table_constraints tc ON t.table_name = tc.table_name
JOIN information_schema.key_column_usage kcu ON tc.constraint_name = kcu.constraint_name
WHERE t.table_schema = 'public'
    AND tc.constraint_type = 'PRIMARY KEY'
    AND t.table_name IN (
        'image_metadata',
        'patient_timelines',
        'health_metrics_timeseries',
        'scan_sequences',
        'risk_progressions'
    )
    AND c.column_name = kcu.column_name
ORDER BY t.table_name;

-- Check 3: RLS Status
SELECT
    'RLS Status' as check_type,
    tablename as table_name,
    CASE
        WHEN rowsecurity = true THEN '✅ ENABLED'
        ELSE '❌ DISABLED'
    END as status,
    'Row Level Security enabled' as expected_value
FROM pg_tables
WHERE schemaname = 'public'
    AND tablename IN (
        'image_metadata',
        'patient_timelines',
        'health_metrics_timeseries',
        'scan_sequences',
        'risk_progressions'
    )
ORDER BY tablename;

-- Check 4: RLS Policies Count
SELECT
    'RLS Policy' as check_type,
    tablename as table_name,
    CASE
        WHEN COUNT(*) = 4 THEN '✅ EXISTS'
        ELSE '❌ MISSING'
    END as status,
    '4 policies (SELECT,INSERT,UPDATE,DELETE)' as expected_value
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
ORDER BY tablename;

-- Check 5: Summary
SELECT
    'Summary Report' as check_type,
    'All Tables' as table_name,
    CASE
        WHEN COUNT(DISTINCT t.table_name) = 5
             AND COUNT(DISTINCT CASE WHEN pt.rowsecurity = true THEN pt.tablename END) = 5
             AND COUNT(DISTINCT pp.tablename) = 5
        THEN '✅ COMPLETE'
        ELSE '❌ INCOMPLETE'
    END as status,
    '5 tables with RLS and 20 policies' as expected_value
FROM information_schema.tables t
LEFT JOIN pg_tables pt ON t.table_name = pt.tablename
LEFT JOIN pg_policies pp ON t.table_name = pp.tablename
WHERE t.table_schema = 'public'
    AND t.table_name IN (
        'image_metadata',
        'patient_timelines',
        'health_metrics_timeseries',
        'scan_sequences',
        'risk_progressions'
    );
