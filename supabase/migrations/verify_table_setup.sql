-- Comprehensive Table Setup Verification Script
-- Run this in Supabase SQL Editor to check if everything is configured correctly

-- =============================================================================
-- STEP 1: Check if tables exist and their structure
-- =============================================================================

SELECT
    'Table Existence Check' as check_type,
    table_name,
    CASE
        WHEN table_name IS NOT NULL THEN '✅ EXISTS'
        ELSE '❌ MISSING'
    END as status
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

-- =============================================================================
-- STEP 2: Check primary keys for each table
-- =============================================================================

SELECT
    'Primary Key Check' as check_type,
    t.table_name,
    c.column_name as primary_key_column,
    c.data_type as primary_key_type,
    CASE
        WHEN c.column_name = 'id' AND c.data_type = 'uuid' THEN '✅ CORRECT'
        ELSE '❌ INCORRECT'
    END as status
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

-- =============================================================================
-- STEP 3: Check foreign keys (user_id references)
-- =============================================================================

SELECT
    'Foreign Key Check' as check_type,
    tc.table_name,
    kcu.column_name as foreign_key_column,
    ccu.table_name as referenced_table,
    ccu.column_name as referenced_column,
    CASE
        WHEN kcu.column_name = 'user_id' AND ccu.table_name = 'users' THEN '✅ CORRECT'
        ELSE '❌ INCORRECT'
    END as status
FROM information_schema.table_constraints tc
JOIN information_schema.key_column_usage kcu ON tc.constraint_name = kcu.constraint_name
JOIN information_schema.constraint_column_usage ccu ON ccu.constraint_name = tc.constraint_name
WHERE tc.constraint_type = 'FOREIGN KEY'
    AND tc.table_name IN (
        'image_metadata',
        'patient_timelines',
        'health_metrics_timeseries',
        'scan_sequences',
        'risk_progressions'
    )
ORDER BY tc.table_name;

-- =============================================================================
-- STEP 4: Check if RLS is enabled on tables
-- =============================================================================

SELECT
    'RLS Status Check' as check_type,
    schemaname,
    tablename,
    CASE
        WHEN rowsecurity = true THEN '✅ ENABLED'
        ELSE '❌ DISABLED'
    END as rls_status
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

-- =============================================================================
-- STEP 5: Check RLS policies exist
-- =============================================================================

SELECT
    'RLS Policy Check' as check_type,
    tablename,
    policyname,
    cmd as operation,
    CASE
        WHEN policyname IS NOT NULL THEN '✅ EXISTS'
        ELSE '❌ MISSING'
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
ORDER BY tablename, policyname;

-- =============================================================================
-- STEP 6: Check table columns and data types
-- =============================================================================

SELECT
    'Column Structure Check' as check_type,
    table_name,
    column_name,
    data_type,
    is_nullable,
    column_default,
    CASE
        WHEN column_name = 'id' AND data_type = 'uuid' THEN '✅ PRIMARY KEY'
        WHEN column_name = 'user_id' AND data_type = 'uuid' THEN '✅ FOREIGN KEY'
        WHEN column_name IN ('created_at', 'updated_at') AND data_type LIKE '%timestamp%' THEN '✅ TIMESTAMP'
        WHEN column_name IN ('analysis_result', 'metadata') AND data_type = 'jsonb' THEN '✅ JSONB'
        ELSE 'ℹ️ REGULAR COLUMN'
    END as column_type
FROM information_schema.columns
WHERE table_schema = 'public'
    AND table_name IN (
        'image_metadata',
        'patient_timelines',
        'health_metrics_timeseries',
        'scan_sequences',
        'risk_progressions'
    )
ORDER BY table_name, ordinal_position;

-- =============================================================================
-- STEP 7: Summary Report
-- =============================================================================

SELECT
    'SUMMARY REPORT' as check_type,
    'Total Tables Checked: ' || COUNT(DISTINCT table_name) as summary,
    'Tables with RLS: ' || COUNT(DISTINCT CASE WHEN rowsecurity = true THEN table_name END) as rls_enabled,
    'Tables with Policies: ' || COUNT(DISTINCT tablename) as tables_with_policies
FROM (
    SELECT table_name FROM information_schema.tables
    WHERE table_schema = 'public'
        AND table_name IN (
            'image_metadata',
            'patient_timelines',
            'health_metrics_timeseries',
            'scan_sequences',
            'risk_progressions'
        )
) t
LEFT JOIN pg_tables pt ON t.table_name = pt.tablename
LEFT JOIN pg_policies pp ON t.table_name = pp.tablename;
