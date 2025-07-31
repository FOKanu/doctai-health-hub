-- Apply RLS Policies for All Tables with CSV Data
-- Run this script BEFORE uploading any CSV data to Supabase

-- =============================================================================
-- STEP 1: Enable RLS on all tables
-- =============================================================================

ALTER TABLE image_metadata ENABLE ROW LEVEL SECURITY;
ALTER TABLE patient_timelines ENABLE ROW LEVEL SECURITY;
ALTER TABLE health_metrics_timeseries ENABLE ROW LEVEL SECURITY;
ALTER TABLE scan_sequences ENABLE ROW LEVEL SECURITY;
ALTER TABLE risk_progressions ENABLE ROW LEVEL SECURITY;

-- =============================================================================
-- STEP 2: Create RLS Policies for image_metadata
-- =============================================================================

CREATE POLICY "Users can view their own images"
    ON image_metadata
    FOR SELECT
    USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own images"
    ON image_metadata
    FOR INSERT
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own images"
    ON image_metadata
    FOR UPDATE
    USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own images"
    ON image_metadata
    FOR DELETE
    USING (auth.uid() = user_id);

-- =============================================================================
-- STEP 3: Create RLS Policies for patient_timelines
-- =============================================================================

CREATE POLICY "Users can view their own timelines"
    ON patient_timelines
    FOR SELECT
    USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own timelines"
    ON patient_timelines
    FOR INSERT
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own timelines"
    ON patient_timelines
    FOR UPDATE
    USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own timelines"
    ON patient_timelines
    FOR DELETE
    USING (auth.uid() = user_id);

-- =============================================================================
-- STEP 4: Create RLS Policies for health_metrics_timeseries
-- =============================================================================

CREATE POLICY "Users can view their own health metrics"
    ON health_metrics_timeseries
    FOR SELECT
    USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own health metrics"
    ON health_metrics_timeseries
    FOR INSERT
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own health metrics"
    ON health_metrics_timeseries
    FOR UPDATE
    USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own health metrics"
    ON health_metrics_timeseries
    FOR DELETE
    USING (auth.uid() = user_id);

-- =============================================================================
-- STEP 5: Create RLS Policies for scan_sequences
-- =============================================================================

CREATE POLICY "Users can view their own scan sequences"
    ON scan_sequences
    FOR SELECT
    USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own scan sequences"
    ON scan_sequences
    FOR INSERT
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own scan sequences"
    ON scan_sequences
    FOR UPDATE
    USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own scan sequences"
    ON scan_sequences
    FOR DELETE
    USING (auth.uid() = user_id);

-- =============================================================================
-- STEP 6: Create RLS Policies for risk_progressions
-- =============================================================================

CREATE POLICY "Users can view their own risk progressions"
    ON risk_progressions
    FOR SELECT
    USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own risk progressions"
    ON risk_progressions
    FOR INSERT
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own risk progressions"
    ON risk_progressions
    FOR UPDATE
    USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own risk progressions"
    ON risk_progressions
    FOR DELETE
    USING (auth.uid() = user_id);

-- =============================================================================
-- STEP 7: Verification Query (Optional)
-- =============================================================================

-- You can run this query to verify all policies were created:
-- SELECT schemaname, tablename, policyname, permissive, roles, cmd, qual, with_check
-- FROM pg_policies
-- WHERE tablename IN ('image_metadata', 'patient_timelines', 'health_metrics_timeseries', 'scan_sequences', 'risk_progressions')
-- ORDER BY tablename, policyname;
