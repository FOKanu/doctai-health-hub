-- Time-Series Health Analytics Database Schema Enhancement
-- This migration adds time-series optimized tables for comprehensive health tracking

-- Create enum for condition types
CREATE TYPE condition_type AS ENUM (
  'skin_lesion', 'cardiovascular', 'respiratory', 'neurological',
  'metabolic', 'musculoskeletal', 'gastrointestinal', 'endocrine'
);

-- Create enum for patient status
CREATE TYPE patient_status AS ENUM ('monitoring', 'improving', 'worsening', 'stable');

-- Create enum for metric types
CREATE TYPE metric_type AS ENUM (
  'heart_rate', 'blood_pressure', 'weight', 'temperature', 'sleep_hours',
  'steps', 'calories', 'water_intake', 'blood_glucose', 'oxygen_saturation'
);

-- Create enum for analysis types
CREATE TYPE analysis_type AS ENUM ('progression', 'treatment_response', 'baseline', 'follow_up');

-- Patient progression tracking table
CREATE TABLE patient_timelines (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  condition_type condition_type NOT NULL,
  baseline_date TIMESTAMP WITH TIME ZONE NOT NULL,
  status patient_status NOT NULL DEFAULT 'monitoring',
  severity_score DECIMAL(3,2) CHECK (severity_score >= 0 AND severity_score <= 1),
  confidence_score DECIMAL(3,2) CHECK (confidence_score >= 0 AND confidence_score <= 1),
  notes TEXT,
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Time-series health metrics table
CREATE TABLE health_metrics_timeseries (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  metric_type metric_type NOT NULL,
  value JSONB NOT NULL, -- Flexible for different metric structures
  recorded_at TIMESTAMP WITH TIME ZONE NOT NULL,
  device_source TEXT,
  accuracy_score DECIMAL(3,2) CHECK (accuracy_score >= 0 AND accuracy_score <= 1),
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Medical scan sequences for progression analysis
CREATE TABLE scan_sequences (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  sequence_name TEXT NOT NULL,
  image_ids UUID[] NOT NULL, -- Array of related image_metadata IDs
  analysis_type analysis_type NOT NULL,
  baseline_image_id UUID REFERENCES image_metadata(id),
  progression_score DECIMAL(3,2) CHECK (progression_score >= 0 AND progression_score <= 1),
  confidence_score DECIMAL(3,2) CHECK (confidence_score >= 0 AND confidence_score <= 1),
  findings JSONB DEFAULT '{}',
  recommendations TEXT[],
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Treatment response tracking
CREATE TABLE treatment_responses (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  timeline_id UUID REFERENCES patient_timelines(id) ON DELETE CASCADE,
  treatment_name TEXT NOT NULL,
  start_date TIMESTAMP WITH TIME ZONE NOT NULL,
  end_date TIMESTAMP WITH TIME ZONE,
  effectiveness_score DECIMAL(3,2) CHECK (effectiveness_score >= 0 AND effectiveness_score <= 1),
  side_effects JSONB DEFAULT '{}',
  adherence_percentage DECIMAL(5,2) CHECK (adherence_percentage >= 0 AND adherence_percentage <= 100),
  notes TEXT,
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Risk progression tracking
CREATE TABLE risk_progressions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  condition_type condition_type NOT NULL,
  risk_level TEXT NOT NULL CHECK (risk_level IN ('low', 'medium', 'high')),
  probability DECIMAL(3,2) CHECK (probability >= 0 AND probability <= 1),
  factors JSONB DEFAULT '{}',
  recorded_at TIMESTAMP WITH TIME ZONE NOT NULL,
  predicted_date TIMESTAMP WITH TIME ZONE,
  confidence_score DECIMAL(3,2) CHECK (confidence_score >= 0 AND confidence_score <= 1),
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for time-series optimization
CREATE INDEX idx_patient_timelines_user_id ON patient_timelines(user_id);
CREATE INDEX idx_patient_timelines_condition_type ON patient_timelines(condition_type);
CREATE INDEX idx_patient_timelines_baseline_date ON patient_timelines(baseline_date);
CREATE INDEX idx_patient_timelines_status ON patient_timelines(status);

CREATE INDEX idx_health_metrics_user_id ON health_metrics_timeseries(user_id);
CREATE INDEX idx_health_metrics_type ON health_metrics_timeseries(metric_type);
CREATE INDEX idx_health_metrics_recorded_at ON health_metrics_timeseries(recorded_at);
CREATE INDEX idx_health_metrics_user_type_time ON health_metrics_timeseries(user_id, metric_type, recorded_at);

CREATE INDEX idx_scan_sequences_user_id ON scan_sequences(user_id);
CREATE INDEX idx_scan_sequences_analysis_type ON scan_sequences(analysis_type);
CREATE INDEX idx_scan_sequences_created_at ON scan_sequences(created_at);

CREATE INDEX idx_treatment_responses_user_id ON treatment_responses(user_id);
CREATE INDEX idx_treatment_responses_timeline_id ON treatment_responses(timeline_id);
CREATE INDEX idx_treatment_responses_start_date ON treatment_responses(start_date);

CREATE INDEX idx_risk_progressions_user_id ON risk_progressions(user_id);
CREATE INDEX idx_risk_progressions_condition_type ON risk_progressions(condition_type);
CREATE INDEX idx_risk_progressions_recorded_at ON risk_progressions(recorded_at);
CREATE INDEX idx_risk_progressions_risk_level ON risk_progressions(risk_level);

-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply updated_at triggers
CREATE TRIGGER update_patient_timelines_updated_at
  BEFORE UPDATE ON patient_timelines
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_scan_sequences_updated_at
  BEFORE UPDATE ON scan_sequences
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_treatment_responses_updated_at
  BEFORE UPDATE ON treatment_responses
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Set up Row Level Security (RLS)
ALTER TABLE patient_timelines ENABLE ROW LEVEL SECURITY;
ALTER TABLE health_metrics_timeseries ENABLE ROW LEVEL SECURITY;
ALTER TABLE scan_sequences ENABLE ROW LEVEL SECURITY;
ALTER TABLE treatment_responses ENABLE ROW LEVEL SECURITY;
ALTER TABLE risk_progressions ENABLE ROW LEVEL SECURITY;

-- Create RLS policies for patient_timelines
CREATE POLICY "Users can view their own timelines"
  ON patient_timelines FOR SELECT
  USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own timelines"
  ON patient_timelines FOR INSERT
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own timelines"
  ON patient_timelines FOR UPDATE
  USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own timelines"
  ON patient_timelines FOR DELETE
  USING (auth.uid() = user_id);

-- Create RLS policies for health_metrics_timeseries
CREATE POLICY "Users can view their own health metrics"
  ON health_metrics_timeseries FOR SELECT
  USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own health metrics"
  ON health_metrics_timeseries FOR INSERT
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own health metrics"
  ON health_metrics_timeseries FOR UPDATE
  USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own health metrics"
  ON health_metrics_timeseries FOR DELETE
  USING (auth.uid() = user_id);

-- Create RLS policies for scan_sequences
CREATE POLICY "Users can view their own scan sequences"
  ON scan_sequences FOR SELECT
  USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own scan sequences"
  ON scan_sequences FOR INSERT
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own scan sequences"
  ON scan_sequences FOR UPDATE
  USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own scan sequences"
  ON scan_sequences FOR DELETE
  USING (auth.uid() = user_id);

-- Create RLS policies for treatment_responses
CREATE POLICY "Users can view their own treatment responses"
  ON treatment_responses FOR SELECT
  USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own treatment responses"
  ON treatment_responses FOR INSERT
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own treatment responses"
  ON treatment_responses FOR UPDATE
  USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own treatment responses"
  ON treatment_responses FOR DELETE
  USING (auth.uid() = user_id);

-- Create RLS policies for risk_progressions
CREATE POLICY "Users can view their own risk progressions"
  ON risk_progressions FOR SELECT
  USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own risk progressions"
  ON risk_progressions FOR INSERT
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own risk progressions"
  ON risk_progressions FOR UPDATE
  USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own risk progressions"
  ON risk_progressions FOR DELETE
  USING (auth.uid() = user_id);

-- Create functions for time-series analytics
CREATE OR REPLACE FUNCTION get_health_metrics_trend(
  p_user_id UUID,
  p_metric_type metric_type,
  p_days INTEGER DEFAULT 30
)
RETURNS TABLE (
  recorded_at TIMESTAMP WITH TIME ZONE,
  value JSONB,
  trend_direction TEXT
) AS $$
BEGIN
  RETURN QUERY
  SELECT
    hmt.recorded_at,
    hmt.value,
    CASE
      WHEN LAG(hmt.value->>'value') OVER (ORDER BY hmt.recorded_at) < (hmt.value->>'value')::DECIMAL THEN 'increasing'
      WHEN LAG(hmt.value->>'value') OVER (ORDER BY hmt.recorded_at) > (hmt.value->>'value')::DECIMAL THEN 'decreasing'
      ELSE 'stable'
    END as trend_direction
  FROM health_metrics_timeseries hmt
  WHERE hmt.user_id = p_user_id
    AND hmt.metric_type = p_metric_type
    AND hmt.recorded_at >= NOW() - INTERVAL '1 day' * p_days
  ORDER BY hmt.recorded_at;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to get patient progression summary
CREATE OR REPLACE FUNCTION get_patient_progression_summary(
  p_user_id UUID,
  p_condition_type condition_type DEFAULT NULL
)
RETURNS TABLE (
  condition_type condition_type,
  status patient_status,
  severity_score DECIMAL(3,2),
  days_since_baseline INTEGER,
  trend TEXT
) AS $$
BEGIN
  RETURN QUERY
  SELECT
    pt.condition_type,
    pt.status,
    pt.severity_score,
    EXTRACT(DAY FROM NOW() - pt.baseline_date)::INTEGER as days_since_baseline,
    CASE
      WHEN pt.status = 'improving' THEN 'positive'
      WHEN pt.status = 'worsening' THEN 'negative'
      ELSE 'neutral'
    END as trend
  FROM patient_timelines pt
  WHERE pt.user_id = p_user_id
    AND (p_condition_type IS NULL OR pt.condition_type = p_condition_type)
  ORDER BY pt.baseline_date DESC;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;
