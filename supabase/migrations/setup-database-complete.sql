-- Complete DoctAI Health Hub Database Setup
-- Run this script in your Supabase SQL Editor to set up all required tables

-- =============================================================================
-- STEP 1: Create Base Tables and Enums
-- =============================================================================

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

-- Create enum for appointment types
CREATE TYPE appointment_type AS ENUM (
  'routine_checkup', 'follow_up', 'emergency', 'consultation', 'procedure'
);

-- Create enum for appointment status
CREATE TYPE appointment_status AS ENUM (
  'scheduled', 'confirmed', 'in_progress', 'completed', 'cancelled', 'no_show'
);

-- Create enum for consultation types
CREATE TYPE consultation_type AS ENUM ('video', 'audio', 'chat', 'in_person');

-- Create enum for consultation status
CREATE TYPE consultation_status AS ENUM (
  'scheduled', 'in_progress', 'completed', 'cancelled', 'no_show'
);

-- Create enum for provider specialties
CREATE TYPE provider_specialty AS ENUM (
  'primary_care', 'cardiology', 'dermatology', 'neurology', 'orthopedics',
  'ophthalmology', 'psychiatry', 'pediatrics', 'gynecology', 'urology',
  'gastroenterology', 'endocrinology', 'pulmonology', 'rheumatology'
);

-- Create enum for alert types
CREATE TYPE alert_type AS ENUM (
  'blood_pressure', 'heart_rate', 'blood_glucose', 'weight', 'sleep_quality',
  'medication_reminder', 'appointment_reminder', 'test_result', 'health_goal'
);

-- Create enum for alert levels
CREATE TYPE alert_level AS ENUM ('low', 'medium', 'high', 'critical');

-- =============================================================================
-- STEP 2: Create Core Tables
-- =============================================================================

-- Image metadata table for storing medical images
CREATE TABLE IF NOT EXISTS image_metadata (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  url TEXT NOT NULL,
  type TEXT NOT NULL,
  analysis_result JSONB DEFAULT '{}',
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Healthcare providers table
CREATE TABLE IF NOT EXISTS healthcare_providers (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  provider_name TEXT NOT NULL,
  specialty provider_specialty NOT NULL,
  license_number TEXT,
  credentials TEXT[],
  experience_years INTEGER,
  languages TEXT[],
  consultation_fee DECIMAL(10,2),
  rating DECIMAL(3,2) CHECK (rating >= 0 AND rating <= 5),
  is_verified BOOLEAN DEFAULT false,
  is_available BOOLEAN DEFAULT true,
  bio TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Appointments table
CREATE TABLE IF NOT EXISTS appointments (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  provider_id UUID REFERENCES healthcare_providers(id) ON DELETE CASCADE,
  appointment_type appointment_type NOT NULL,
  scheduled_at TIMESTAMP WITH TIME ZONE NOT NULL,
  status appointment_status DEFAULT 'scheduled',
  notes TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Telemedicine consultations table
CREATE TABLE IF NOT EXISTS telemedicine_consultations (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  provider_id UUID REFERENCES healthcare_providers(id) ON DELETE CASCADE,
  consultation_type consultation_type NOT NULL,
  status consultation_status DEFAULT 'scheduled',
  scheduled_at TIMESTAMP WITH TIME ZONE NOT NULL,
  consultation_notes TEXT,
  diagnosis TEXT,
  prescriptions JSONB DEFAULT '[]',
  recommendations TEXT[],
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =============================================================================
-- STEP 3: Create Time-Series Tables
-- =============================================================================

-- Patient progression tracking table
CREATE TABLE IF NOT EXISTS patient_timelines (
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
CREATE TABLE IF NOT EXISTS health_metrics_timeseries (
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
CREATE TABLE IF NOT EXISTS scan_sequences (
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
CREATE TABLE IF NOT EXISTS treatment_responses (
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
CREATE TABLE IF NOT EXISTS risk_progressions (
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

-- =============================================================================
-- STEP 4: Create Specialized Health Tables
-- =============================================================================

-- Cardiovascular metrics table
CREATE TABLE IF NOT EXISTS cardiovascular_metrics (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  heart_rate_resting INTEGER,
  heart_rate_active INTEGER,
  heart_rate_variability DECIMAL(5,2),
  blood_pressure_systolic INTEGER,
  blood_pressure_diastolic INTEGER,
  pulse_pressure INTEGER,
  mean_arterial_pressure DECIMAL(5,2),
  ecg_rhythm TEXT,
  recorded_at TIMESTAMP WITH TIME ZONE NOT NULL,
  device_source TEXT,
  accuracy_score DECIMAL(3,2) CHECK (accuracy_score >= 0 AND accuracy_score <= 1),
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Sleep metrics table
CREATE TABLE IF NOT EXISTS sleep_metrics (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  sleep_date DATE NOT NULL,
  total_duration DECIMAL(4,2), -- hours
  deep_sleep_duration DECIMAL(4,2),
  rem_sleep_duration DECIMAL(4,2),
  sleep_efficiency DECIMAL(5,2), -- percentage
  sleep_latency INTEGER, -- minutes
  wake_count INTEGER,
  sleep_quality_score DECIMAL(3,1) CHECK (sleep_quality_score >= 0 AND sleep_quality_score <= 10),
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Fitness metrics table
CREATE TABLE IF NOT EXISTS fitness_metrics (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  activity_date DATE NOT NULL,
  steps_count INTEGER,
  calories_burned INTEGER,
  active_minutes INTEGER,
  distance_km DECIMAL(6,2),
  vo2_max DECIMAL(4,1),
  resting_heart_rate INTEGER,
  max_heart_rate INTEGER,
  avg_heart_rate INTEGER,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Mental health metrics table
CREATE TABLE IF NOT EXISTS mental_health_metrics (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  mood_score DECIMAL(3,1) CHECK (mood_score >= 0 AND mood_score <= 10),
  stress_level DECIMAL(3,1) CHECK (stress_level >= 0 AND stress_level <= 10),
  anxiety_level DECIMAL(3,1) CHECK (anxiety_level >= 0 AND anxiety_level <= 10),
  depression_score DECIMAL(3,1) CHECK (depression_score >= 0 AND depression_score <= 10),
  sleep_quality DECIMAL(3,1) CHECK (sleep_quality >= 0 AND sleep_quality <= 10),
  social_connections DECIMAL(3,1) CHECK (social_connections >= 0 AND social_connections <= 10),
  recorded_at TIMESTAMP WITH TIME ZONE NOT NULL,
  device_source TEXT,
  accuracy_score DECIMAL(3,2) CHECK (accuracy_score >= 0 AND accuracy_score <= 1),
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Health alerts table
CREATE TABLE IF NOT EXISTS health_alerts (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  alert_type alert_type NOT NULL,
  alert_level alert_level NOT NULL,
  title TEXT NOT NULL,
  message TEXT NOT NULL,
  metric_name TEXT,
  metric_value JSONB,
  threshold_value JSONB,
  is_read BOOLEAN DEFAULT false,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Provider availability table
CREATE TABLE IF NOT EXISTS provider_availability (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  provider_id UUID REFERENCES healthcare_providers(id) ON DELETE CASCADE,
  day_of_week TEXT NOT NULL CHECK (day_of_week IN ('monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday')),
  start_time TIME NOT NULL,
  end_time TIME NOT NULL,
  is_available BOOLEAN DEFAULT true,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Patient-provider relationships table
CREATE TABLE IF NOT EXISTS patient_provider_relationships (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  provider_id UUID REFERENCES healthcare_providers(id) ON DELETE CASCADE,
  relationship_type TEXT NOT NULL CHECK (relationship_type IN ('primary_care', 'specialist', 'consultant')),
  start_date TIMESTAMP WITH TIME ZONE NOT NULL,
  end_date TIMESTAMP WITH TIME ZONE,
  is_active BOOLEAN DEFAULT true,
  notes TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Remote monitoring table
CREATE TABLE IF NOT EXISTS remote_monitoring (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  provider_id UUID REFERENCES healthcare_providers(id) ON DELETE CASCADE,
  monitoring_type TEXT NOT NULL,
  device_id TEXT NOT NULL,
  device_type TEXT NOT NULL,
  is_active BOOLEAN DEFAULT true,
  start_date TIMESTAMP WITH TIME ZONE NOT NULL,
  end_date TIMESTAMP WITH TIME ZONE,
  monitoring_frequency TEXT,
  alert_thresholds JSONB DEFAULT '{}',
  last_reading_at TIMESTAMP WITH TIME ZONE,
  last_reading_value JSONB,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =============================================================================
-- STEP 5: Create Indexes for Performance
-- =============================================================================

-- Indexes for patient_timelines
CREATE INDEX idx_patient_timelines_user_id ON patient_timelines(user_id);
CREATE INDEX idx_patient_timelines_condition_type ON patient_timelines(condition_type);
CREATE INDEX idx_patient_timelines_baseline_date ON patient_timelines(baseline_date);
CREATE INDEX idx_patient_timelines_status ON patient_timelines(status);

-- Indexes for health_metrics_timeseries
CREATE INDEX idx_health_metrics_user_id ON health_metrics_timeseries(user_id);
CREATE INDEX idx_health_metrics_type ON health_metrics_timeseries(metric_type);
CREATE INDEX idx_health_metrics_recorded_at ON health_metrics_timeseries(recorded_at);
CREATE INDEX idx_health_metrics_user_type_time ON health_metrics_timeseries(user_id, metric_type, recorded_at);

-- Indexes for scan_sequences
CREATE INDEX idx_scan_sequences_user_id ON scan_sequences(user_id);
CREATE INDEX idx_scan_sequences_analysis_type ON scan_sequences(analysis_type);
CREATE INDEX idx_scan_sequences_created_at ON scan_sequences(created_at);

-- Indexes for treatment_responses
CREATE INDEX idx_treatment_responses_user_id ON treatment_responses(user_id);
CREATE INDEX idx_treatment_responses_timeline_id ON treatment_responses(timeline_id);
CREATE INDEX idx_treatment_responses_start_date ON treatment_responses(start_date);

-- Indexes for risk_progressions
CREATE INDEX idx_risk_progressions_user_id ON risk_progressions(user_id);
CREATE INDEX idx_risk_progressions_condition_type ON risk_progressions(condition_type);
CREATE INDEX idx_risk_progressions_recorded_at ON risk_progressions(recorded_at);
CREATE INDEX idx_risk_progressions_risk_level ON risk_progressions(risk_level);

-- Indexes for other tables
CREATE INDEX idx_image_metadata_user_id ON image_metadata(user_id);
CREATE INDEX idx_image_metadata_type ON image_metadata(type);
CREATE INDEX idx_healthcare_providers_specialty ON healthcare_providers(specialty);
CREATE INDEX idx_appointments_patient_id ON appointments(patient_id);
CREATE INDEX idx_appointments_provider_id ON appointments(provider_id);
CREATE INDEX idx_appointments_scheduled_at ON appointments(scheduled_at);
CREATE INDEX idx_telemedicine_consultations_patient_id ON telemedicine_consultations(patient_id);
CREATE INDEX idx_health_alerts_patient_id ON health_alerts(patient_id);
CREATE INDEX idx_health_alerts_alert_type ON health_alerts(alert_type);

-- =============================================================================
-- STEP 6: Create Triggers and Functions
-- =============================================================================

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

CREATE TRIGGER update_image_metadata_updated_at
  BEFORE UPDATE ON image_metadata
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_healthcare_providers_updated_at
  BEFORE UPDATE ON healthcare_providers
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_appointments_updated_at
  BEFORE UPDATE ON appointments
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_telemedicine_consultations_updated_at
  BEFORE UPDATE ON telemedicine_consultations
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_patient_provider_relationships_updated_at
  BEFORE UPDATE ON patient_provider_relationships
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_remote_monitoring_updated_at
  BEFORE UPDATE ON remote_monitoring
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- =============================================================================
-- STEP 7: Set up Row Level Security (RLS)
-- =============================================================================

-- Enable RLS on all tables
ALTER TABLE image_metadata ENABLE ROW LEVEL SECURITY;
ALTER TABLE healthcare_providers ENABLE ROW LEVEL SECURITY;
ALTER TABLE appointments ENABLE ROW LEVEL SECURITY;
ALTER TABLE telemedicine_consultations ENABLE ROW LEVEL SECURITY;
ALTER TABLE patient_timelines ENABLE ROW LEVEL SECURITY;
ALTER TABLE health_metrics_timeseries ENABLE ROW LEVEL SECURITY;
ALTER TABLE scan_sequences ENABLE ROW LEVEL SECURITY;
ALTER TABLE treatment_responses ENABLE ROW LEVEL SECURITY;
ALTER TABLE risk_progressions ENABLE ROW LEVEL SECURITY;
ALTER TABLE cardiovascular_metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE sleep_metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE fitness_metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE mental_health_metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE health_alerts ENABLE ROW LEVEL SECURITY;
ALTER TABLE provider_availability ENABLE ROW LEVEL SECURITY;
ALTER TABLE patient_provider_relationships ENABLE ROW LEVEL SECURITY;
ALTER TABLE remote_monitoring ENABLE ROW LEVEL SECURITY;

-- Create RLS policies for all tables (users can only access their own data)
-- This is a simplified policy - you may want to customize based on your needs

-- Image metadata policies
CREATE POLICY "Users can view their own images" ON image_metadata FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY "Users can insert their own images" ON image_metadata FOR INSERT WITH CHECK (auth.uid() = user_id);
CREATE POLICY "Users can update their own images" ON image_metadata FOR UPDATE USING (auth.uid() = user_id);
CREATE POLICY "Users can delete their own images" ON image_metadata FOR DELETE USING (auth.uid() = user_id);

-- Healthcare providers policies (providers can view their own profiles)
CREATE POLICY "Providers can view their own profiles" ON healthcare_providers FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY "Providers can insert their own profiles" ON healthcare_providers FOR INSERT WITH CHECK (auth.uid() = user_id);
CREATE POLICY "Providers can update their own profiles" ON healthcare_providers FOR UPDATE USING (auth.uid() = user_id);

-- Appointments policies
CREATE POLICY "Users can view their own appointments" ON appointments FOR SELECT USING (auth.uid() = patient_id OR auth.uid() = (SELECT user_id FROM healthcare_providers WHERE id = provider_id));
CREATE POLICY "Users can insert their own appointments" ON appointments FOR INSERT WITH CHECK (auth.uid() = patient_id);
CREATE POLICY "Users can update their own appointments" ON appointments FOR UPDATE USING (auth.uid() = patient_id OR auth.uid() = (SELECT user_id FROM healthcare_providers WHERE id = provider_id));

-- Telemedicine consultations policies
CREATE POLICY "Users can view their own consultations" ON telemedicine_consultations FOR SELECT USING (auth.uid() = patient_id OR auth.uid() = (SELECT user_id FROM healthcare_providers WHERE id = provider_id));
CREATE POLICY "Users can insert their own consultations" ON telemedicine_consultations FOR INSERT WITH CHECK (auth.uid() = patient_id);
CREATE POLICY "Users can update their own consultations" ON telemedicine_consultations FOR UPDATE USING (auth.uid() = patient_id OR auth.uid() = (SELECT user_id FROM healthcare_providers WHERE id = provider_id));

-- Patient timelines policies
CREATE POLICY "Users can view their own timelines" ON patient_timelines FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY "Users can insert their own timelines" ON patient_timelines FOR INSERT WITH CHECK (auth.uid() = user_id);
CREATE POLICY "Users can update their own timelines" ON patient_timelines FOR UPDATE USING (auth.uid() = user_id);
CREATE POLICY "Users can delete their own timelines" ON patient_timelines FOR DELETE USING (auth.uid() = user_id);

-- Health metrics policies
CREATE POLICY "Users can view their own health metrics" ON health_metrics_timeseries FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY "Users can insert their own health metrics" ON health_metrics_timeseries FOR INSERT WITH CHECK (auth.uid() = user_id);
CREATE POLICY "Users can update their own health metrics" ON health_metrics_timeseries FOR UPDATE USING (auth.uid() = user_id);
CREATE POLICY "Users can delete their own health metrics" ON health_metrics_timeseries FOR DELETE USING (auth.uid() = user_id);

-- Scan sequences policies
CREATE POLICY "Users can view their own scan sequences" ON scan_sequences FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY "Users can insert their own scan sequences" ON scan_sequences FOR INSERT WITH CHECK (auth.uid() = user_id);
CREATE POLICY "Users can update their own scan sequences" ON scan_sequences FOR UPDATE USING (auth.uid() = user_id);
CREATE POLICY "Users can delete their own scan sequences" ON scan_sequences FOR DELETE USING (auth.uid() = user_id);

-- Treatment responses policies
CREATE POLICY "Users can view their own treatment responses" ON treatment_responses FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY "Users can insert their own treatment responses" ON treatment_responses FOR INSERT WITH CHECK (auth.uid() = user_id);
CREATE POLICY "Users can update their own treatment responses" ON treatment_responses FOR UPDATE USING (auth.uid() = user_id);
CREATE POLICY "Users can delete their own treatment responses" ON treatment_responses FOR DELETE USING (auth.uid() = user_id);

-- Risk progressions policies
CREATE POLICY "Users can view their own risk progressions" ON risk_progressions FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY "Users can insert their own risk progressions" ON risk_progressions FOR INSERT WITH CHECK (auth.uid() = user_id);
CREATE POLICY "Users can update their own risk progressions" ON risk_progressions FOR UPDATE USING (auth.uid() = user_id);
CREATE POLICY "Users can delete their own risk progressions" ON risk_progressions FOR DELETE USING (auth.uid() = user_id);

-- Specialized health metrics policies
CREATE POLICY "Users can view their own cardiovascular metrics" ON cardiovascular_metrics FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY "Users can insert their own cardiovascular metrics" ON cardiovascular_metrics FOR INSERT WITH CHECK (auth.uid() = user_id);
CREATE POLICY "Users can update their own cardiovascular metrics" ON cardiovascular_metrics FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "Users can view their own sleep metrics" ON sleep_metrics FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY "Users can insert their own sleep metrics" ON sleep_metrics FOR INSERT WITH CHECK (auth.uid() = user_id);
CREATE POLICY "Users can update their own sleep metrics" ON sleep_metrics FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "Users can view their own fitness metrics" ON fitness_metrics FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY "Users can insert their own fitness metrics" ON fitness_metrics FOR INSERT WITH CHECK (auth.uid() = user_id);
CREATE POLICY "Users can update their own fitness metrics" ON fitness_metrics FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "Users can view their own mental health metrics" ON mental_health_metrics FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY "Users can insert their own mental health metrics" ON mental_health_metrics FOR INSERT WITH CHECK (auth.uid() = user_id);
CREATE POLICY "Users can update their own mental health metrics" ON mental_health_metrics FOR UPDATE USING (auth.uid() = user_id);

-- Health alerts policies
CREATE POLICY "Users can view their own health alerts" ON health_alerts FOR SELECT USING (auth.uid() = patient_id);
CREATE POLICY "Users can insert their own health alerts" ON health_alerts FOR INSERT WITH CHECK (auth.uid() = patient_id);
CREATE POLICY "Users can update their own health alerts" ON health_alerts FOR UPDATE USING (auth.uid() = patient_id);

-- Provider availability policies
CREATE POLICY "Providers can view their own availability" ON provider_availability FOR SELECT USING (auth.uid() = (SELECT user_id FROM healthcare_providers WHERE id = provider_id));
CREATE POLICY "Providers can insert their own availability" ON provider_availability FOR INSERT WITH CHECK (auth.uid() = (SELECT user_id FROM healthcare_providers WHERE id = provider_id));
CREATE POLICY "Providers can update their own availability" ON provider_availability FOR UPDATE USING (auth.uid() = (SELECT user_id FROM healthcare_providers WHERE id = provider_id));

-- Patient-provider relationships policies
CREATE POLICY "Users can view their own relationships" ON patient_provider_relationships FOR SELECT USING (auth.uid() = patient_id OR auth.uid() = (SELECT user_id FROM healthcare_providers WHERE id = provider_id));
CREATE POLICY "Users can insert their own relationships" ON patient_provider_relationships FOR INSERT WITH CHECK (auth.uid() = patient_id);
CREATE POLICY "Users can update their own relationships" ON patient_provider_relationships FOR UPDATE USING (auth.uid() = patient_id OR auth.uid() = (SELECT user_id FROM healthcare_providers WHERE id = provider_id));

-- Remote monitoring policies
CREATE POLICY "Users can view their own monitoring" ON remote_monitoring FOR SELECT USING (auth.uid() = patient_id OR auth.uid() = (SELECT user_id FROM healthcare_providers WHERE id = provider_id));
CREATE POLICY "Users can insert their own monitoring" ON remote_monitoring FOR INSERT WITH CHECK (auth.uid() = patient_id);
CREATE POLICY "Users can update their own monitoring" ON remote_monitoring FOR UPDATE USING (auth.uid() = patient_id OR auth.uid() = (SELECT user_id FROM healthcare_providers WHERE id = provider_id));

-- =============================================================================
-- STEP 8: Create Analytics Functions
-- =============================================================================

-- Function to get health metrics trend
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

-- =============================================================================
-- STEP 9: Insert Sample Data
-- =============================================================================

-- Insert sample health metrics for mock_user
INSERT INTO health_metrics_timeseries (user_id, metric_type, value, recorded_at, device_source, accuracy_score)
VALUES
  ('550e8400-e29b-41d4-a716-446655440004', 'heart_rate', '{"value": 72}', NOW() - INTERVAL '1 day', 'smartwatch', 0.95),
  ('550e8400-e29b-41d4-a716-446655440004', 'heart_rate', '{"value": 75}', NOW() - INTERVAL '2 days', 'smartwatch', 0.95),
  ('550e8400-e29b-41d4-a716-446655440004', 'heart_rate', '{"value": 68}', NOW() - INTERVAL '3 days', 'smartwatch', 0.95),
  ('550e8400-e29b-41d4-a716-446655440004', 'blood_pressure', '{"systolic": 120, "diastolic": 80}', NOW() - INTERVAL '1 day', 'blood_pressure_monitor', 0.90),
  ('550e8400-e29b-41d4-a716-446655440004', 'blood_pressure', '{"systolic": 118, "diastolic": 78}', NOW() - INTERVAL '2 days', 'blood_pressure_monitor', 0.90),
  ('550e8400-e29b-41d4-a716-446655440004', 'temperature', '{"value": 98.6}', NOW() - INTERVAL '1 day', 'thermometer', 0.98),
  ('550e8400-e29b-41d4-a716-446655440004', 'weight', '{"value": 70.5}', NOW() - INTERVAL '1 day', 'smart_scale', 0.95),
  ('550e8400-e29b-41d4-a716-446655440004', 'sleep_hours', '{"hours": 7.5}', NOW() - INTERVAL '1 day', 'smartwatch', 0.90),
  ('550e8400-e29b-41d4-a716-446655440004', 'steps', '{"count": 8500}', NOW() - INTERVAL '1 day', 'smartwatch', 0.95),
  ('550e8400-e29b-41d4-a716-446655440004', 'water_intake', '{"amount": 2000}', NOW() - INTERVAL '1 day', 'water_tracker', 0.85)
ON CONFLICT DO NOTHING;

-- Insert sample risk progression data
INSERT INTO risk_progressions (user_id, condition_type, risk_level, probability, factors, recorded_at, confidence_score)
VALUES
  ('550e8400-e29b-41d4-a716-446655440004', 'cardiovascular', 'low', 0.15, '{"age": 35, "family_history": false, "lifestyle": "active"}', NOW() - INTERVAL '1 day', 0.85),
  ('550e8400-e29b-41d4-a716-446655440004', 'cardiovascular', 'low', 0.12, '{"age": 35, "family_history": false, "lifestyle": "active"}', NOW() - INTERVAL '2 days', 0.88),
  ('550e8400-e29b-41d4-a716-446655440004', 'metabolic', 'medium', 0.45, '{"bmi": 26.5, "blood_glucose": 95, "family_history": true}', NOW() - INTERVAL '1 day', 0.75),
  ('550e8400-e29b-41d4-a716-446655440004', 'respiratory', 'low', 0.08, '{"age": 35, "no_smoking_history": true, "environment": "clean"}', NOW() - INTERVAL '1 day', 0.92)
ON CONFLICT DO NOTHING;

-- Insert sample patient timeline
INSERT INTO patient_timelines (user_id, condition_type, baseline_date, status, severity_score, confidence_score, notes)
VALUES
  ('550e8400-e29b-41d4-a716-446655440004', 'cardiovascular', NOW() - INTERVAL '30 days', 'stable', 0.15, 0.85, 'Regular monitoring of cardiovascular health'),
  ('550e8400-e29b-41d4-a716-446655440004', 'metabolic', NOW() - INTERVAL '45 days', 'monitoring', 0.45, 0.75, 'Monitoring blood glucose levels and lifestyle changes')
ON CONFLICT DO NOTHING;

-- Insert sample image metadata
INSERT INTO image_metadata (user_id, url, type, analysis_result, metadata)
VALUES
  ('550e8400-e29b-41d4-a716-446655440004', 'https://example.com/skin_lesion_1.jpg', 'skin_lesion', '{"diagnosis": "benign_mole", "confidence": 0.92, "recommendations": ["monitor_for_changes", "sun_protection"]}', '{"location": "left_arm", "size": "5mm"}'),
  ('550e8400-e29b-41d4-a716-446655440004', 'https://example.com/xray_chest_1.jpg', 'xray', '{"diagnosis": "normal_chest_xray", "confidence": 0.95, "findings": ["clear_lung_fields", "normal_cardiac_silhouette"]}', '{"view": "PA", "technique": "standard"}')
ON CONFLICT DO NOTHING;

-- Insert sample scan sequences
INSERT INTO scan_sequences (user_id, sequence_name, image_ids, analysis_type, progression_score, confidence_score, findings, recommendations)
VALUES
  ('550e8400-e29b-41d4-a716-446655440004', 'Chest X-Ray Series', ARRAY['550e8400-e29b-41d4-a716-446655440004'], 'baseline', 0.10, 0.95, '{"lung_fields": "clear", "cardiac_silhouette": "normal", "bony_structures": "intact"}', ARRAY['Continue annual screening', 'Maintain healthy lifestyle'])
ON CONFLICT DO NOTHING;

-- Insert sample cardiovascular metrics
INSERT INTO cardiovascular_metrics (user_id, heart_rate_resting, heart_rate_active, heart_rate_variability, blood_pressure_systolic, blood_pressure_diastolic, pulse_pressure, mean_arterial_pressure, ecg_rhythm, recorded_at, device_source, accuracy_score)
VALUES
  ('550e8400-e29b-41d4-a716-446655440004', 72, 140, 45.2, 120, 80, 40, 93.33, 'normal', NOW() - INTERVAL '1 day', 'smartwatch', 0.95),
  ('550e8400-e29b-41d4-a716-446655440004', 70, 135, 48.1, 118, 78, 40, 91.33, 'normal', NOW() - INTERVAL '2 days', 'smartwatch', 0.95)
ON CONFLICT DO NOTHING;

-- Insert sample sleep metrics
INSERT INTO sleep_metrics (user_id, sleep_date, total_duration, deep_sleep_duration, rem_sleep_duration, sleep_efficiency, sleep_latency, wake_count, sleep_quality_score)
VALUES
  ('550e8400-e29b-41d4-a716-446655440004', CURRENT_DATE - INTERVAL '1 day', 7.5, 2.1, 1.8, 85.5, 15.2, 2, 8.2),
  ('550e8400-e29b-41d4-a716-446655440004', CURRENT_DATE - INTERVAL '2 days', 8.0, 2.3, 2.0, 88.0, 12.5, 1, 8.5)
ON CONFLICT DO NOTHING;

-- Insert sample fitness metrics
INSERT INTO fitness_metrics (user_id, activity_date, steps_count, calories_burned, active_minutes, distance_km, vo2_max, resting_heart_rate, max_heart_rate, avg_heart_rate)
VALUES
  ('550e8400-e29b-41d4-a716-446655440004', CURRENT_DATE - INTERVAL '1 day', 8500, 320, 45, 6.2, 42.5, 70, 140, 95),
  ('550e8400-e29b-41d4-a716-446655440004', CURRENT_DATE - INTERVAL '2 days', 9200, 350, 52, 6.8, 43.1, 68, 142, 98)
ON CONFLICT DO NOTHING;

-- Insert sample mental health metrics
INSERT INTO mental_health_metrics (user_id, mood_score, stress_level, anxiety_level, depression_score, sleep_quality, social_connections, recorded_at, device_source, accuracy_score)
VALUES
  ('550e8400-e29b-41d4-a716-446655440004', 7.5, 4.2, 3.8, 2.1, 8.2, 7.8, NOW() - INTERVAL '1 day', 'mobile_app', 0.80),
  ('550e8400-e29b-41d4-a716-446655440004', 8.0, 3.5, 3.2, 1.8, 8.5, 8.0, NOW() - INTERVAL '2 days', 'mobile_app', 0.80)
ON CONFLICT DO NOTHING;

-- Insert sample health alerts
INSERT INTO health_alerts (patient_id, alert_type, alert_level, title, message, metric_name, metric_value, threshold_value, is_read, created_at)
VALUES
  ('550e8400-e29b-41d4-a716-446655440004', 'blood_pressure', 'medium', 'Blood Pressure Elevated', 'Your blood pressure reading is slightly elevated. Consider reducing salt intake and increasing physical activity.', 'blood_pressure', '{"systolic": 135, "diastolic": 85}', '{"systolic": 130, "diastolic": 80}', false, NOW() - INTERVAL '2 hours'),
  ('550e8400-e29b-41d4-a716-446655440004', 'sleep_quality', 'low', 'Sleep Quality Improvement', 'Great job! Your sleep quality has improved by 15% this week.', 'sleep_quality', '{"score": 8.5}', '{"score": 7.0}', false, NOW() - INTERVAL '1 day')
ON CONFLICT DO NOTHING;

-- =============================================================================
-- COMPLETION MESSAGE
-- =============================================================================

-- This will show a success message when the script completes
DO $$
BEGIN
  RAISE NOTICE '✅ DoctAI Health Hub Database Setup Complete!';
  RAISE NOTICE '📊 Created tables:';
  RAISE NOTICE '   • health_metrics_timeseries';
  RAISE NOTICE '   • risk_progressions';
  RAISE NOTICE '   • patient_timelines';
  RAISE NOTICE '   • scan_sequences';
  RAISE NOTICE '   • treatment_responses';
  RAISE NOTICE '   • image_metadata';
  RAISE NOTICE '   • healthcare_providers';
  RAISE NOTICE '   • appointments';
  RAISE NOTICE '   • telemedicine_consultations';
  RAISE NOTICE '   • cardiovascular_metrics';
  RAISE NOTICE '   • sleep_metrics';
  RAISE NOTICE '   • fitness_metrics';
  RAISE NOTICE '   • mental_health_metrics';
  RAISE NOTICE '   • health_alerts';
  RAISE NOTICE '   • provider_availability';
  RAISE NOTICE '   • patient_provider_relationships';
  RAISE NOTICE '   • remote_monitoring';
  RAISE NOTICE '';
  RAISE NOTICE '🔑 Sample data has been created for user: 550e8400-e29b-41d4-a716-446655440004';
  RAISE NOTICE '🚀 Your application should now work without console errors!';
END $$;
