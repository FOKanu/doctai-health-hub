-- DoctAI Health Hub - Complete Database Setup
-- Run this script in your Supabase SQL Editor to set up all tables and sample data

-- =============================================================================
-- STEP 1: Create Enums
-- =============================================================================

-- Create enum for condition types
DO $$ BEGIN
    CREATE TYPE condition_type AS ENUM (
      'skin_lesion', 'cardiovascular', 'respiratory', 'neurological',
      'metabolic', 'musculoskeletal', 'gastrointestinal', 'endocrine'
    );
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- Create enum for patient status
DO $$ BEGIN
    CREATE TYPE patient_status AS ENUM ('monitoring', 'improving', 'worsening', 'stable');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- Create enum for metric types
DO $$ BEGIN
    CREATE TYPE metric_type AS ENUM (
      'heart_rate', 'blood_pressure', 'weight', 'temperature', 'sleep_hours',
      'steps', 'calories', 'water_intake', 'blood_glucose', 'oxygen_saturation',
      'heart_rate_variability', 'blood_pressure_systolic', 'blood_pressure_diastolic',
      'respiratory_rate', 'blood_glucose_fasting', 'blood_glucose_postprandial',
      'hba1c', 'cholesterol_total', 'cholesterol_hdl', 'cholesterol_ldl',
      'triglycerides', 'sleep_duration', 'sleep_efficiency', 'sleep_latency',
      'vo2_max', 'mood_score', 'stress_level', 'cortisol_morning', 'cortisol_evening',
      'tsh', 'testosterone', 'estrogen'
    );
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- Create enum for analysis types
DO $$ BEGIN
    CREATE TYPE analysis_type AS ENUM ('progression', 'treatment_response', 'baseline', 'follow_up');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- Create enum for consultation status
DO $$ BEGIN
    CREATE TYPE consultation_status AS ENUM ('scheduled', 'in_progress', 'completed', 'cancelled', 'no_show');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- Create enum for consultation type
DO $$ BEGIN
    CREATE TYPE consultation_type AS ENUM ('video', 'audio', 'chat', 'follow_up', 'emergency');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- Create enum for provider specialty
DO $$ BEGIN
    CREATE TYPE provider_specialty AS ENUM (
      'primary_care', 'cardiology', 'dermatology', 'endocrinology', 'neurology',
      'psychiatry', 'orthopedics', 'pediatrics', 'gynecology', 'oncology',
      'pulmonology', 'gastroenterology', 'ophthalmology', 'urology', 'general'
    );
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- Create enum for monitoring alert level
DO $$ BEGIN
    CREATE TYPE monitoring_alert_level AS ENUM ('low', 'medium', 'high', 'critical');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- Create enum for appointment type
DO $$ BEGIN
    CREATE TYPE appointment_type AS ENUM ('consultation', 'follow_up', 'emergency', 'routine_checkup', 'specialist_referral');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- Create enum for image types
DO $$ BEGIN
    CREATE TYPE image_type AS ENUM ('skin_lesion', 'ct_scan', 'mri', 'xray', 'eeg');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- =============================================================================
-- STEP 2: Create Core Tables
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
-- STEP 3: Create Specialized Health Tables
-- =============================================================================

-- Cardiovascular Health Table
CREATE TABLE IF NOT EXISTS cardiovascular_metrics (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  heart_rate_resting INTEGER CHECK (heart_rate_resting >= 30 AND heart_rate_resting <= 200),
  heart_rate_active INTEGER CHECK (heart_rate_active >= 40 AND heart_rate_active <= 220),
  heart_rate_variability DECIMAL(5,2), -- HRV in milliseconds
  blood_pressure_systolic INTEGER CHECK (blood_pressure_systolic >= 70 AND blood_pressure_systolic <= 250),
  blood_pressure_diastolic INTEGER CHECK (blood_pressure_diastolic >= 40 AND blood_pressure_diastolic <= 150),
  pulse_pressure INTEGER,
  mean_arterial_pressure DECIMAL(5,2),
  ecg_rhythm TEXT CHECK (ecg_rhythm IN ('normal', 'irregular', 'atrial_fibrillation', 'bradycardia', 'tachycardia')),
  qt_interval INTEGER, -- milliseconds
  st_segment DECIMAL(4,2), -- millimeters
  recorded_at TIMESTAMP WITH TIME ZONE NOT NULL,
  device_source TEXT,
  accuracy_score DECIMAL(3,2) CHECK (accuracy_score >= 0 AND accuracy_score <= 1),
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Sleep Quality Table
CREATE TABLE IF NOT EXISTS sleep_metrics (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  sleep_date DATE NOT NULL,
  total_duration DECIMAL(4,2), -- hours
  deep_sleep_duration DECIMAL(4,2), -- hours
  rem_sleep_duration DECIMAL(4,2), -- hours
  sleep_efficiency DECIMAL(4,2), -- percentage
  sleep_latency DECIMAL(4,2), -- minutes to fall asleep
  wake_count INTEGER,
  sleep_quality_score DECIMAL(3,1) CHECK (sleep_quality_score >= 0 AND sleep_quality_score <= 10),
  device_source TEXT,
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Fitness Metrics Table
CREATE TABLE IF NOT EXISTS fitness_metrics (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  activity_date DATE NOT NULL,
  steps_count INTEGER,
  calories_burned INTEGER,
  active_minutes INTEGER,
  distance_km DECIMAL(5,2),
  vo2_max DECIMAL(4,2),
  resting_heart_rate INTEGER,
  max_heart_rate INTEGER,
  avg_heart_rate INTEGER,
  device_source TEXT,
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Mental Health Metrics Table
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
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =============================================================================
-- STEP 4: Create Telemedicine Tables
-- =============================================================================

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
  availability_schedule JSONB DEFAULT '{}',
  consultation_fee DECIMAL(8,2),
  rating DECIMAL(3,2) CHECK (rating >= 0 AND rating <= 5),
  total_consultations INTEGER DEFAULT 0,
  is_verified BOOLEAN DEFAULT false,
  is_available BOOLEAN DEFAULT true,
  profile_image_url TEXT,
  bio TEXT,
  contact_info JSONB DEFAULT '{}',
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Telemedicine consultations table
CREATE TABLE IF NOT EXISTS telemedicine_consultations (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  provider_id UUID REFERENCES healthcare_providers(id) ON DELETE CASCADE,
  consultation_type consultation_type NOT NULL,
  status consultation_status NOT NULL DEFAULT 'scheduled',
  scheduled_at TIMESTAMP WITH TIME ZONE NOT NULL,
  started_at TIMESTAMP WITH TIME ZONE,
  ended_at TIMESTAMP WITH TIME ZONE,
  duration_minutes INTEGER,
  meeting_url TEXT,
  meeting_id TEXT,
  consultation_notes TEXT,
  diagnosis TEXT,
  prescriptions JSONB DEFAULT '[]',
  recommendations TEXT[],
  follow_up_date TIMESTAMP WITH TIME ZONE,
  follow_up_required BOOLEAN DEFAULT false,
  emergency_contact TEXT,
  symptoms TEXT[],
  vital_signs JSONB DEFAULT '{}',
  attachments JSONB DEFAULT '[]',
  metadata JSONB DEFAULT '{}',
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
  status consultation_status NOT NULL DEFAULT 'scheduled',
  notes TEXT,
  location TEXT,
  duration_minutes INTEGER DEFAULT 30,
  is_telemedicine BOOLEAN DEFAULT false,
  meeting_url TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Health alerts and notifications table
CREATE TABLE IF NOT EXISTS health_alerts (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  provider_id UUID REFERENCES healthcare_providers(id),
  alert_type TEXT NOT NULL,
  alert_level monitoring_alert_level NOT NULL,
  title TEXT NOT NULL,
  message TEXT NOT NULL,
  metric_name TEXT,
  metric_value JSONB,
  threshold_value JSONB,
  is_read BOOLEAN DEFAULT false,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =============================================================================
-- STEP 5: Create Medical Image Tables
-- =============================================================================

-- Create image_metadata table
CREATE TABLE IF NOT EXISTS image_metadata (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    url TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    type image_type NOT NULL,
    analysis_result JSONB,
    metadata JSONB,
    CONSTRAINT valid_url CHECK (url ~ '^https?://')
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

-- =============================================================================
-- STEP 6: Create Indexes
-- =============================================================================

-- Core tables indexes
CREATE INDEX IF NOT EXISTS idx_patient_timelines_user_id ON patient_timelines(user_id);
CREATE INDEX IF NOT EXISTS idx_patient_timelines_condition_type ON patient_timelines(condition_type);
CREATE INDEX IF NOT EXISTS idx_patient_timelines_baseline_date ON patient_timelines(baseline_date);
CREATE INDEX IF NOT EXISTS idx_patient_timelines_status ON patient_timelines(status);

CREATE INDEX IF NOT EXISTS idx_health_metrics_user_id ON health_metrics_timeseries(user_id);
CREATE INDEX IF NOT EXISTS idx_health_metrics_type ON health_metrics_timeseries(metric_type);
CREATE INDEX IF NOT EXISTS idx_health_metrics_recorded_at ON health_metrics_timeseries(recorded_at);
CREATE INDEX IF NOT EXISTS idx_health_metrics_user_type_time ON health_metrics_timeseries(user_id, metric_type, recorded_at);

CREATE INDEX IF NOT EXISTS idx_risk_progressions_user_id ON risk_progressions(user_id);
CREATE INDEX IF NOT EXISTS idx_risk_progressions_condition_type ON risk_progressions(condition_type);
CREATE INDEX IF NOT EXISTS idx_risk_progressions_recorded_at ON risk_progressions(recorded_at);
CREATE INDEX IF NOT EXISTS idx_risk_progressions_risk_level ON risk_progressions(risk_level);

-- Specialized health tables indexes
CREATE INDEX IF NOT EXISTS idx_cardiovascular_user_date ON cardiovascular_metrics(user_id, recorded_at);
CREATE INDEX IF NOT EXISTS idx_sleep_user_date ON sleep_metrics(user_id, sleep_date);
CREATE INDEX IF NOT EXISTS idx_fitness_user_date ON fitness_metrics(user_id, activity_date);
CREATE INDEX IF NOT EXISTS idx_mental_health_user_date ON mental_health_metrics(user_id, recorded_at);

-- Telemedicine tables indexes
CREATE INDEX IF NOT EXISTS idx_healthcare_providers_specialty ON healthcare_providers(specialty);
CREATE INDEX IF NOT EXISTS idx_healthcare_providers_available ON healthcare_providers(is_available);
CREATE INDEX IF NOT EXISTS idx_telemedicine_consultations_patient ON telemedicine_consultations(patient_id);
CREATE INDEX IF NOT EXISTS idx_telemedicine_consultations_provider ON telemedicine_consultations(provider_id);
CREATE INDEX IF NOT EXISTS idx_telemedicine_consultations_status ON telemedicine_consultations(status);
CREATE INDEX IF NOT EXISTS idx_telemedicine_consultations_scheduled ON telemedicine_consultations(scheduled_at);
CREATE INDEX IF NOT EXISTS idx_appointments_patient ON appointments(patient_id);
CREATE INDEX IF NOT EXISTS idx_appointments_provider ON appointments(provider_id);
CREATE INDEX IF NOT EXISTS idx_appointments_scheduled ON appointments(scheduled_at);
CREATE INDEX IF NOT EXISTS idx_health_alerts_patient ON health_alerts(patient_id);
CREATE INDEX IF NOT EXISTS idx_health_alerts_level ON health_alerts(alert_level);
CREATE INDEX IF NOT EXISTS idx_health_alerts_unread ON health_alerts(is_read);

-- Image tables indexes
CREATE INDEX IF NOT EXISTS idx_image_metadata_user_id ON image_metadata(user_id);
CREATE INDEX IF NOT EXISTS idx_image_metadata_type ON image_metadata(type);
CREATE INDEX IF NOT EXISTS idx_image_metadata_created_at ON image_metadata(created_at);
CREATE INDEX IF NOT EXISTS idx_scan_sequences_user_id ON scan_sequences(user_id);
CREATE INDEX IF NOT EXISTS idx_scan_sequences_analysis_type ON scan_sequences(analysis_type);
CREATE INDEX IF NOT EXISTS idx_scan_sequences_created_at ON scan_sequences(created_at);

-- =============================================================================
-- STEP 7: Set up Row Level Security (RLS)
-- =============================================================================

-- Enable RLS on all tables
ALTER TABLE patient_timelines ENABLE ROW LEVEL SECURITY;
ALTER TABLE health_metrics_timeseries ENABLE ROW LEVEL SECURITY;
ALTER TABLE risk_progressions ENABLE ROW LEVEL SECURITY;
ALTER TABLE cardiovascular_metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE sleep_metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE fitness_metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE mental_health_metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE healthcare_providers ENABLE ROW LEVEL SECURITY;
ALTER TABLE telemedicine_consultations ENABLE ROW LEVEL SECURITY;
ALTER TABLE appointments ENABLE ROW LEVEL SECURITY;
ALTER TABLE health_alerts ENABLE ROW LEVEL SECURITY;
ALTER TABLE image_metadata ENABLE ROW LEVEL SECURITY;
ALTER TABLE scan_sequences ENABLE ROW LEVEL SECURITY;

-- Create RLS policies for patient_timelines
DROP POLICY IF EXISTS "Users can view their own timelines" ON patient_timelines;
CREATE POLICY "Users can view their own timelines"
  ON patient_timelines FOR SELECT
  USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can insert their own timelines" ON patient_timelines;
CREATE POLICY "Users can insert their own timelines"
  ON patient_timelines FOR INSERT
  WITH CHECK (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can update their own timelines" ON patient_timelines;
CREATE POLICY "Users can update their own timelines"
  ON patient_timelines FOR UPDATE
  USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can delete their own timelines" ON patient_timelines;
CREATE POLICY "Users can delete their own timelines"
  ON patient_timelines FOR DELETE
  USING (auth.uid() = user_id);

-- Create RLS policies for health_metrics_timeseries
DROP POLICY IF EXISTS "Users can view their own health metrics" ON health_metrics_timeseries;
CREATE POLICY "Users can view their own health metrics"
  ON health_metrics_timeseries FOR SELECT
  USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can insert their own health metrics" ON health_metrics_timeseries;
CREATE POLICY "Users can insert their own health metrics"
  ON health_metrics_timeseries FOR INSERT
  WITH CHECK (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can update their own health metrics" ON health_metrics_timeseries;
CREATE POLICY "Users can update their own health metrics"
  ON health_metrics_timeseries FOR UPDATE
  USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can delete their own health metrics" ON health_metrics_timeseries;
CREATE POLICY "Users can delete their own health metrics"
  ON health_metrics_timeseries FOR DELETE
  USING (auth.uid() = user_id);

-- Create RLS policies for risk_progressions
DROP POLICY IF EXISTS "Users can view their own risk progressions" ON risk_progressions;
CREATE POLICY "Users can view their own risk progressions"
  ON risk_progressions FOR SELECT
  USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can insert their own risk progressions" ON risk_progressions;
CREATE POLICY "Users can insert their own risk progressions"
  ON risk_progressions FOR INSERT
  WITH CHECK (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can update their own risk progressions" ON risk_progressions;
CREATE POLICY "Users can update their own risk progressions"
  ON risk_progressions FOR UPDATE
  USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can delete their own risk progressions" ON risk_progressions;
CREATE POLICY "Users can delete their own risk progressions"
  ON risk_progressions FOR DELETE
  USING (auth.uid() = user_id);

-- Create RLS policies for healthcare_providers
DROP POLICY IF EXISTS "Users can view their own provider profile" ON healthcare_providers;
CREATE POLICY "Users can view their own provider profile"
  ON healthcare_providers FOR SELECT
  USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can insert their own provider profile" ON healthcare_providers;
CREATE POLICY "Users can insert their own provider profile"
  ON healthcare_providers FOR INSERT
  WITH CHECK (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can update their own provider profile" ON healthcare_providers;
CREATE POLICY "Users can update their own provider profile"
  ON healthcare_providers FOR UPDATE
  USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can delete their own provider profile" ON healthcare_providers;
CREATE POLICY "Users can delete their own provider profile"
  ON healthcare_providers FOR DELETE
  USING (auth.uid() = user_id);

-- Create RLS policies for telemedicine_consultations
DROP POLICY IF EXISTS "Users can view their own consultations" ON telemedicine_consultations;
CREATE POLICY "Users can view their own consultations"
  ON telemedicine_consultations FOR SELECT
  USING (auth.uid() = patient_id OR auth.uid() = (SELECT user_id FROM healthcare_providers WHERE id = provider_id));

DROP POLICY IF EXISTS "Users can insert their own consultations" ON telemedicine_consultations;
CREATE POLICY "Users can insert their own consultations"
  ON telemedicine_consultations FOR INSERT
  WITH CHECK (auth.uid() = patient_id OR auth.uid() = (SELECT user_id FROM healthcare_providers WHERE id = provider_id));

DROP POLICY IF EXISTS "Users can update their own consultations" ON telemedicine_consultations;
CREATE POLICY "Users can update their own consultations"
  ON telemedicine_consultations FOR UPDATE
  USING (auth.uid() = patient_id OR auth.uid() = (SELECT user_id FROM healthcare_providers WHERE id = provider_id));

DROP POLICY IF EXISTS "Users can delete their own consultations" ON telemedicine_consultations;
CREATE POLICY "Users can delete their own consultations"
  ON telemedicine_consultations FOR DELETE
  USING (auth.uid() = patient_id OR auth.uid() = (SELECT user_id FROM healthcare_providers WHERE id = provider_id));

-- Create RLS policies for appointments
DROP POLICY IF EXISTS "Users can view their own appointments" ON appointments;
CREATE POLICY "Users can view their own appointments"
  ON appointments FOR SELECT
  USING (auth.uid() = patient_id OR auth.uid() = (SELECT user_id FROM healthcare_providers WHERE id = provider_id));

DROP POLICY IF EXISTS "Users can insert their own appointments" ON appointments;
CREATE POLICY "Users can insert their own appointments"
  ON appointments FOR INSERT
  WITH CHECK (auth.uid() = patient_id OR auth.uid() = (SELECT user_id FROM healthcare_providers WHERE id = provider_id));

DROP POLICY IF EXISTS "Users can update their own appointments" ON appointments;
CREATE POLICY "Users can update their own appointments"
  ON appointments FOR UPDATE
  USING (auth.uid() = patient_id OR auth.uid() = (SELECT user_id FROM healthcare_providers WHERE id = provider_id));

DROP POLICY IF EXISTS "Users can delete their own appointments" ON appointments;
CREATE POLICY "Users can delete their own appointments"
  ON appointments FOR DELETE
  USING (auth.uid() = patient_id OR auth.uid() = (SELECT user_id FROM healthcare_providers WHERE id = provider_id));

-- Create RLS policies for health_alerts
DROP POLICY IF EXISTS "Users can view their own health alerts" ON health_alerts;
CREATE POLICY "Users can view their own health alerts"
  ON health_alerts FOR SELECT
  USING (auth.uid() = patient_id OR auth.uid() = (SELECT user_id FROM healthcare_providers WHERE id = provider_id));

DROP POLICY IF EXISTS "Users can insert their own health alerts" ON health_alerts;
CREATE POLICY "Users can insert their own health alerts"
  ON health_alerts FOR INSERT
  WITH CHECK (auth.uid() = patient_id OR auth.uid() = (SELECT user_id FROM healthcare_providers WHERE id = provider_id));

DROP POLICY IF EXISTS "Users can update their own health alerts" ON health_alerts;
CREATE POLICY "Users can update their own health alerts"
  ON health_alerts FOR UPDATE
  USING (auth.uid() = patient_id OR auth.uid() = (SELECT user_id FROM healthcare_providers WHERE id = provider_id));

DROP POLICY IF EXISTS "Users can delete their own health alerts" ON health_alerts;
CREATE POLICY "Users can delete their own health alerts"
  ON health_alerts FOR DELETE
  USING (auth.uid() = patient_id OR auth.uid() = (SELECT user_id FROM healthcare_providers WHERE id = provider_id));

-- Create RLS policies for image_metadata
DROP POLICY IF EXISTS "Users can view their own images" ON image_metadata;
CREATE POLICY "Users can view their own images"
  ON image_metadata FOR SELECT
  USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can insert their own images" ON image_metadata;
CREATE POLICY "Users can insert their own images"
  ON image_metadata FOR INSERT
  WITH CHECK (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can update their own images" ON image_metadata;
CREATE POLICY "Users can update their own images"
  ON image_metadata FOR UPDATE
  USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can delete their own images" ON image_metadata;
CREATE POLICY "Users can delete their own images"
  ON image_metadata FOR DELETE
  USING (auth.uid() = user_id);

-- Create RLS policies for scan_sequences
DROP POLICY IF EXISTS "Users can view their own scan sequences" ON scan_sequences;
CREATE POLICY "Users can view their own scan sequences"
  ON scan_sequences FOR SELECT
  USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can insert their own scan sequences" ON scan_sequences;
CREATE POLICY "Users can insert their own scan sequences"
  ON scan_sequences FOR INSERT
  WITH CHECK (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can update their own scan sequences" ON scan_sequences;
CREATE POLICY "Users can update their own scan sequences"
  ON scan_sequences FOR UPDATE
  USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can delete their own scan sequences" ON scan_sequences;
CREATE POLICY "Users can delete their own scan sequences"
  ON scan_sequences FOR DELETE
  USING (auth.uid() = user_id);

-- =============================================================================
-- STEP 8: Create Triggers
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
DROP TRIGGER IF EXISTS update_patient_timelines_updated_at ON patient_timelines;
CREATE TRIGGER update_patient_timelines_updated_at
  BEFORE UPDATE ON patient_timelines
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_scan_sequences_updated_at ON scan_sequences;
CREATE TRIGGER update_scan_sequences_updated_at
  BEFORE UPDATE ON scan_sequences
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_healthcare_providers_updated_at ON healthcare_providers;
CREATE TRIGGER update_healthcare_providers_updated_at
  BEFORE UPDATE ON healthcare_providers
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_telemedicine_consultations_updated_at ON telemedicine_consultations;
CREATE TRIGGER update_telemedicine_consultations_updated_at
  BEFORE UPDATE ON telemedicine_consultations
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_appointments_updated_at ON appointments;
CREATE TRIGGER update_appointments_updated_at
  BEFORE UPDATE ON appointments
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_health_alerts_updated_at ON health_alerts;
CREATE TRIGGER update_health_alerts_updated_at
  BEFORE UPDATE ON health_alerts
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- =============================================================================
-- STEP 9: Insert Sample Data
-- =============================================================================

-- Insert sample health metrics for the mock user (replace with actual user ID)
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

-- Insert sample health alerts
INSERT INTO health_alerts (patient_id, alert_type, alert_level, title, message, metric_name, metric_value, threshold_value, is_read, created_at)
VALUES
  ('550e8400-e29b-41d4-a716-446655440004', 'blood_pressure', 'medium', 'Blood Pressure Elevated', 'Your blood pressure reading is slightly elevated. Consider reducing salt intake and increasing physical activity.', 'blood_pressure', '{"systolic": 135, "diastolic": 85}', '{"systolic": 130, "diastolic": 80}', false, NOW() - INTERVAL '2 hours'),
  ('550e8400-e29b-41d4-a716-446655440004', 'sleep_quality', 'low', 'Sleep Quality Improvement', 'Great job! Your sleep quality has improved by 15% this week.', 'sleep_quality', '{"score": 8.5}', '{"score": 7.0}', false, NOW() - INTERVAL '1 day')
ON CONFLICT DO NOTHING;

-- =============================================================================
-- STEP 10: Verification
-- =============================================================================

-- Verify tables were created
SELECT 'Tables created successfully!' as status;

-- Count sample data
SELECT
  'health_metrics_timeseries' as table_name,
  COUNT(*) as record_count
FROM health_metrics_timeseries
UNION ALL
SELECT
  'risk_progressions' as table_name,
  COUNT(*) as record_count
FROM risk_progressions
UNION ALL
SELECT
  'patient_timelines' as table_name,
  COUNT(*) as record_count
FROM patient_timelines
UNION ALL
SELECT
  'cardiovascular_metrics' as table_name,
  COUNT(*) as record_count
FROM cardiovascular_metrics
UNION ALL
SELECT
  'sleep_metrics' as table_name,
  COUNT(*) as record_count
FROM sleep_metrics
UNION ALL
SELECT
  'fitness_metrics' as table_name,
  COUNT(*) as record_count
FROM fitness_metrics
UNION ALL
SELECT
  'mental_health_metrics' as table_name,
  COUNT(*) as record_count
FROM mental_health_metrics
UNION ALL
SELECT
  'image_metadata' as table_name,
  COUNT(*) as record_count
FROM image_metadata
UNION ALL
SELECT
  'scan_sequences' as table_name,
  COUNT(*) as record_count
FROM scan_sequences
UNION ALL
SELECT
  'health_alerts' as table_name,
  COUNT(*) as record_count
FROM health_alerts;
