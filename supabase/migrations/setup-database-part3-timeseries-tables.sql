-- DoctAI Health Hub Database Setup - Part 3: Time-Series Tables
-- Run this script after Part 2 in your Supabase SQL Editor
-- This creates the tables that are causing the 404 errors

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

-- Time-series health metrics table (THIS IS THE MAIN TABLE CAUSING 404 ERRORS)
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

-- Risk progression tracking (THIS IS THE SECOND TABLE CAUSING 404 ERRORS)
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

-- Success message
DO $$
BEGIN
  RAISE NOTICE '✅ Part 3 Complete: Time-series tables created successfully!';
  RAISE NOTICE '🎉 This should fix the 404 errors in your application!';
END $$;
