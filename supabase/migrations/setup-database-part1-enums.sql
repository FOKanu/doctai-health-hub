-- DoctAI Health Hub Database Setup - Part 1: Enums and Types
-- Run this script first in your Supabase SQL Editor

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

-- Success message
DO $$
BEGIN
  RAISE NOTICE '✅ Part 1 Complete: Enums and Types created successfully!';
END $$;
