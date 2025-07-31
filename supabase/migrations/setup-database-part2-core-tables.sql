-- DoctAI Health Hub Database Setup - Part 2: Core Tables
-- Run this script after Part 1 in your Supabase SQL Editor

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

-- Success message
DO $$
BEGIN
  RAISE NOTICE '✅ Part 2 Complete: Core tables created successfully!';
END $$;
