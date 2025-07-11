-- Telemedicine Integration Database Schema
-- Enables remote health monitoring and virtual consultations

-- Create enums for telemedicine features
CREATE TYPE consultation_status AS ENUM ('scheduled', 'in_progress', 'completed', 'cancelled', 'no_show');
CREATE TYPE consultation_type AS ENUM ('video', 'audio', 'chat', 'follow_up', 'emergency');
CREATE TYPE provider_specialty AS ENUM (
  'primary_care', 'cardiology', 'dermatology', 'endocrinology', 'neurology',
  'psychiatry', 'orthopedics', 'pediatrics', 'gynecology', 'oncology',
  'pulmonology', 'gastroenterology', 'ophthalmology', 'urology', 'general'
);
CREATE TYPE monitoring_alert_level AS ENUM ('low', 'medium', 'high', 'critical');
CREATE TYPE appointment_type AS ENUM ('consultation', 'follow_up', 'emergency', 'routine_checkup', 'specialist_referral');

-- Healthcare providers table
CREATE TABLE healthcare_providers (
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
CREATE TABLE telemedicine_consultations (
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

-- Remote health monitoring table
CREATE TABLE remote_monitoring (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  provider_id UUID REFERENCES healthcare_providers(id),
  monitoring_type TEXT NOT NULL,
  device_id TEXT,
  device_type TEXT,
  is_active BOOLEAN DEFAULT true,
  start_date TIMESTAMP WITH TIME ZONE NOT NULL,
  end_date TIMESTAMP WITH TIME ZONE,
  monitoring_frequency TEXT, -- 'continuous', 'daily', 'weekly', 'custom'
  alert_thresholds JSONB DEFAULT '{}',
  last_reading_at TIMESTAMP WITH TIME ZONE,
  last_reading_value JSONB,
  alert_level monitoring_alert_level,
  is_alert_active BOOLEAN DEFAULT false,
  alert_message TEXT,
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Health alerts and notifications table
CREATE TABLE health_alerts (
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
  is_acknowledged BOOLEAN DEFAULT false,
  acknowledged_at TIMESTAMP WITH TIME ZONE,
  acknowledged_by UUID REFERENCES auth.users(id),
  action_taken TEXT,
  follow_up_required BOOLEAN DEFAULT false,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Appointment scheduling table
CREATE TABLE appointments (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  provider_id UUID REFERENCES healthcare_providers(id) ON DELETE CASCADE,
  appointment_type appointment_type NOT NULL,
  scheduled_at TIMESTAMP WITH TIME ZONE NOT NULL,
  duration_minutes INTEGER DEFAULT 30,
  status consultation_status NOT NULL DEFAULT 'scheduled',
  reason TEXT,
  symptoms TEXT[],
  is_urgent BOOLEAN DEFAULT false,
  reminder_sent BOOLEAN DEFAULT false,
  reminder_sent_at TIMESTAMP WITH TIME ZONE,
  notes TEXT,
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Provider availability table
CREATE TABLE provider_availability (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  provider_id UUID REFERENCES healthcare_providers(id) ON DELETE CASCADE,
  day_of_week INTEGER CHECK (day_of_week >= 0 AND day_of_week <= 6), -- 0 = Sunday
  start_time TIME NOT NULL,
  end_time TIME NOT NULL,
  is_available BOOLEAN DEFAULT true,
  consultation_type consultation_type,
  max_patients_per_slot INTEGER DEFAULT 1,
  slot_duration_minutes INTEGER DEFAULT 30,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Patient-provider relationships table
CREATE TABLE patient_provider_relationships (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  provider_id UUID REFERENCES healthcare_providers(id) ON DELETE CASCADE,
  relationship_type TEXT NOT NULL, -- 'primary_care', 'specialist', 'consultant'
  start_date TIMESTAMP WITH TIME ZONE NOT NULL,
  end_date TIMESTAMP WITH TIME ZONE,
  is_active BOOLEAN DEFAULT true,
  notes TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  UNIQUE(patient_id, provider_id, relationship_type)
);

-- Telemedicine session recordings table
CREATE TABLE consultation_recordings (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  consultation_id UUID REFERENCES telemedicine_consultations(id) ON DELETE CASCADE,
  recording_url TEXT NOT NULL,
  recording_type TEXT NOT NULL, -- 'video', 'audio', 'screen_share'
  file_size_bytes BIGINT,
  duration_seconds INTEGER,
  is_encrypted BOOLEAN DEFAULT true,
  access_level TEXT DEFAULT 'provider_patient', -- 'provider_only', 'provider_patient', 'public'
  expires_at TIMESTAMP WITH TIME ZONE,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX idx_healthcare_providers_specialty ON healthcare_providers(specialty);
CREATE INDEX idx_healthcare_providers_available ON healthcare_providers(is_available);
CREATE INDEX idx_telemedicine_consultations_patient ON telemedicine_consultations(patient_id);
CREATE INDEX idx_telemedicine_consultations_provider ON telemedicine_consultations(provider_id);
CREATE INDEX idx_telemedicine_consultations_status ON telemedicine_consultations(status);
CREATE INDEX idx_telemedicine_consultations_scheduled ON telemedicine_consultations(scheduled_at);
CREATE INDEX idx_remote_monitoring_patient ON remote_monitoring(patient_id);
CREATE INDEX idx_remote_monitoring_active ON remote_monitoring(is_active);
CREATE INDEX idx_health_alerts_patient ON health_alerts(patient_id);
CREATE INDEX idx_health_alerts_level ON health_alerts(alert_level);
CREATE INDEX idx_health_alerts_unread ON health_alerts(is_read);
CREATE INDEX idx_appointments_patient ON appointments(patient_id);
CREATE INDEX idx_appointments_provider ON appointments(provider_id);
CREATE INDEX idx_appointments_scheduled ON appointments(scheduled_at);
CREATE INDEX idx_provider_availability_provider ON provider_availability(provider_id);
CREATE INDEX idx_patient_provider_relationships_patient ON patient_provider_relationships(patient_id);
CREATE INDEX idx_patient_provider_relationships_provider ON patient_provider_relationships(provider_id);

-- Enable RLS on all tables
ALTER TABLE healthcare_providers ENABLE ROW LEVEL SECURITY;
ALTER TABLE telemedicine_consultations ENABLE ROW LEVEL SECURITY;
ALTER TABLE remote_monitoring ENABLE ROW LEVEL SECURITY;
ALTER TABLE health_alerts ENABLE ROW LEVEL SECURITY;
ALTER TABLE appointments ENABLE ROW LEVEL SECURITY;
ALTER TABLE provider_availability ENABLE ROW LEVEL SECURITY;
ALTER TABLE patient_provider_relationships ENABLE ROW LEVEL SECURITY;
ALTER TABLE consultation_recordings ENABLE ROW LEVEL SECURITY;

-- Create RLS policies for healthcare_providers
CREATE POLICY "Providers can view their own profiles"
  ON healthcare_providers FOR SELECT
  USING (auth.uid() = user_id);

CREATE POLICY "Providers can update their own profiles"
  ON healthcare_providers FOR UPDATE
  USING (auth.uid() = user_id);

CREATE POLICY "Patients can view available providers"
  ON healthcare_providers FOR SELECT
  USING (is_available = true AND is_verified = true);

-- Create RLS policies for telemedicine_consultations
CREATE POLICY "Patients can view their own consultations"
  ON telemedicine_consultations FOR SELECT
  USING (auth.uid() = patient_id);

CREATE POLICY "Providers can view consultations they're involved in"
  ON telemedicine_consultations FOR SELECT
  USING (provider_id IN (
    SELECT id FROM healthcare_providers WHERE user_id = auth.uid()
  ));

CREATE POLICY "Patients can create consultations"
  ON telemedicine_consultations FOR INSERT
  WITH CHECK (auth.uid() = patient_id);

CREATE POLICY "Providers can update consultations they're involved in"
  ON telemedicine_consultations FOR UPDATE
  USING (provider_id IN (
    SELECT id FROM healthcare_providers WHERE user_id = auth.uid()
  ));

-- Create RLS policies for remote_monitoring
CREATE POLICY "Patients can view their own monitoring"
  ON remote_monitoring FOR SELECT
  USING (auth.uid() = patient_id);

CREATE POLICY "Providers can view monitoring for their patients"
  ON remote_monitoring FOR SELECT
  USING (provider_id IN (
    SELECT id FROM healthcare_providers WHERE user_id = auth.uid()
  ));

CREATE POLICY "Patients can manage their own monitoring"
  ON remote_monitoring FOR ALL
  USING (auth.uid() = patient_id);

-- Create RLS policies for health_alerts
CREATE POLICY "Patients can view their own alerts"
  ON health_alerts FOR SELECT
  USING (auth.uid() = patient_id);

CREATE POLICY "Providers can view alerts for their patients"
  ON health_alerts FOR SELECT
  USING (provider_id IN (
    SELECT id FROM healthcare_providers WHERE user_id = auth.uid()
  ));

CREATE POLICY "Patients can update their own alerts"
  ON health_alerts FOR UPDATE
  USING (auth.uid() = patient_id);

-- Create RLS policies for appointments
CREATE POLICY "Patients can view their own appointments"
  ON appointments FOR SELECT
  USING (auth.uid() = patient_id);

CREATE POLICY "Providers can view appointments they're involved in"
  ON appointments FOR SELECT
  USING (provider_id IN (
    SELECT id FROM healthcare_providers WHERE user_id = auth.uid()
  ));

CREATE POLICY "Patients can create appointments"
  ON appointments FOR INSERT
  WITH CHECK (auth.uid() = patient_id);

CREATE POLICY "Providers can update appointments they're involved in"
  ON appointments FOR UPDATE
  USING (provider_id IN (
    SELECT id FROM healthcare_providers WHERE user_id = auth.uid()
  ));

-- Create RLS policies for provider_availability
CREATE POLICY "Providers can manage their own availability"
  ON provider_availability FOR ALL
  USING (provider_id IN (
    SELECT id FROM healthcare_providers WHERE user_id = auth.uid()
  ));

CREATE POLICY "Patients can view provider availability"
  ON provider_availability FOR SELECT
  USING (true);

-- Create RLS policies for patient_provider_relationships
CREATE POLICY "Patients can view their provider relationships"
  ON patient_provider_relationships FOR SELECT
  USING (auth.uid() = patient_id);

CREATE POLICY "Providers can view their patient relationships"
  ON patient_provider_relationships FOR SELECT
  USING (provider_id IN (
    SELECT id FROM healthcare_providers WHERE user_id = auth.uid()
  ));

-- Create RLS policies for consultation_recordings
CREATE POLICY "Patients can view their consultation recordings"
  ON consultation_recordings FOR SELECT
  USING (consultation_id IN (
    SELECT id FROM telemedicine_consultations WHERE patient_id = auth.uid()
  ));

CREATE POLICY "Providers can view recordings for their consultations"
  ON consultation_recordings FOR SELECT
  USING (consultation_id IN (
    SELECT id FROM telemedicine_consultations WHERE provider_id IN (
      SELECT id FROM healthcare_providers WHERE user_id = auth.uid()
    )
  ));

-- Create functions for telemedicine features

-- Function to get available appointment slots
CREATE OR REPLACE FUNCTION get_available_appointment_slots(
  p_provider_id UUID,
  p_date DATE,
  p_duration_minutes INTEGER DEFAULT 30
)
RETURNS TABLE (
  start_time TIME,
  end_time TIME,
  is_available BOOLEAN
) AS $$
BEGIN
  RETURN QUERY
  SELECT
    pa.start_time,
    pa.start_time + INTERVAL '1 minute' * pa.slot_duration_minutes as end_time,
    CASE
      WHEN EXISTS (
        SELECT 1 FROM appointments a
        WHERE a.provider_id = p_provider_id
          AND a.scheduled_at::date = p_date
          AND a.scheduled_at::time >= pa.start_time
          AND a.scheduled_at::time < pa.start_time + INTERVAL '1 minute' * pa.slot_duration_minutes
          AND a.status NOT IN ('cancelled', 'no_show')
      ) THEN false
      ELSE true
    END as is_available
  FROM provider_availability pa
  WHERE pa.provider_id = p_provider_id
    AND pa.day_of_week = EXTRACT(DOW FROM p_date)
    AND pa.is_available = true
    AND pa.slot_duration_minutes >= p_duration_minutes
  ORDER BY pa.start_time;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to get patient health summary for providers
CREATE OR REPLACE FUNCTION get_patient_health_summary(
  p_patient_id UUID,
  p_days INTEGER DEFAULT 30
)
RETURNS TABLE (
  recent_consultations INTEGER,
  active_monitoring_count INTEGER,
  unread_alerts INTEGER,
  last_consultation_date TIMESTAMP WITH TIME ZONE,
  health_score DECIMAL(3,2),
  risk_level TEXT
) AS $$
BEGIN
  RETURN QUERY
  SELECT
    COUNT(tc.id)::INTEGER as recent_consultations,
    COUNT(rm.id)::INTEGER as active_monitoring_count,
    COUNT(ha.id)::INTEGER as unread_alerts,
    MAX(tc.scheduled_at) as last_consultation_date,
    AVG(COALESCE(rm.last_reading_value->>'health_score', '0')::DECIMAL) as health_score,
    CASE
      WHEN COUNT(ha.id) > 5 THEN 'high'
      WHEN COUNT(ha.id) > 2 THEN 'medium'
      ELSE 'low'
    END as risk_level
  FROM auth.users u
  LEFT JOIN telemedicine_consultations tc ON u.id = tc.patient_id
    AND tc.scheduled_at >= NOW() - INTERVAL '1 day' * p_days
  LEFT JOIN remote_monitoring rm ON u.id = rm.patient_id
    AND rm.is_active = true
  LEFT JOIN health_alerts ha ON u.id = ha.patient_id
    AND ha.is_read = false
  WHERE u.id = p_patient_id
  GROUP BY u.id;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to create health alert
CREATE OR REPLACE FUNCTION create_health_alert(
  p_patient_id UUID,
  p_alert_type TEXT,
  p_alert_level monitoring_alert_level,
  p_title TEXT,
  p_message TEXT,
  p_metric_name TEXT DEFAULT NULL,
  p_metric_value JSONB DEFAULT NULL,
  p_threshold_value JSONB DEFAULT NULL
)
RETURNS UUID AS $$
DECLARE
  v_alert_id UUID;
BEGIN
  INSERT INTO health_alerts (
    patient_id,
    alert_type,
    alert_level,
    title,
    message,
    metric_name,
    metric_value,
    threshold_value
  ) VALUES (
    p_patient_id,
    p_alert_type,
    p_alert_level,
    p_title,
    p_message,
    p_metric_name,
    p_metric_value,
    p_threshold_value
  ) RETURNING id INTO v_alert_id;

  RETURN v_alert_id;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to get consultation statistics
CREATE OR REPLACE FUNCTION get_consultation_stats(
  p_provider_id UUID,
  p_start_date DATE,
  p_end_date DATE
)
RETURNS TABLE (
  total_consultations INTEGER,
  completed_consultations INTEGER,
  cancelled_consultations INTEGER,
  average_duration_minutes DECIMAL(5,2),
  total_revenue DECIMAL(10,2)
) AS $$
BEGIN
  RETURN QUERY
  SELECT
    COUNT(tc.id)::INTEGER as total_consultations,
    COUNT(CASE WHEN tc.status = 'completed' THEN 1 END)::INTEGER as completed_consultations,
    COUNT(CASE WHEN tc.status = 'cancelled' THEN 1 END)::INTEGER as cancelled_consultations,
    AVG(tc.duration_minutes) as average_duration_minutes,
    COUNT(CASE WHEN tc.status = 'completed' THEN 1 END) * hp.consultation_fee as total_revenue
  FROM telemedicine_consultations tc
  JOIN healthcare_providers hp ON tc.provider_id = hp.id
  WHERE tc.provider_id = p_provider_id
    AND tc.scheduled_at::date BETWEEN p_start_date AND p_end_date;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Create triggers for updated_at
CREATE TRIGGER update_healthcare_providers_updated_at
  BEFORE UPDATE ON healthcare_providers
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_telemedicine_consultations_updated_at
  BEFORE UPDATE ON telemedicine_consultations
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_remote_monitoring_updated_at
  BEFORE UPDATE ON remote_monitoring
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_health_alerts_updated_at
  BEFORE UPDATE ON health_alerts
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_appointments_updated_at
  BEFORE UPDATE ON appointments
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_provider_availability_updated_at
  BEFORE UPDATE ON provider_availability
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_patient_provider_relationships_updated_at
  BEFORE UPDATE ON patient_provider_relationships
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
