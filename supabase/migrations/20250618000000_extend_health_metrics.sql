-- Extended Health Metrics Migration
-- Adds comprehensive health monitoring capabilities

-- Extend the existing metric_type enum
ALTER TYPE metric_type ADD VALUE 'heart_rate_variability';
ALTER TYPE metric_type ADD VALUE 'blood_pressure_systolic';
ALTER TYPE metric_type ADD VALUE 'blood_pressure_diastolic';
ALTER TYPE metric_type ADD VALUE 'respiratory_rate';
ALTER TYPE metric_type ADD VALUE 'oxygen_saturation';
ALTER TYPE metric_type ADD VALUE 'blood_glucose_fasting';
ALTER TYPE metric_type ADD VALUE 'blood_glucose_postprandial';
ALTER TYPE metric_type ADD VALUE 'hba1c';
ALTER TYPE metric_type ADD VALUE 'cholesterol_total';
ALTER TYPE metric_type ADD VALUE 'cholesterol_hdl';
ALTER TYPE metric_type ADD VALUE 'cholesterol_ldl';
ALTER TYPE metric_type ADD VALUE 'triglycerides';
ALTER TYPE metric_type ADD VALUE 'sleep_duration';
ALTER TYPE metric_type ADD VALUE 'sleep_efficiency';
ALTER TYPE metric_type ADD VALUE 'sleep_latency';
ALTER TYPE metric_type ADD VALUE 'vo2_max';
ALTER TYPE metric_type ADD VALUE 'mood_score';
ALTER TYPE metric_type ADD VALUE 'stress_level';
ALTER TYPE metric_type ADD VALUE 'cortisol_morning';
ALTER TYPE metric_type ADD VALUE 'cortisol_evening';
ALTER TYPE metric_type ADD VALUE 'tsh';
ALTER TYPE metric_type ADD VALUE 'testosterone';
ALTER TYPE metric_type ADD VALUE 'estrogen';

-- Create specialized health monitoring tables

-- Cardiovascular Health Table
CREATE TABLE cardiovascular_metrics (
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

-- Respiratory Health Table
CREATE TABLE respiratory_metrics (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  respiratory_rate INTEGER CHECK (respiratory_rate >= 8 AND respiratory_rate <= 40),
  oxygen_saturation DECIMAL(4,2) CHECK (oxygen_saturation >= 70 AND oxygen_saturation <= 100),
  peak_flow INTEGER, -- L/min
  forced_expiratory_volume DECIMAL(4,2), -- FEV1 in liters
  lung_capacity DECIMAL(4,2), -- Total lung capacity in liters
  breathing_pattern TEXT CHECK (breathing_pattern IN ('normal', 'shallow', 'rapid', 'irregular', 'labored')),
  recorded_at TIMESTAMP WITH TIME ZONE NOT NULL,
  device_source TEXT,
  accuracy_score DECIMAL(3,2) CHECK (accuracy_score >= 0 AND accuracy_score <= 1),
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Metabolic Health Table
CREATE TABLE metabolic_metrics (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  blood_glucose_fasting DECIMAL(4,1) CHECK (blood_glucose_fasting >= 40 AND blood_glucose_fasting <= 500),
  blood_glucose_postprandial DECIMAL(4,1) CHECK (blood_glucose_postprandial >= 40 AND blood_glucose_postprandial <= 500),
  hba1c DECIMAL(3,1) CHECK (hba1c >= 3.0 AND hba1c <= 15.0),
  glucose_variability DECIMAL(4,2),
  cholesterol_total INTEGER CHECK (cholesterol_total >= 100 AND cholesterol_total <= 400),
  cholesterol_hdl INTEGER CHECK (cholesterol_hdl >= 20 AND cholesterol_hdl <= 100),
  cholesterol_ldl INTEGER CHECK (cholesterol_ldl >= 50 AND cholesterol_ldl <= 200),
  triglycerides INTEGER CHECK (triglycerides >= 50 AND triglycerides <= 500),
  cholesterol_ratio DECIMAL(3,2),
  insulin_fasting DECIMAL(4,1),
  insulin_sensitivity DECIMAL(4,2),
  insulin_resistance DECIMAL(4,2),
  ketones DECIMAL(3,1),
  recorded_at TIMESTAMP WITH TIME ZONE NOT NULL,
  device_source TEXT,
  accuracy_score DECIMAL(3,2) CHECK (accuracy_score >= 0 AND accuracy_score <= 1),
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Sleep Quality Table
CREATE TABLE sleep_metrics (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  sleep_date DATE NOT NULL,
  total_duration DECIMAL(4,2), -- hours
  deep_sleep_duration DECIMAL(4,2), -- hours
  rem_sleep_duration DECIMAL(4,2), -- hours
  light_sleep_duration DECIMAL(4,2), -- hours
  sleep_efficiency DECIMAL(4,2) CHECK (sleep_efficiency >= 0 AND sleep_efficiency <= 100),
  sleep_latency DECIMAL(4,2), -- minutes to fall asleep
  awakenings_count INTEGER,
  restlessness_score DECIMAL(3,2) CHECK (restlessness_score >= 0 AND restlessness_score <= 1),
  room_temperature DECIMAL(4,1),
  humidity DECIMAL(4,1),
  noise_level DECIMAL(4,1), -- decibels
  light_level DECIMAL(4,1), -- lux
  quality_score DECIMAL(3,2) CHECK (quality_score >= 0 AND quality_score <= 1),
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Fitness & Activity Table
CREATE TABLE fitness_metrics (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  activity_date DATE NOT NULL,
  steps_count INTEGER,
  distance_km DECIMAL(6,2),
  calories_burned INTEGER,
  active_minutes INTEGER,
  sedentary_minutes INTEGER,
  workouts_count INTEGER,
  workout_duration DECIMAL(4,2), -- hours
  workout_intensity TEXT CHECK (workout_intensity IN ('low', 'moderate', 'high')),
  workout_type TEXT CHECK (workout_type IN ('cardio', 'strength', 'flexibility', 'balance', 'mixed')),
  vo2_max DECIMAL(4,1), -- ml/kg/min
  strength_upper_body INTEGER, -- max weight in kg
  strength_lower_body INTEGER, -- max weight in kg
  strength_core INTEGER, -- max weight in kg
  flexibility_score DECIMAL(3,2) CHECK (flexibility_score >= 0 AND flexibility_score <= 1),
  balance_score DECIMAL(3,2) CHECK (balance_score >= 0 AND balance_score <= 1),
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Mental Health Table
CREATE TABLE mental_health_metrics (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  mood_score INTEGER CHECK (mood_score >= 1 AND mood_score <= 10),
  mood_stability DECIMAL(3,2) CHECK (mood_stability >= 0 AND mood_stability <= 1),
  mood_triggers TEXT[],
  stress_level INTEGER CHECK (stress_level >= 1 AND stress_level <= 10),
  cortisol_level DECIMAL(5,2), -- nmol/L
  perceived_stress_score INTEGER CHECK (perceived_stress_score >= 0 AND perceived_stress_score <= 40),
  memory_score DECIMAL(3,2) CHECK (memory_score >= 0 AND memory_score <= 1),
  attention_span INTEGER, -- minutes
  reaction_time DECIMAL(4,2), -- milliseconds
  processing_speed DECIMAL(3,2) CHECK (processing_speed >= 0 AND processing_speed <= 1),
  recorded_at TIMESTAMP WITH TIME ZONE NOT NULL,
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Hormonal Health Table
CREATE TABLE hormonal_metrics (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  tsh DECIMAL(4,2), -- mIU/L
  t3 DECIMAL(4,2), -- pg/mL
  t4 DECIMAL(4,2), -- ng/dL
  thyroid_antibodies DECIMAL(5,2), -- IU/mL
  testosterone DECIMAL(5,2), -- ng/dL
  estrogen DECIMAL(4,2), -- pg/mL
  progesterone DECIMAL(4,2), -- ng/mL
  shbg DECIMAL(4,2), -- nmol/L
  cortisol_morning DECIMAL(5,2), -- nmol/L
  cortisol_evening DECIMAL(5,2), -- nmol/L
  cortisol_diurnal_pattern DECIMAL(4,2),
  adrenaline DECIMAL(4,2), -- pg/mL
  noradrenaline DECIMAL(4,2), -- pg/mL
  recorded_at TIMESTAMP WITH TIME ZONE NOT NULL,
  device_source TEXT,
  accuracy_score DECIMAL(3,2) CHECK (accuracy_score >= 0 AND accuracy_score <= 1),
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX idx_cardiovascular_user_date ON cardiovascular_metrics(user_id, recorded_at);
CREATE INDEX idx_respiratory_user_date ON respiratory_metrics(user_id, recorded_at);
CREATE INDEX idx_metabolic_user_date ON metabolic_metrics(user_id, recorded_at);
CREATE INDEX idx_sleep_user_date ON sleep_metrics(user_id, sleep_date);
CREATE INDEX idx_fitness_user_date ON fitness_metrics(user_id, activity_date);
CREATE INDEX idx_mental_health_user_date ON mental_health_metrics(user_id, recorded_at);
CREATE INDEX idx_hormonal_user_date ON hormonal_metrics(user_id, recorded_at);

-- Enable RLS on all new tables
ALTER TABLE cardiovascular_metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE respiratory_metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE metabolic_metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE sleep_metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE fitness_metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE mental_health_metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE hormonal_metrics ENABLE ROW LEVEL SECURITY;

-- Create RLS policies for cardiovascular_metrics
CREATE POLICY "Users can view their own cardiovascular metrics"
  ON cardiovascular_metrics FOR SELECT
  USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own cardiovascular metrics"
  ON cardiovascular_metrics FOR INSERT
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own cardiovascular metrics"
  ON cardiovascular_metrics FOR UPDATE
  USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own cardiovascular metrics"
  ON cardiovascular_metrics FOR DELETE
  USING (auth.uid() = user_id);

-- Create RLS policies for respiratory_metrics
CREATE POLICY "Users can view their own respiratory metrics"
  ON respiratory_metrics FOR SELECT
  USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own respiratory metrics"
  ON respiratory_metrics FOR INSERT
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own respiratory metrics"
  ON respiratory_metrics FOR UPDATE
  USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own respiratory metrics"
  ON respiratory_metrics FOR DELETE
  USING (auth.uid() = user_id);

-- Create RLS policies for metabolic_metrics
CREATE POLICY "Users can view their own metabolic metrics"
  ON metabolic_metrics FOR SELECT
  USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own metabolic metrics"
  ON metabolic_metrics FOR INSERT
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own metabolic metrics"
  ON metabolic_metrics FOR UPDATE
  USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own metabolic metrics"
  ON metabolic_metrics FOR DELETE
  USING (auth.uid() = user_id);

-- Create RLS policies for sleep_metrics
CREATE POLICY "Users can view their own sleep metrics"
  ON sleep_metrics FOR SELECT
  USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own sleep metrics"
  ON sleep_metrics FOR INSERT
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own sleep metrics"
  ON sleep_metrics FOR UPDATE
  USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own sleep metrics"
  ON sleep_metrics FOR DELETE
  USING (auth.uid() = user_id);

-- Create RLS policies for fitness_metrics
CREATE POLICY "Users can view their own fitness metrics"
  ON fitness_metrics FOR SELECT
  USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own fitness metrics"
  ON fitness_metrics FOR INSERT
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own fitness metrics"
  ON fitness_metrics FOR UPDATE
  USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own fitness metrics"
  ON fitness_metrics FOR DELETE
  USING (auth.uid() = user_id);

-- Create RLS policies for mental_health_metrics
CREATE POLICY "Users can view their own mental health metrics"
  ON mental_health_metrics FOR SELECT
  USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own mental health metrics"
  ON mental_health_metrics FOR INSERT
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own mental health metrics"
  ON mental_health_metrics FOR UPDATE
  USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own mental health metrics"
  ON mental_health_metrics FOR DELETE
  USING (auth.uid() = user_id);

-- Create RLS policies for hormonal_metrics
CREATE POLICY "Users can view their own hormonal metrics"
  ON hormonal_metrics FOR SELECT
  USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own hormonal metrics"
  ON hormonal_metrics FOR INSERT
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own hormonal metrics"
  ON hormonal_metrics FOR UPDATE
  USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own hormonal metrics"
  ON hormonal_metrics FOR DELETE
  USING (auth.uid() = user_id);

-- Create analytics functions for the new metrics

-- Function to get cardiovascular health trends
CREATE OR REPLACE FUNCTION get_cardiovascular_trends(
  p_user_id UUID,
  p_days INTEGER DEFAULT 30
)
RETURNS TABLE (
  recorded_at TIMESTAMP WITH TIME ZONE,
  heart_rate_resting INTEGER,
  heart_rate_variability DECIMAL(5,2),
  blood_pressure_systolic INTEGER,
  blood_pressure_diastolic INTEGER,
  trend_direction TEXT
) AS $$
BEGIN
  RETURN QUERY
  SELECT
    cm.recorded_at,
    cm.heart_rate_resting,
    cm.heart_rate_variability,
    cm.blood_pressure_systolic,
    cm.blood_pressure_diastolic,
    CASE
      WHEN LAG(cm.heart_rate_resting) OVER (ORDER BY cm.recorded_at) < cm.heart_rate_resting THEN 'increasing'
      WHEN LAG(cm.heart_rate_resting) OVER (ORDER BY cm.recorded_at) > cm.heart_rate_resting THEN 'decreasing'
      ELSE 'stable'
    END as trend_direction
  FROM cardiovascular_metrics cm
  WHERE cm.user_id = p_user_id
    AND cm.recorded_at >= NOW() - INTERVAL '1 day' * p_days
  ORDER BY cm.recorded_at;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to get sleep quality summary
CREATE OR REPLACE FUNCTION get_sleep_quality_summary(
  p_user_id UUID,
  p_days INTEGER DEFAULT 30
)
RETURNS TABLE (
  avg_duration DECIMAL(4,2),
  avg_efficiency DECIMAL(4,2),
  avg_latency DECIMAL(4,2),
  quality_trend TEXT,
  deep_sleep_percentage DECIMAL(4,2),
  rem_sleep_percentage DECIMAL(4,2)
) AS $$
BEGIN
  RETURN QUERY
  SELECT
    AVG(sm.total_duration) as avg_duration,
    AVG(sm.sleep_efficiency) as avg_efficiency,
    AVG(sm.sleep_latency) as avg_latency,
    CASE
      WHEN AVG(sm.sleep_efficiency) > 85 THEN 'excellent'
      WHEN AVG(sm.sleep_efficiency) > 75 THEN 'good'
      WHEN AVG(sm.sleep_efficiency) > 65 THEN 'fair'
      ELSE 'poor'
    END as quality_trend,
    AVG(sm.deep_sleep_duration / sm.total_duration * 100) as deep_sleep_percentage,
    AVG(sm.rem_sleep_duration / sm.total_duration * 100) as rem_sleep_percentage
  FROM sleep_metrics sm
  WHERE sm.user_id = p_user_id
    AND sm.sleep_date >= CURRENT_DATE - INTERVAL '1 day' * p_days;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to get metabolic health summary
CREATE OR REPLACE FUNCTION get_metabolic_health_summary(
  p_user_id UUID,
  p_days INTEGER DEFAULT 30
)
RETURNS TABLE (
  avg_glucose_fasting DECIMAL(4,1),
  avg_hba1c DECIMAL(3,1),
  cholesterol_ratio DECIMAL(3,2),
  metabolic_risk TEXT,
  insulin_sensitivity_status TEXT
) AS $$
BEGIN
  RETURN QUERY
  SELECT
    AVG(mm.blood_glucose_fasting) as avg_glucose_fasting,
    AVG(mm.hba1c) as avg_hba1c,
    AVG(mm.cholesterol_ratio) as cholesterol_ratio,
    CASE
      WHEN AVG(mm.blood_glucose_fasting) > 126 OR AVG(mm.hba1c) > 6.5 THEN 'high'
      WHEN AVG(mm.blood_glucose_fasting) > 100 OR AVG(mm.hba1c) > 5.7 THEN 'moderate'
      ELSE 'low'
    END as metabolic_risk,
    CASE
      WHEN AVG(mm.insulin_sensitivity) > 0.8 THEN 'excellent'
      WHEN AVG(mm.insulin_sensitivity) > 0.6 THEN 'good'
      WHEN AVG(mm.insulin_sensitivity) > 0.4 THEN 'fair'
      ELSE 'poor'
    END as insulin_sensitivity_status
  FROM metabolic_metrics mm
  WHERE mm.user_id = p_user_id
    AND mm.recorded_at >= NOW() - INTERVAL '1 day' * p_days;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;
