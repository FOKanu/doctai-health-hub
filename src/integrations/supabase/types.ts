import { Database as DatabaseGenerated } from './types.generated';

export type Database = DatabaseGenerated;

export type Tables<T extends keyof Database['public']['Tables']> = Database['public']['Tables'][T]['Row'];
export type Enums<T extends keyof Database['public']['Enums']> = Database['public']['Enums'][T];

// Time-series specific types
export interface PatientTimeline {
  id: string;
  user_id: string;
  condition_type: 'skin_lesion' | 'cardiovascular' | 'respiratory' | 'neurological' | 'metabolic' | 'musculoskeletal' | 'gastrointestinal' | 'endocrine';
  baseline_date: string;
  status: 'monitoring' | 'improving' | 'worsening' | 'stable';
  severity_score?: number;
  confidence_score?: number;
  notes?: string;
  metadata?: Record<string, unknown>;
  created_at: string;
  updated_at: string;
}

export interface HealthMetricTimeseries {
  id: string;
  user_id: string;
  metric_type: 'heart_rate' | 'blood_pressure' | 'weight' | 'temperature' | 'sleep_hours' | 'steps' | 'calories' | 'water_intake' | 'blood_glucose' | 'oxygen_saturation';
  value: unknown;
  recorded_at: string;
  device_source?: string;
  accuracy_score?: number;
  metadata?: Record<string, unknown>;
  created_at: string;
}

export interface ScanSequence {
  id: string;
  user_id: string;
  sequence_name: string;
  image_ids: string[];
  analysis_type: 'progression' | 'treatment_response' | 'baseline' | 'follow_up';
  baseline_image_id?: string;
  progression_score?: number;
  confidence_score?: number;
  findings?: Record<string, unknown>;
  recommendations?: string[];
  metadata?: Record<string, unknown>;
  created_at: string;
  updated_at: string;
}

export interface TreatmentResponse {
  id: string;
  user_id: string;
  timeline_id: string;
  treatment_name: string;
  start_date: string;
  end_date?: string;
  effectiveness_score?: number;
  side_effects?: string[];
  adherence_percentage?: number;
  notes?: string;
  metadata?: Record<string, unknown>;
  created_at: string;
  updated_at: string;
}

export interface RiskProgression {
  id: string;
  user_id: string;
  condition_type: 'skin_lesion' | 'cardiovascular' | 'respiratory' | 'neurological' | 'metabolic' | 'musculoskeletal' | 'gastrointestinal' | 'endocrine';
  risk_level: 'low' | 'medium' | 'high';
  probability: number;
  factors?: Record<string, unknown>;
  recorded_at: string;
  predicted_date?: string;
  confidence_score?: number;
  metadata?: Record<string, unknown>;
  created_at: string;
}

// Health metric value types
export interface HeartRateValue {
  value: number;
  type: 'resting' | 'active' | 'max';
  unit: 'bpm';
}

export interface BloodPressureValue {
  systolic: number;
  diastolic: number;
  unit: 'mmHg';
}

export interface WeightValue {
  value: number;
  unit: 'kg' | 'lbs';
}

export interface TemperatureValue {
  value: number;
  unit: 'celsius' | 'fahrenheit';
}

export interface SleepValue {
  hours: number;
  quality?: 'poor' | 'fair' | 'good' | 'excellent';
  efficiency?: number;
}

export interface StepsValue {
  count: number;
  distance?: number;
  calories?: number;
}

export interface WaterIntakeValue {
  amount: number;
  unit: 'ml' | 'oz' | 'glasses';
}

export interface BloodGlucoseValue {
  value: number;
  unit: 'mg/dL' | 'mmol/L';
  context?: 'fasting' | 'post_meal' | 'random';
}

export interface OxygenSaturationValue {
  value: number;
  unit: '%';
}

// Time-series analytics types
export interface HealthMetricsTrend {
  recorded_at: string;
  value: unknown;
  trend_direction: 'increasing' | 'decreasing' | 'stable';
}

export interface PatientProgressionSummary {
  condition_type: string;
  status: string;
  severity_score?: number;
  days_since_baseline: number;
  trend: 'positive' | 'negative' | 'neutral';
}

export const Constants = {
  public: {
    Enums: {
      image_type: ["skin_lesion", "ct_scan", "mri", "xray", "eeg"],
      condition_type: ["skin_lesion", "cardiovascular", "respiratory", "neurological", "metabolic", "musculoskeletal", "gastrointestinal", "endocrine"],
      patient_status: ["monitoring", "improving", "worsening", "stable"],
      metric_type: ["heart_rate", "blood_pressure", "weight", "temperature", "sleep_hours", "steps", "calories", "water_intake", "blood_glucose", "oxygen_saturation"],
      analysis_type: ["progression", "treatment_response", "baseline", "follow_up"],
    },
  },
} as const;

// Extended Health Metrics Types
export interface CardiovascularMetrics {
  id: string;
  user_id: string;
  heart_rate_resting?: number;
  heart_rate_active?: number;
  heart_rate_variability?: number;
  blood_pressure_systolic?: number;
  blood_pressure_diastolic?: number;
  pulse_pressure?: number;
  mean_arterial_pressure?: number;
  ecg_rhythm?: 'normal' | 'irregular' | 'atrial_fibrillation' | 'bradycardia' | 'tachycardia';
  qt_interval?: number;
  st_segment?: number;
  recorded_at: string;
  device_source?: string;
  accuracy_score?: number;
  metadata?: Record<string, unknown>;
  created_at: string;
}

export interface RespiratoryMetrics {
  id: string;
  user_id: string;
  respiratory_rate?: number;
  oxygen_saturation?: number;
  peak_flow?: number;
  forced_expiratory_volume?: number;
  lung_capacity?: number;
  breathing_pattern?: 'normal' | 'shallow' | 'rapid' | 'irregular' | 'labored';
  recorded_at: string;
  device_source?: string;
  accuracy_score?: number;
  metadata?: Record<string, unknown>;
  created_at: string;
}

export interface MetabolicMetrics {
  id: string;
  user_id: string;
  blood_glucose_fasting?: number;
  blood_glucose_postprandial?: number;
  hba1c?: number;
  glucose_variability?: number;
  cholesterol_total?: number;
  cholesterol_hdl?: number;
  cholesterol_ldl?: number;
  triglycerides?: number;
  cholesterol_ratio?: number;
  insulin_fasting?: number;
  insulin_sensitivity?: number;
  insulin_resistance?: number;
  ketones?: number;
  recorded_at: string;
  device_source?: string;
  accuracy_score?: number;
  metadata?: Record<string, unknown>;
  created_at: string;
}

export interface SleepMetrics {
  id: string;
  user_id: string;
  sleep_date: string;
  total_duration?: number;
  deep_sleep_duration?: number;
  rem_sleep_duration?: number;
  light_sleep_duration?: number;
  sleep_efficiency?: number;
  sleep_latency?: number;
  awakenings_count?: number;
  restlessness_score?: number;
  room_temperature?: number;
  humidity?: number;
  noise_level?: number;
  light_level?: number;
  quality_score?: number;
  metadata?: Record<string, unknown>;
  created_at: string;
}

export interface FitnessMetrics {
  id: string;
  user_id: string;
  activity_date: string;
  steps_count?: number;
  distance_km?: number;
  calories_burned?: number;
  active_minutes?: number;
  sedentary_minutes?: number;
  workouts_count?: number;
  workout_duration?: number;
  workout_intensity?: 'low' | 'moderate' | 'high';
  workout_type?: 'cardio' | 'strength' | 'flexibility' | 'balance' | 'mixed';
  vo2_max?: number;
  strength_upper_body?: number;
  strength_lower_body?: number;
  strength_core?: number;
  flexibility_score?: number;
  balance_score?: number;
  metadata?: Record<string, unknown>;
  created_at: string;
}

export interface MentalHealthMetrics {
  id: string;
  user_id: string;
  mood_score?: number;
  mood_stability?: number;
  mood_triggers?: string[];
  stress_level?: number;
  cortisol_level?: number;
  perceived_stress_score?: number;
  memory_score?: number;
  attention_span?: number;
  reaction_time?: number;
  processing_speed?: number;
  recorded_at: string;
  metadata?: Record<string, unknown>;
  created_at: string;
}

export interface HormonalMetrics {
  id: string;
  user_id: string;
  tsh?: number;
  t3?: number;
  t4?: number;
  thyroid_antibodies?: number;
  testosterone?: number;
  estrogen?: number;
  progesterone?: number;
  shbg?: number;
  cortisol_morning?: number;
  cortisol_evening?: number;
  cortisol_diurnal_pattern?: number;
  adrenaline?: number;
  noradrenaline?: number;
  recorded_at: string;
  device_source?: string;
  accuracy_score?: number;
  metadata?: Record<string, unknown>;
  created_at: string;
}

// Analytics result types
export interface CardiovascularTrends {
  recorded_at: string;
  heart_rate_resting: number;
  heart_rate_variability: number;
  blood_pressure_systolic: number;
  blood_pressure_diastolic: number;
  trend_direction: 'increasing' | 'decreasing' | 'stable';
}

export interface SleepQualitySummary {
  avg_duration: number;
  avg_efficiency: number;
  avg_latency: number;
  quality_trend: 'excellent' | 'good' | 'fair' | 'poor';
  deep_sleep_percentage: number;
  rem_sleep_percentage: number;
}

export interface MetabolicHealthSummary {
  avg_glucose_fasting: number;
  avg_hba1c: number;
  cholesterol_ratio: number;
  metabolic_risk: 'low' | 'moderate' | 'high';
  insulin_sensitivity_status: 'excellent' | 'good' | 'fair' | 'poor';
}

// Telemedicine Integration Types
export interface HealthcareProvider {
  id: string;
  user_id: string;
  provider_name: string;
  specialty: 'primary_care' | 'cardiology' | 'dermatology' | 'endocrinology' | 'neurology' | 'psychiatry' | 'orthopedics' | 'pediatrics' | 'gynecology' | 'oncology' | 'pulmonology' | 'gastroenterology' | 'ophthalmology' | 'urology' | 'general';
  license_number?: string;
  credentials?: string[];
  experience_years?: number;
  languages?: string[];
  availability_schedule?: Record<string, unknown>;
  consultation_fee?: number;
  rating?: number;
  total_consultations: number;
  is_verified: boolean;
  is_available: boolean;
  profile_image_url?: string;
  bio?: string;
  contact_info?: Record<string, unknown>;
  created_at: string;
  updated_at: string;
}

export interface TelemedicineConsultation {
  id: string;
  patient_id: string;
  provider_id: string;
  consultation_type: 'video' | 'audio' | 'chat' | 'follow_up' | 'emergency';
  status: 'scheduled' | 'in_progress' | 'completed' | 'cancelled' | 'no_show';
  scheduled_at: string;
  started_at?: string;
  ended_at?: string;
  duration_minutes?: number;
  meeting_url?: string;
  meeting_id?: string;
  consultation_notes?: string;
  diagnosis?: string;
  prescriptions?: Record<string, unknown>[];
  recommendations?: string[];
  follow_up_date?: string;
  follow_up_required: boolean;
  emergency_contact?: string;
  symptoms?: string[];
  vital_signs?: Record<string, unknown>;
  attachments?: Record<string, unknown>[];
  metadata?: Record<string, unknown>;
  created_at: string;
  updated_at: string;
}

export interface RemoteMonitoring {
  id: string;
  patient_id: string;
  provider_id?: string;
  monitoring_type: string;
  device_id?: string;
  device_type?: string;
  is_active: boolean;
  start_date: string;
  end_date?: string;
  monitoring_frequency?: string;
  alert_thresholds?: Record<string, unknown>;
  last_reading_at?: string;
  last_reading_value?: unknown;
  alert_level?: 'low' | 'medium' | 'high' | 'critical';
  is_alert_active: boolean;
  alert_message?: string;
  metadata?: Record<string, unknown>;
  created_at: string;
  updated_at: string;
}

export interface HealthAlert {
  id: string;
  patient_id: string;
  provider_id?: string;
  alert_type: string;
  alert_level: 'low' | 'medium' | 'high' | 'critical';
  title: string;
  message: string;
  metric_name?: string;
  metric_value?: unknown;
  threshold_value?: unknown;
  is_read: boolean;
  is_acknowledged: boolean;
  acknowledged_at?: string;
  acknowledged_by?: string;
  action_taken?: string;
  follow_up_required: boolean;
  created_at: string;
  updated_at: string;
}

export interface Appointment {
  id: string;
  patient_id: string;
  provider_id: string;
  appointment_type: 'consultation' | 'follow_up' | 'emergency' | 'routine_checkup' | 'specialist_referral';
  scheduled_at: string;
  duration_minutes: number;
  status: 'scheduled' | 'in_progress' | 'completed' | 'cancelled' | 'no_show';
  reason?: string;
  symptoms?: string[];
  is_urgent: boolean;
  reminder_sent: boolean;
  reminder_sent_at?: string;
  notes?: string;
  metadata?: Record<string, unknown>;
  created_at: string;
  updated_at: string;
}

export interface ProviderAvailability {
  id: string;
  provider_id: string;
  day_of_week: number;
  start_time: string;
  end_time: string;
  is_available: boolean;
  consultation_type?: 'video' | 'audio' | 'chat' | 'follow_up' | 'emergency';
  max_patients_per_slot: number;
  slot_duration_minutes: number;
  created_at: string;
  updated_at: string;
}

export interface PatientProviderRelationship {
  id: string;
  patient_id: string;
  provider_id: string;
  relationship_type: string;
  start_date: string;
  end_date?: string;
  is_active: boolean;
  notes?: string;
  created_at: string;
  updated_at: string;
}

export interface ConsultationRecording {
  id: string;
  consultation_id: string;
  recording_url: string;
  recording_type: string;
  file_size_bytes?: number;
  duration_seconds?: number;
  is_encrypted: boolean;
  access_level: string;
  expires_at?: string;
  created_at: string;
}

// Telemedicine analytics types
export interface AvailableAppointmentSlot {
  start_time: string;
  end_time: string;
  is_available: boolean;
}

export interface PatientHealthSummary {
  recent_consultations: number;
  active_monitoring_count: number;
  unread_alerts: number;
  last_consultation_date?: string;
  health_score?: number;
  risk_level: string;
}

export interface ConsultationStats {
  total_consultations: number;
  completed_consultations: number;
  cancelled_consultations: number;
  average_duration_minutes?: number;
  total_revenue?: number;
}
