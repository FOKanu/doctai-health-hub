#!/bin/bash

# DoctAI Health Hub - Database Setup Script
# This script sets up all required Supabase tables and populates them with sample data

set -e

echo "🏥 Setting up DoctAI Health Hub Database..."
echo "=============================================="

# Check if Supabase CLI is installed
if ! command -v supabase &> /dev/null; then
    echo "❌ Supabase CLI is not installed. Please install it first:"
    echo "   npm install -g supabase"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "supabase/config.toml" ]; then
    echo "❌ Not in the DoctAI project directory. Please run this from the project root."
    exit 1
fi

echo "📋 Running database migrations..."

# Run all migrations
supabase db reset --linked

echo "✅ Database migrations completed successfully!"

echo "📊 Creating sample data for all user roles..."

# Create a SQL script to populate the database with sample data
cat > supabase/seed.sql << 'EOF'
-- Sample data for DoctAI Health Hub
-- This script populates the database with sample data for testing

-- Insert sample users (these should match your auth.users table)
-- Note: Replace these UUIDs with actual user IDs from your auth.users table

-- Sample Patient User
INSERT INTO auth.users (id, email, encrypted_password, email_confirmed_at, created_at, updated_at)
VALUES (
  '550e8400-e29b-41d4-a716-446655440001',
  'patient@doctai.com',
  crypt('password123', gen_salt('bf')),
  NOW(),
  NOW(),
  NOW()
) ON CONFLICT (id) DO NOTHING;

-- Sample Provider User
INSERT INTO auth.users (id, email, encrypted_password, email_confirmed_at, created_at, updated_at)
VALUES (
  '550e8400-e29b-41d4-a716-446655440002',
  'provider@doctai.com',
  crypt('password123', gen_salt('bf')),
  NOW(),
  NOW(),
  NOW()
) ON CONFLICT (id) DO NOTHING;

-- Sample Engineer User
INSERT INTO auth.users (id, email, encrypted_password, email_confirmed_at, created_at, updated_at)
VALUES (
  '550e8400-e29b-41d4-a716-446655440003',
  'engineer@doctai.com',
  crypt('password123', gen_salt('bf')),
  NOW(),
  NOW(),
  NOW()
) ON CONFLICT (id) DO NOTHING;

-- Sample Mock User (for development)
INSERT INTO auth.users (id, email, encrypted_password, email_confirmed_at, created_at, updated_at)
VALUES (
  '550e8400-e29b-41d4-a716-446655440004',
  'mock_user@doctai.com',
  crypt('password123', gen_salt('bf')),
  NOW(),
  NOW(),
  NOW()
) ON CONFLICT (id) DO NOTHING;

-- Insert sample health metrics for the mock user
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
  ('550e8400-e29b-41d4-a716-446655440004', 'water_intake', '{"amount": 2000}', NOW() - INTERVAL '1 day', 'water_tracker', 0.85);

-- Insert sample risk progression data
INSERT INTO risk_progressions (user_id, condition_type, risk_level, probability, factors, recorded_at, confidence_score)
VALUES
  ('550e8400-e29b-41d4-a716-446655440004', 'cardiovascular', 'low', 0.15, '{"age": 35, "family_history": false, "lifestyle": "active"}', NOW() - INTERVAL '1 day', 0.85),
  ('550e8400-e29b-41d4-a716-446655440004', 'cardiovascular', 'low', 0.12, '{"age": 35, "family_history": false, "lifestyle": "active"}', NOW() - INTERVAL '2 days', 0.88),
  ('550e8400-e29b-41d4-a716-446655440004', 'metabolic', 'medium', 0.45, '{"bmi": 26.5, "blood_glucose": 95, "family_history": true}', NOW() - INTERVAL '1 day', 0.75),
  ('550e8400-e29b-41d4-a716-446655440004', 'respiratory', 'low', 0.08, '{"age": 35, "no_smoking_history": true, "environment": "clean"}', NOW() - INTERVAL '1 day', 0.92);

-- Insert sample patient timeline
INSERT INTO patient_timelines (user_id, condition_type, baseline_date, status, severity_score, confidence_score, notes)
VALUES
  ('550e8400-e29b-41d4-a716-446655440004', 'cardiovascular', NOW() - INTERVAL '30 days', 'stable', 0.15, 0.85, 'Regular monitoring of cardiovascular health'),
  ('550e8400-e29b-41d4-a716-446655440004', 'metabolic', NOW() - INTERVAL '45 days', 'monitoring', 0.45, 0.75, 'Monitoring blood glucose levels and lifestyle changes');

-- Insert sample healthcare provider
INSERT INTO healthcare_providers (user_id, provider_name, specialty, license_number, credentials, experience_years, languages, consultation_fee, rating, is_verified, is_available, bio)
VALUES
  ('550e8400-e29b-41d4-a716-446655440002', 'Dr. Sarah Johnson', 'primary_care', 'MD123456', ARRAY['MD', 'Board Certified'], 8, ARRAY['English', 'Spanish'], 150.00, 4.8, true, true, 'Experienced primary care physician with focus on preventive medicine');

-- Insert sample appointments
INSERT INTO appointments (patient_id, provider_id, appointment_type, scheduled_at, status, notes)
VALUES
  ('550e8400-e29b-41d4-a716-446655440004', '550e8400-e29b-41d4-a716-446655440002', 'routine_checkup', NOW() + INTERVAL '7 days', 'scheduled', 'Annual health checkup'),
  ('550e8400-e29b-41d4-a716-446655440004', '550e8400-e29b-41d4-a716-446655440002', 'follow_up', NOW() + INTERVAL '14 days', 'scheduled', 'Follow-up on blood pressure monitoring');

-- Insert sample telemedicine consultation
INSERT INTO telemedicine_consultations (patient_id, provider_id, consultation_type, status, scheduled_at, consultation_notes, diagnosis, prescriptions, recommendations)
VALUES
  ('550e8400-e29b-41d4-a716-446655440004', '550e8400-e29b-41d4-a716-446655440002', 'video', 'completed', NOW() - INTERVAL '3 days', 'Patient reported mild fatigue and stress', 'Stress-related symptoms', '[]', ARRAY['Increase sleep to 8 hours', 'Practice stress management techniques', 'Follow up in 2 weeks']);

-- Insert sample cardiovascular metrics
INSERT INTO cardiovascular_metrics (user_id, heart_rate_resting, heart_rate_active, heart_rate_variability, blood_pressure_systolic, blood_pressure_diastolic, pulse_pressure, mean_arterial_pressure, ecg_rhythm, recorded_at, device_source, accuracy_score)
VALUES
  ('550e8400-e29b-41d4-a716-446655440004', 72, 140, 45.2, 120, 80, 40, 93.33, 'normal', NOW() - INTERVAL '1 day', 'smartwatch', 0.95),
  ('550e8400-e29b-41d4-a716-446655440004', 70, 135, 48.1, 118, 78, 40, 91.33, 'normal', NOW() - INTERVAL '2 days', 'smartwatch', 0.95);

-- Insert sample sleep metrics
INSERT INTO sleep_metrics (user_id, sleep_date, total_duration, deep_sleep_duration, rem_sleep_duration, sleep_efficiency, sleep_latency, wake_count, sleep_quality_score)
VALUES
  ('550e8400-e29b-41d4-a716-446655440004', CURRENT_DATE - INTERVAL '1 day', 7.5, 2.1, 1.8, 85.5, 15.2, 2, 8.2),
  ('550e8400-e29b-41d4-a716-446655440004', CURRENT_DATE - INTERVAL '2 days', 8.0, 2.3, 2.0, 88.0, 12.5, 1, 8.5);

-- Insert sample fitness metrics
INSERT INTO fitness_metrics (user_id, activity_date, steps_count, calories_burned, active_minutes, distance_km, vo2_max, resting_heart_rate, max_heart_rate, avg_heart_rate)
VALUES
  ('550e8400-e29b-41d4-a716-446655440004', CURRENT_DATE - INTERVAL '1 day', 8500, 320, 45, 6.2, 42.5, 70, 140, 95),
  ('550e8400-e29b-41d4-a716-446655440004', CURRENT_DATE - INTERVAL '2 days', 9200, 350, 52, 6.8, 43.1, 68, 142, 98);

-- Insert sample mental health metrics
INSERT INTO mental_health_metrics (user_id, mood_score, stress_level, anxiety_level, depression_score, sleep_quality, social_connections, recorded_at, device_source, accuracy_score)
VALUES
  ('550e8400-e29b-41d4-a716-446655440004', 7.5, 4.2, 3.8, 2.1, 8.2, 7.8, NOW() - INTERVAL '1 day', 'mobile_app', 0.80),
  ('550e8400-e29b-41d4-a716-446655440004', 8.0, 3.5, 3.2, 1.8, 8.5, 8.0, NOW() - INTERVAL '2 days', 'mobile_app', 0.80);

-- Insert sample image metadata
INSERT INTO image_metadata (user_id, url, type, analysis_result, metadata)
VALUES
  ('550e8400-e29b-41d4-a716-446655440004', 'https://example.com/skin_lesion_1.jpg', 'skin_lesion', '{"diagnosis": "benign_mole", "confidence": 0.92, "recommendations": ["monitor_for_changes", "sun_protection"]}', '{"location": "left_arm", "size": "5mm"}'),
  ('550e8400-e29b-41d4-a716-446655440004', 'https://example.com/xray_chest_1.jpg', 'xray', '{"diagnosis": "normal_chest_xray", "confidence": 0.95, "findings": ["clear_lung_fields", "normal_cardiac_silhouette"]}', '{"view": "PA", "technique": "standard"}');

-- Insert sample scan sequences
INSERT INTO scan_sequences (user_id, sequence_name, image_ids, analysis_type, progression_score, confidence_score, findings, recommendations)
VALUES
  ('550e8400-e29b-41d4-a716-446655440004', 'Chest X-Ray Series', ARRAY['550e8400-e29b-41d4-a716-446655440004'], 'baseline', 0.10, 0.95, '{"lung_fields": "clear", "cardiac_silhouette": "normal", "bony_structures": "intact"}', ARRAY['Continue annual screening', 'Maintain healthy lifestyle']);

-- Insert sample treatment responses
INSERT INTO treatment_responses (user_id, timeline_id, treatment_name, start_date, effectiveness_score, adherence_percentage, notes)
SELECT
  '550e8400-e29b-41d4-a716-446655440004',
  pt.id,
  'Lifestyle Modification',
  NOW() - INTERVAL '30 days',
  0.75,
  85.0,
  'Diet and exercise program for cardiovascular health'
FROM patient_timelines pt
WHERE pt.user_id = '550e8400-e29b-41d4-a716-446655440004'
AND pt.condition_type = 'cardiovascular'
LIMIT 1;

-- Insert sample health alerts
INSERT INTO health_alerts (patient_id, alert_type, alert_level, title, message, metric_name, metric_value, threshold_value, is_read, created_at)
VALUES
  ('550e8400-e29b-41d4-a716-446655440004', 'blood_pressure', 'medium', 'Blood Pressure Elevated', 'Your blood pressure reading is slightly elevated. Consider reducing salt intake and increasing physical activity.', 'blood_pressure', '{"systolic": 135, "diastolic": 85}', '{"systolic": 130, "diastolic": 80}', false, NOW() - INTERVAL '2 hours'),
  ('550e8400-e29b-41d4-a716-446655440004', 'sleep_quality', 'low', 'Sleep Quality Improvement', 'Great job! Your sleep quality has improved by 15% this week.', 'sleep_quality', '{"score": 8.5}', '{"score": 7.0}', false, NOW() - INTERVAL '1 day');

-- Insert sample provider availability
INSERT INTO provider_availability (provider_id, day_of_week, start_time, end_time, is_available)
VALUES
  ('550e8400-e29b-41d4-a716-446655440002', 'monday', '09:00:00', '17:00:00', true),
  ('550e8400-e29b-41d4-a716-446655440002', 'tuesday', '09:00:00', '17:00:00', true),
  ('550e8400-e29b-41d4-a716-446655440002', 'wednesday', '09:00:00', '17:00:00', true),
  ('550e8400-e29b-41d4-a716-446655440002', 'thursday', '09:00:00', '17:00:00', true),
  ('550e8400-e29b-41d4-a716-446655440002', 'friday', '09:00:00', '17:00:00', true);

-- Insert sample patient-provider relationships
INSERT INTO patient_provider_relationships (patient_id, provider_id, relationship_type, start_date, is_active, notes)
VALUES
  ('550e8400-e29b-41d4-a716-446655440004', '550e8400-e29b-41d4-a716-446655440002', 'primary_care', NOW() - INTERVAL '6 months', true, 'Primary care physician relationship established');

-- Insert sample remote monitoring
INSERT INTO remote_monitoring (patient_id, provider_id, monitoring_type, device_id, device_type, is_active, start_date, monitoring_frequency, alert_thresholds, last_reading_at, last_reading_value)
VALUES
  ('550e8400-e29b-41d4-a716-446655440004', '550e8400-e29b-41d4-a716-446655440002', 'blood_pressure', 'bp_monitor_001', 'blood_pressure_monitor', true, NOW() - INTERVAL '30 days', 'daily', '{"systolic_max": 140, "diastolic_max": 90}', NOW() - INTERVAL '1 day', '{"systolic": 120, "diastolic": 80}'),
  ('550e8400-e29b-41d4-a716-446655440004', '550e8400-e29b-41d4-a716-446655440002', 'heart_rate', 'hr_monitor_001', 'smartwatch', true, NOW() - INTERVAL '30 days', 'continuous', '{"max_heart_rate": 100}', NOW() - INTERVAL '1 hour', '{"heart_rate": 72}');

EOF

echo "📝 Running seed script..."
supabase db reset --linked

echo "✅ Database setup completed successfully!"
echo ""
echo "🎉 Your DoctAI Health Hub database is now ready!"
echo ""
echo "📊 Sample data has been created for:"
echo "   • Patient user (mock_user@doctai.com)"
echo "   • Provider user (provider@doctai.com)"
echo "   • Engineer user (engineer@doctai.com)"
echo ""
echo "🔑 Test credentials:"
echo "   • Email: mock_user@doctai.com"
echo "   • Password: password123"
echo ""
echo "📋 Available tables:"
echo "   • health_metrics_timeseries"
echo "   • risk_progressions"
echo "   • patient_timelines"
echo "   • healthcare_providers"
echo "   • telemedicine_consultations"
echo "   • appointments"
echo "   • cardiovascular_metrics"
echo "   • sleep_metrics"
echo "   • fitness_metrics"
echo "   • mental_health_metrics"
echo "   • image_metadata"
echo "   • scan_sequences"
echo "   • treatment_responses"
echo "   • health_alerts"
echo "   • provider_availability"
echo "   • patient_provider_relationships"
echo "   • remote_monitoring"
echo ""
echo "🚀 You can now run the application without console errors!"
