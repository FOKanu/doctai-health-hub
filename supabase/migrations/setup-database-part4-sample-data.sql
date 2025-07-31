-- DoctAI Health Hub Database Setup - Part 4: Sample Data
-- Run this script after Part 3 in your Supabase SQL Editor
-- This adds sample data to test your application

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

-- Success message
DO $$
BEGIN
  RAISE NOTICE '✅ Part 4 Complete: Sample data inserted successfully!';
  RAISE NOTICE '🎉 Your application should now work without 404 errors!';
  RAISE NOTICE '📊 Test the analytics and health tracking features.';
END $$;
