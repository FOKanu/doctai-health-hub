# Supabase Table Setup Guide

## Step 1: Create the Critical Tables (Fix 404 Errors)

### Table 1: health_metrics_timeseries

**Column Setup:**
- `id` (uuid, primary key, default: gen_random_uuid())
- `user_id` (uuid, foreign key to auth.users(id))
- `metric_type` (text, not null)
- `value` (jsonb, not null)
- `recorded_at` (timestamptz, not null)
- `device_source` (text)
- `accuracy_score` (decimal(3,2), check: >= 0 AND <= 1)
- `metadata` (jsonb, default: '{}')
- `created_at` (timestamptz, default: now())

**Upload CSV:** `health_metrics_timeseries.csv`

### Table 2: risk_progressions

**Column Setup:**
- `id` (uuid, primary key, default: gen_random_uuid())
- `user_id` (uuid, foreign key to auth.users(id))
- `condition_type` (text, not null)
- `risk_level` (text, not null, check: IN ('low', 'medium', 'high'))
- `probability` (decimal(3,2), check: >= 0 AND <= 1)
- `factors` (jsonb, default: '{}')
- `recorded_at` (timestamptz, not null)
- `predicted_date` (timestamptz)
- `confidence_score` (decimal(3,2), check: >= 0 AND <= 1)
- `metadata` (jsonb, default: '{}')
- `created_at` (timestamptz, default: now())

**Upload CSV:** `risk_progressions.csv`

### Table 3: patient_timelines

**Column Setup:**
- `id` (uuid, primary key, default: gen_random_uuid())
- `user_id` (uuid, foreign key to auth.users(id))
- `condition_type` (text, not null)
- `baseline_date` (timestamptz, not null)
- `status` (text, not null, default: 'monitoring')
- `severity_score` (decimal(3,2), check: >= 0 AND <= 1)
- `confidence_score` (decimal(3,2), check: >= 0 AND <= 1)
- `notes` (text)
- `metadata` (jsonb, default: '{}')
- `created_at` (timestamptz, default: now())
- `updated_at` (timestamptz, default: now())

**Upload CSV:** `patient_timelines.csv`

### Table 4: image_metadata

**Column Setup:**
- `id` (uuid, primary key, default: gen_random_uuid())
- `user_id` (uuid, foreign key to auth.users(id))
- `url` (text, not null)
- `type` (text, not null)
- `analysis_result` (jsonb, default: '{}')
- `metadata` (jsonb, default: '{}')
- `created_at` (timestamptz, default: now())
- `updated_at` (timestamptz, default: now())

**Upload CSV:** `image_metadata.csv`

### Table 5: scan_sequences

**Column Setup:**
- `id` (uuid, primary key, default: gen_random_uuid())
- `user_id` (uuid, foreign key to auth.users(id))
- `sequence_name` (text, not null)
- `image_ids` (uuid[], not null)
- `analysis_type` (text, not null)
- `baseline_image_id` (uuid, foreign key to image_metadata(id))
- `progression_score` (decimal(3,2), check: >= 0 AND <= 1)
- `confidence_score` (decimal(3,2), check: >= 0 AND <= 1)
- `findings` (jsonb, default: '{}')
- `recommendations` (text[])
- `metadata` (jsonb, default: '{}')
- `created_at` (timestamptz, default: now())
- `updated_at` (timestamptz, default: now())

**Upload CSV:** `scan_sequences.csv`

## Step 2: Enable Row Level Security (RLS)

For each table, go to Settings → Row Level Security and enable it.

## Step 3: Create RLS Policies

For each table, create these policies:

**For SELECT:**
```sql
CREATE POLICY "Users can view their own data" ON table_name FOR SELECT USING (auth.uid() = user_id);
```

**For INSERT:**
```sql
CREATE POLICY "Users can insert their own data" ON table_name FOR INSERT WITH CHECK (auth.uid() = user_id);
```

**For UPDATE:**
```sql
CREATE POLICY "Users can update their own data" ON table_name FOR UPDATE USING (auth.uid() = user_id);
```

**For DELETE:**
```sql
CREATE POLICY "Users can delete their own data" ON table_name FOR DELETE USING (auth.uid() = user_id);
```

## Step 4: Test Your Application

After creating these tables and uploading the CSV data, your application should work without 404 errors!

## Notes:

1. **CSV Format:** All CSV files use proper JSON formatting for complex fields
2. **Timestamps:** All dates are in ISO format (2025-01-27T15:00:00Z)
3. **UUIDs:** All IDs are properly formatted UUIDs
4. **Foreign Keys:** Make sure to set up foreign key relationships in the table editor

## Quick Setup Order:

1. Create `health_metrics_timeseries` table + upload CSV
2. Create `risk_progressions` table + upload CSV
3. Create `patient_timelines` table + upload CSV
4. Create `image_metadata` table + upload CSV
5. Create `scan_sequences` table + upload CSV
6. Enable RLS on all tables
7. Test your application!

This should resolve all the 404 errors you're seeing in the browser console.
