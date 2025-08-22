# 🏥 DoctAI Health Hub - Database Setup Guide

This guide will help you set up all the required Supabase tables and populate them with sample data for all user roles (Patient, Provider, Engineer).

## 📋 Prerequisites

1. **Supabase CLI** installed:
   ```bash
   npm install -g supabase
   ```

2. **Supabase Project** linked to your local development:
   ```bash
   supabase link --project-ref YOUR_PROJECT_REF
   ```

## 🚀 Quick Setup

### Option 1: Automated Setup (Recommended)

Run the automated setup script:

```bash
./scripts/setup-database.sh
```

This script will:
- ✅ Run all database migrations
- ✅ Create all required tables
- ✅ Populate with sample data for all roles
- ✅ Set up Row Level Security (RLS) policies
- ✅ Create indexes for optimal performance

### Option 2: Manual Setup

If you prefer to run migrations manually:

```bash
# Reset database and run all migrations
supabase db reset --linked

# Apply seed data
supabase db reset --linked
```

## 📊 Database Schema Overview

### Core Tables

| Table | Purpose | User Roles |
|-------|---------|------------|
| `health_metrics_timeseries` | Time-series health data | Patient |
| `risk_progressions` | Risk assessment tracking | Patient |
| `patient_timelines` | Patient condition tracking | Patient |
| `healthcare_providers` | Provider profiles | Provider |
| `telemedicine_consultations` | Virtual consultations | Patient, Provider |
| `appointments` | Appointment scheduling | Patient, Provider |
| `image_metadata` | Medical image storage | Patient, Provider |
| `scan_sequences` | Medical scan analysis | Patient, Provider |

### Specialized Health Tables

| Table | Purpose | Data Type |
|-------|---------|-----------|
| `cardiovascular_metrics` | Heart health data | Heart rate, blood pressure |
| `sleep_metrics` | Sleep quality data | Duration, efficiency, stages |
| `fitness_metrics` | Physical activity data | Steps, calories, VO2 max |
| `mental_health_metrics` | Mental wellness data | Mood, stress, anxiety |
| `metabolic_metrics` | Metabolic health data | Blood glucose, cholesterol |
| `respiratory_metrics` | Respiratory health data | Oxygen saturation, lung capacity |
| `hormonal_metrics` | Hormone levels data | Cortisol, thyroid, sex hormones |

### Telemedicine Tables

| Table | Purpose | Features |
|-------|---------|----------|
| `remote_monitoring` | Remote health monitoring | Device integration, alerts |
| `health_alerts` | Health notifications | Real-time alerts, notifications |
| `provider_availability` | Provider scheduling | Availability management |
| `patient_provider_relationships` | Care relationships | Patient-provider connections |

## 👥 User Roles & Permissions

### 🏥 Patient Role
- **Access**: Own health data, appointments, consultations
- **Tables**: `health_metrics_timeseries`, `patient_timelines`, `appointments`
- **Features**: Health tracking, appointment booking, telemedicine

### 👨‍⚕️ Provider Role
- **Access**: Patient data (with consent), own provider profile
- **Tables**: `healthcare_providers`, `telemedicine_consultations`, `appointments`
- **Features**: Patient management, consultations, scheduling

### 🔧 Engineer Role
- **Access**: System monitoring, analytics, technical data
- **Tables**: All tables (read-only for analytics)
- **Features**: System health, performance monitoring, debugging

## 🔐 Row Level Security (RLS)

All tables have RLS policies that ensure:

- **Patients** can only access their own data
- **Providers** can access patient data with proper relationships
- **Engineers** have read-only access for analytics
- **Data isolation** between different users

## 📈 Sample Data

The setup includes comprehensive sample data:

### Health Metrics
- Heart rate, blood pressure, temperature
- Sleep quality, steps, water intake
- Cardiovascular, respiratory, metabolic data

### Risk Assessments
- Cardiovascular risk: Low (15% probability)
- Metabolic risk: Medium (45% probability)
- Respiratory risk: Low (8% probability)

### Appointments & Consultations
- Scheduled appointments
- Completed telemedicine consultations
- Provider availability schedules

### Medical Images
- Sample skin lesion analysis
- Chest X-ray data
- Scan sequence tracking

## 🧪 Test Credentials

After setup, you can test with these credentials:

| Role | Email | Password | Features |
|------|-------|----------|----------|
| Patient | `mock_user@doctai.com` | `password123` | Health tracking, appointments |
| Provider | `provider@doctai.com` | `password123` | Patient management, consultations |
| Engineer | `engineer@doctai.com` | `password123` | System monitoring, analytics |

## 🔧 Troubleshooting

### Common Issues

1. **Migration Errors**
   ```bash
   # Reset database completely
   supabase db reset --linked
   ```

2. **RLS Policy Issues**
   ```bash
   # Check RLS policies
   supabase db diff --linked
   ```

3. **Sample Data Not Loading**
   ```bash
   # Re-run seed script
   supabase db reset --linked
   ```

### Verification Commands

Check if tables exist:
```sql
SELECT table_name FROM information_schema.tables
WHERE table_schema = 'public'
ORDER BY table_name;
```

Check sample data:
```sql
SELECT COUNT(*) FROM health_metrics_timeseries;
SELECT COUNT(*) FROM healthcare_providers;
SELECT COUNT(*) FROM appointments;
```

## 🚀 Next Steps

After successful setup:

1. **Start the application**:
   ```bash
   npm run dev
   ```

2. **Test the login flow** with the provided credentials

3. **Verify data loading** in the analytics dashboard

4. **Check console errors** - they should be resolved

## 📞 Support

If you encounter issues:

1. Check the Supabase dashboard for migration status
2. Verify RLS policies are properly applied
3. Ensure sample data is loaded correctly
4. Check application logs for specific errors

The database is now ready for all user roles with comprehensive health tracking, telemedicine, and analytics capabilities!
