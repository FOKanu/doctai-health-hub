# Health Extensions: Personalized Health Scores & Telemedicine Integration

## 🎯 Overview

This document describes the implementation of two major health analytics extensions:

1. **Personalized Health Scores** - Comprehensive health assessment combining multiple metrics
2. **Telemedicine Integration** - Remote health monitoring and virtual consultations

## 🏥 1. Personalized Health Scores System

### Features

#### **Multi-Domain Health Assessment**
- **Cardiovascular Health** (25% weight) - Heart rate, blood pressure, HRV
- **Metabolic Health** (20% weight) - Glucose, HbA1c, cholesterol
- **Sleep Quality** (20% weight) - Duration, efficiency, latency
- **Fitness Level** (15% weight) - Steps, VO2 max, active minutes
- **Mental Health** (10% weight) - Mood, stress, cognitive performance
- **Respiratory Health** (5% weight) - Oxygen saturation, respiratory rate
- **Hormonal Balance** (5% weight) - Thyroid, cortisol patterns

#### **Intelligent Scoring Algorithm**
```typescript
// Optimal ranges for each metric
const optimalRanges = {
  cardiovascular: {
    heartRateResting: { min: 60, max: 100, optimal: 70 },
    bloodPressureSystolic: { min: 90, max: 140, optimal: 120 },
    bloodPressureDiastolic: { min: 60, max: 90, optimal: 80 }
  },
  // ... other domains
};
```

#### **Personalized Recommendations**
- **Priority-based** (high/medium/low)
- **Actionable items** with expected impact
- **Timeframe guidance** (immediate/short-term/long-term)
- **Category-specific** recommendations

### Database Schema

#### **Extended Health Metrics Tables**
```sql
-- Cardiovascular metrics
CREATE TABLE cardiovascular_metrics (
  id UUID PRIMARY KEY,
  user_id UUID REFERENCES auth.users(id),
  heart_rate_resting DECIMAL(5,2),
  heart_rate_active DECIMAL(5,2),
  heart_rate_variability DECIMAL(5,2),
  blood_pressure_systolic INTEGER,
  blood_pressure_diastolic INTEGER,
  pulse_pressure DECIMAL(5,2),
  mean_arterial_pressure DECIMAL(5,2),
  ecg_rhythm TEXT,
  qt_interval DECIMAL(5,2),
  st_segment DECIMAL(5,2),
  recorded_at TIMESTAMP WITH TIME ZONE,
  device_source TEXT,
  accuracy_score DECIMAL(3,2),
  metadata JSONB,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Similar tables for other health domains
```

### API Endpoints

#### **Health Scoring Service**
```typescript
// Calculate comprehensive health score
await healthScoringService.calculateHealthScore(userId);

// Get detailed breakdown
await healthScoringService.getHealthScoreBreakdown(userId);

// Get personalized recommendations
await healthScoringService.getPersonalizedRecommendations(userId);

// Get health score trends
await healthScoringService.getHealthScoreTrends(userId, days);
```

### Frontend Components

#### **HealthScoreCard Component**
```tsx
<HealthScoreCard userId={userId} />
```

**Features:**
- **Overview Tab** - Overall score with domain breakdown
- **Breakdown Tab** - Detailed factor analysis
- **Recommendations Tab** - Personalized action items
- **Real-time updates** with refresh capability
- **Visual indicators** for trends and risk levels

## 🏥 2. Telemedicine Integration System

### Features

#### **Healthcare Provider Management**
- **Provider profiles** with specialties and ratings
- **Availability scheduling** with real-time slots
- **Verification system** for provider credentials
- **Multi-language support** and experience tracking

#### **Virtual Consultation Types**
- **Video consultations** with screen sharing
- **Audio-only consultations** for privacy
- **Chat-based consultations** for quick questions
- **Follow-up consultations** with history tracking
- **Emergency consultations** with priority routing

#### **Remote Health Monitoring**
- **Continuous monitoring** for critical metrics
- **Alert thresholds** with customizable levels
- **Device integration** (smartwatches, medical devices)
- **Real-time notifications** for providers and patients

#### **Appointment Management**
- **Smart scheduling** with availability checking
- **Reminder system** with multiple channels
- **Urgent appointment** prioritization
- **Cancellation and rescheduling** workflows

### Database Schema

#### **Telemedicine Tables**
```sql
-- Healthcare providers
CREATE TABLE healthcare_providers (
  id UUID PRIMARY KEY,
  user_id UUID REFERENCES auth.users(id),
  provider_name TEXT NOT NULL,
  specialty provider_specialty NOT NULL,
  license_number TEXT,
  credentials TEXT[],
  experience_years INTEGER,
  languages TEXT[],
  availability_schedule JSONB,
  consultation_fee DECIMAL(8,2),
  rating DECIMAL(3,2),
  total_consultations INTEGER DEFAULT 0,
  is_verified BOOLEAN DEFAULT false,
  is_available BOOLEAN DEFAULT true,
  profile_image_url TEXT,
  bio TEXT,
  contact_info JSONB,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Telemedicine consultations
CREATE TABLE telemedicine_consultations (
  id UUID PRIMARY KEY,
  patient_id UUID REFERENCES auth.users(id),
  provider_id UUID REFERENCES healthcare_providers(id),
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
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Remote health monitoring
CREATE TABLE remote_monitoring (
  id UUID PRIMARY KEY,
  patient_id UUID REFERENCES auth.users(id),
  provider_id UUID REFERENCES healthcare_providers(id),
  monitoring_type TEXT NOT NULL,
  device_id TEXT,
  device_type TEXT,
  is_active BOOLEAN DEFAULT true,
  start_date TIMESTAMP WITH TIME ZONE NOT NULL,
  end_date TIMESTAMP WITH TIME ZONE,
  monitoring_frequency TEXT,
  alert_thresholds JSONB DEFAULT '{}',
  last_reading_at TIMESTAMP WITH TIME ZONE,
  last_reading_value JSONB,
  alert_level monitoring_alert_level,
  is_alert_active BOOLEAN DEFAULT false,
  alert_message TEXT,
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Health alerts
CREATE TABLE health_alerts (
  id UUID PRIMARY KEY,
  patient_id UUID REFERENCES auth.users(id),
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
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

### API Endpoints

#### **Telemedicine Service**
```typescript
// Provider management
await telemedicineService.getHealthcareProviders(specialty);
await telemedicineService.getProviderById(providerId);
await telemedicineService.getProviderAvailability(providerId, date);

// Appointment booking
await telemedicineService.bookAppointment(params);
await telemedicineService.getAppointments(params);
await telemedicineService.updateAppointmentStatus(appointmentId, status);

// Consultations
await telemedicineService.createConsultation(consultation);
await telemedicineService.getConsultations(params);
await telemedicineService.updateConsultation(consultationId, updates);
await telemedicineService.joinConsultation(consultationId, userId);

// Remote monitoring
await telemedicineService.startRemoteMonitoring(params);
await telemedicineService.getRemoteMonitoring(params);
await telemedicineService.updateMonitoringReading(monitoringId, reading);
await telemedicineService.stopRemoteMonitoring(monitoringId);

// Health alerts
await telemedicineService.createHealthAlert(params);
await telemedicineService.getHealthAlerts(params);
await telemedicineService.markAlertAsRead(alertId);
await telemedicineService.getUnreadAlertsCount(userId);
```

### Frontend Components

#### **TelemedicineConsultation Component**
```tsx
<TelemedicineConsultation userId={userId} />
```

**Features:**
- **Book Consultation Tab** - Provider selection and scheduling
- **My Appointments Tab** - Upcoming appointment management
- **Past Consultations Tab** - Consultation history and records
- **Real-time availability** checking
- **Multi-format consultations** (video/audio/chat)
- **Symptom tracking** and urgent flagging

## 🔧 Implementation Details

### Environment Variables

```bash
# Health Scoring
VITE_ENABLE_HEALTH_SCORING=true
VITE_HEALTH_SCORE_WEIGHTS={"cardiovascular":0.25,"metabolic":0.20,"sleep":0.20,"fitness":0.15,"mentalHealth":0.10,"respiratory":0.05,"hormonal":0.05}

# Telemedicine
VITE_ENABLE_TELEMEDICINE=true
VITE_VIDEO_PROVIDER=zoom  # or teams, meet, custom
VITE_TELEMEDICINE_API_KEY=your_api_key
VITE_ALERT_NOTIFICATION_ENABLED=true
```

### Security & Compliance

#### **Data Protection**
- **HIPAA compliance** for all health data
- **End-to-end encryption** for consultations
- **Audit trails** for all interactions
- **Role-based access** control (RLS)

#### **Privacy Features**
- **Patient consent** management
- **Data anonymization** for analytics
- **Secure video** consultations
- **Encrypted recordings** with expiration

### Performance Optimization

#### **Health Scoring**
- **Caching** of calculated scores
- **Incremental updates** for real-time metrics
- **Background processing** for heavy calculations
- **Optimized queries** with proper indexing

#### **Telemedicine**
- **WebRTC optimization** for video calls
- **CDN integration** for global access
- **Real-time notifications** via WebSockets
- **Offline capability** for critical functions

## 🚀 Deployment

### Database Migration
```bash
# Run the new migrations
supabase db push

# Verify the new tables
supabase db diff
```

### Service Integration
```bash
# Install dependencies
npm install

# Build the application
npm run build

# Deploy to production
npm run deploy
```

### Health Check
```bash
# Verify health scoring service
curl -X GET /api/health/score/test

# Verify telemedicine service
curl -X GET /api/telemedicine/providers
```

## 📊 Analytics & Monitoring

### Health Score Analytics
- **Trend analysis** over time
- **Population health** insights
- **Risk factor** identification
- **Intervention effectiveness** tracking

### Telemedicine Analytics
- **Consultation volume** tracking
- **Provider performance** metrics
- **Patient satisfaction** scores
- **System utilization** monitoring

## 🔮 Future Enhancements

### Health Scoring
- **AI-powered** recommendations
- **Predictive analytics** for health risks
- **Integration** with wearable devices
- **Social determinants** of health

### Telemedicine
- **AI triage** system
- **Virtual waiting rooms**
- **Prescription management**
- **Insurance integration**

## 📝 API Documentation

### Health Scoring Endpoints

#### `GET /api/health/score/{userId}`
Calculate comprehensive health score for a user.

**Response:**
```json
{
  "overall": 78,
  "cardiovascular": 75,
  "metabolic": 82,
  "sleep": 70,
  "fitness": 65,
  "mentalHealth": 80,
  "respiratory": 85,
  "hormonal": 78,
  "trend": "improving",
  "riskLevel": "medium",
  "insights": ["Your cardiovascular health could benefit from more exercise."],
  "recommendations": ["Start with 30 minutes of moderate exercise daily"],
  "lastUpdated": "2024-01-15T10:30:00Z"
}
```

#### `GET /api/health/score/{userId}/breakdown`
Get detailed breakdown of health score components.

#### `GET /api/health/score/{userId}/recommendations`
Get personalized health recommendations.

### Telemedicine Endpoints

#### `GET /api/telemedicine/providers`
Get available healthcare providers.

#### `POST /api/telemedicine/appointments`
Book a new appointment.

#### `GET /api/telemedicine/consultations/{userId}`
Get user's consultation history.

#### `POST /api/telemedicine/monitoring/start`
Start remote health monitoring.

## 🛠️ Troubleshooting

### Common Issues

#### Health Scoring
- **Insufficient data** - Ensure user has enough health metrics
- **Calculation errors** - Check metric ranges and weights
- **Performance issues** - Optimize database queries

#### Telemedicine
- **Video call issues** - Check WebRTC configuration
- **Scheduling conflicts** - Verify provider availability
- **Alert delays** - Check notification service status

### Debug Commands
```bash
# Check health scoring service
npm run test:health-scoring

# Check telemedicine service
npm run test:telemedicine

# Monitor system logs
npm run logs:health
```

## 📞 Support

For technical support or questions about these health extensions:

- **Documentation**: See inline code comments
- **Issues**: Create GitHub issues with detailed descriptions
- **Email**: health-support@example.com
- **Slack**: #health-extensions channel

---

**Version**: 1.0.0
**Last Updated**: January 2024
**Maintainer**: Health Analytics Team
