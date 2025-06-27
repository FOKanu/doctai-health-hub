# 🚀 API Integration Guide for DoctAI Health Hub

This comprehensive guide walks you through integrating various APIs into your DoctAI Health Hub application.

## 📋 Table of Contents

1. [Environment Setup](#environment-setup)
2. [API Services Overview](#api-services-overview)
3. [Installation & Dependencies](#installation--dependencies)
4. [Service Implementation](#service-implementation)
5. [Component Integration](#component-integration)
6. [Testing & Validation](#testing--validation)
7. [Production Deployment](#production-deployment)
8. [Troubleshooting](#troubleshooting)

## 🔧 Environment Setup

### Step 1: Create Environment File

Create a `.env` file in your project root:

```bash
# =============================================================================
# CLOUD HEALTHCARE APIs (Already implemented)
# =============================================================================
VITE_USE_CLOUD_HEALTHCARE=true
VITE_CLOUD_HEALTHCARE_FALLBACK=true
VITE_ENABLE_CONSENSUS_ANALYSIS=false
VITE_PRIMARY_CLOUD_PROVIDER=google

# Google Cloud Healthcare
VITE_ENABLE_GOOGLE_HEALTHCARE=true
VITE_GOOGLE_HEALTHCARE_PROJECT_ID=doctai-project
VITE_GOOGLE_HEALTHCARE_LOCATION=us-central1
VITE_GOOGLE_HEALTHCARE_DATASET_ID=your-dataset-id
VITE_GOOGLE_HEALTHCARE_API_KEY=your-api-key

# Azure Health Bot
VITE_ENABLE_AZURE_HEALTH_BOT=true
VITE_AZURE_HEALTH_BOT_ENDPOINT=https://your-bot.azurewebsites.net
VITE_AZURE_HEALTH_BOT_API_KEY=your-api-key

# IBM Watson Health
VITE_ENABLE_WATSON_HEALTH=true
VITE_WATSON_HEALTH_API_KEY=your-api-key
VITE_WATSON_HEALTH_ENDPOINT=https://your-watson-instance.com
VITE_WATSON_HEALTH_VERSION=2023-01-01

# =============================================================================
# NEW API INTEGRATIONS
# =============================================================================

# OpenAI API (for AI-powered features)
VITE_OPENAI_API_KEY=your-openai-api-key
VITE_OPENAI_MODEL=gpt-4

# Twilio API (for SMS notifications)
VITE_TWILIO_ACCOUNT_SID=your-twilio-account-sid
VITE_TWILIO_AUTH_TOKEN=your-twilio-auth-token
VITE_TWILIO_PHONE_NUMBER=+1234567890

# SendGrid API (for email notifications)
VITE_SENDGRID_API_KEY=your-sendgrid-api-key
VITE_SENDGRID_FROM_EMAIL=noreply@doctai.com

# Performance Settings
VITE_CLOUD_HEALTHCARE_TIMEOUT=30000
VITE_CLOUD_HEALTHCARE_MAX_RETRIES=3
VITE_CLOUD_HEALTHCARE_DEBUG=false
```

### Step 2: Get API Keys

#### OpenAI API
1. Visit [OpenAI Platform](https://platform.openai.com/)
2. Create an account and verify your email
3. Navigate to API Keys section
4. Create a new API key
5. Copy the key to your `.env` file

#### Twilio API
1. Visit [Twilio Console](https://console.twilio.com/)
2. Create an account
3. Get your Account SID and Auth Token
4. Purchase a phone number
5. Add credentials to your `.env` file

#### SendGrid API
1. Visit [SendGrid](https://sendgrid.com/)
2. Create an account
3. Navigate to Settings > API Keys
4. Create a new API key
5. Add to your `.env` file

## 📦 Installation & Dependencies

### Install Required Packages

```bash
# Core API libraries
npm install axios @tanstack/react-query

# Payment processing (optional)
npm install @stripe/stripe-js

# Maps and location (optional)
npm install @googlemaps/js-api-loader

# Utilities
npm install date-fns lodash
```

## 🏗️ API Services Overview

### 1. Base API Service (`src/services/api/baseApiService.ts`)
- **Purpose**: Common functionality for all API integrations
- **Features**:
  - Error handling and retries
  - Authentication management
  - Request/response interceptors
  - Timeout management

### 2. OpenAI Service (`src/services/api/openaiService.ts`)
- **Purpose**: AI-powered health insights and analysis
- **Features**:
  - Symptom analysis
  - Medical term explanations
  - Health tips generation
  - Medical report summarization

### 3. Notification Service (`src/services/api/notificationService.ts`)
- **Purpose**: Multi-channel notifications
- **Features**:
  - SMS notifications (Twilio)
  - Email notifications (SendGrid)
  - Push notifications
  - Appointment reminders
  - Medication reminders
  - Emergency alerts

### 4. API Service Manager (`src/services/api/apiServiceManager.ts`)
- **Purpose**: Centralized service management
- **Features**:
  - Service initialization
  - Health checks
  - Status monitoring
  - Error handling

## 🔧 Service Implementation

### Step 1: Initialize Services in App.tsx

```typescript
// src/App.tsx
import { apiServiceManager } from "./services/api/apiServiceManager";

const initializeApiServices = () => {
  const config = {
    openai: {
      apiKey: import.meta.env.VITE_OPENAI_API_KEY,
      model: import.meta.env.VITE_OPENAI_MODEL || 'gpt-4',
      maxTokens: 1000,
      temperature: 0.7
    },
    notifications: {
      twilio: {
        accountSid: import.meta.env.VITE_TWILIO_ACCOUNT_SID,
        authToken: import.meta.env.VITE_TWILIO_AUTH_TOKEN,
        phoneNumber: import.meta.env.VITE_TWILIO_PHONE_NUMBER
      },
      sendGrid: {
        apiKey: import.meta.env.VITE_SENDGRID_API_KEY,
        fromEmail: import.meta.env.VITE_SENDGRID_FROM_EMAIL
      }
    }
  };

  apiServiceManager.initialize(config);
};
```

### Step 2: Use Services in Components

```typescript
// Example: Using OpenAI service
import { apiServiceManager } from '@/services/api/apiServiceManager';

const MyComponent = () => {
  const handleHealthAnalysis = async (symptoms: string[]) => {
    try {
      const result = await apiServiceManager.generateHealthInsights(symptoms);
      if (result.success) {
        console.log('Health insights:', result.data);
      }
    } catch (error) {
      console.error('Error:', error);
    }
  };

  return (
    <button onClick={() => handleHealthAnalysis(['fever', 'cough'])}>
      Analyze Symptoms
    </button>
  );
};
```

## 🧩 Component Integration

### 1. AI Health Assistant Component

```typescript
import { HealthAssistant } from '@/components/ai/HealthAssistant';

// In your page/component
<HealthAssistant className="w-full" />
```

**Features**:
- Symptom analysis
- Medical term explanations
- Health tips generation
- Real-time AI responses

### 2. Notification Manager Component

```typescript
import { NotificationManager } from '@/components/notifications/NotificationManager';

// In your page/component
<NotificationManager className="w-full" />
```

**Features**:
- Appointment reminders
- Medication reminders
- Emergency alerts
- Multi-channel delivery (SMS, Email, Push)

## 🧪 Testing & Validation

### Step 1: Test API Services

```typescript
// Test OpenAI service
const testOpenAI = async () => {
  try {
    const result = await apiServiceManager.explainMedicalTerm('hypertension');
    console.log('OpenAI test result:', result);
  } catch (error) {
    console.error('OpenAI test failed:', error);
  }
};

// Test notification service
const testNotifications = async () => {
  try {
    const result = await apiServiceManager.sendAppointmentReminder(
      'user123',
      {
        date: '2024-01-15',
        time: '10:00',
        doctor: 'Dr. Smith',
        specialty: 'Cardiology'
      },
      { email: 'test@example.com' }
    );
    console.log('Notification test result:', result);
  } catch (error) {
    console.error('Notification test failed:', error);
  }
};
```

### Step 2: Health Check

```typescript
// Check all services
const checkServices = async () => {
  const status = await apiServiceManager.healthCheck();
  console.log('Service status:', status);

  if (status.errors.length > 0) {
    console.error('Service errors:', status.errors);
  }
};
```

### Step 3: Environment Validation

```typescript
// Validate environment variables
const validateEnvironment = () => {
  const requiredVars = [
    'VITE_OPENAI_API_KEY',
    'VITE_TWILIO_ACCOUNT_SID',
    'VITE_SENDGRID_API_KEY'
  ];

  const missing = requiredVars.filter(varName => !import.meta.env[varName]);

  if (missing.length > 0) {
    console.warn('Missing environment variables:', missing);
  }
};
```

## 🚀 Production Deployment

### Step 1: Environment Configuration

1. **Set up production environment variables**:
   ```bash
   # Production .env
   VITE_OPENAI_API_KEY=sk-prod-your-production-key
   VITE_TWILIO_ACCOUNT_SID=your-production-sid
   VITE_SENDGRID_API_KEY=your-production-sendgrid-key
   ```

2. **Configure API rate limits**:
   ```bash
   VITE_CLOUD_HEALTHCARE_TIMEOUT=45000
   VITE_CLOUD_HEALTHCARE_MAX_RETRIES=5
   ```

### Step 2: Security Considerations

1. **API Key Security**:
   - Never commit API keys to version control
   - Use environment variables for all sensitive data
   - Rotate API keys regularly
   - Monitor API usage for anomalies

2. **Rate Limiting**:
   - Implement client-side rate limiting
   - Monitor API quotas
   - Set up alerts for quota exhaustion

3. **Error Handling**:
   - Implement graceful fallbacks
   - Log errors for monitoring
   - Provide user-friendly error messages

### Step 3: Monitoring & Analytics

```typescript
// Add monitoring to your services
const monitorApiCall = async (serviceName: string, call: () => Promise<any>) => {
  const startTime = Date.now();
  try {
    const result = await call();
    const duration = Date.now() - startTime;

    // Log success metrics
    console.log(`${serviceName} call successful in ${duration}ms`);
    return result;
  } catch (error) {
    const duration = Date.now() - startTime;

    // Log error metrics
    console.error(`${serviceName} call failed after ${duration}ms:`, error);
    throw error;
  }
};
```

## 🔍 Troubleshooting

### Common Issues

#### 1. "OpenAI service not initialized"
**Cause**: Missing or invalid API key
**Solution**:
- Check `VITE_OPENAI_API_KEY` in your `.env` file
- Verify the API key is valid
- Ensure the service is properly initialized

#### 2. "Twilio SMS service not configured"
**Cause**: Missing Twilio credentials
**Solution**:
- Verify `VITE_TWILIO_ACCOUNT_SID` and `VITE_TWILIO_AUTH_TOKEN`
- Check if phone number is purchased and active
- Ensure proper formatting of phone numbers

#### 3. "SendGrid email service not configured"
**Cause**: Missing SendGrid API key
**Solution**:
- Verify `VITE_SENDGRID_API_KEY` is set
- Check if the API key has proper permissions
- Ensure `VITE_SENDGRID_FROM_EMAIL` is verified

#### 4. API Rate Limiting
**Cause**: Exceeding API quotas
**Solution**:
- Implement exponential backoff
- Add request queuing
- Monitor usage and upgrade plans if needed

### Debug Mode

Enable debug mode for detailed logging:

```bash
# Add to your .env file
VITE_CLOUD_HEALTHCARE_DEBUG=true
VITE_CLOUD_HEALTHCARE_LOG_REQUESTS=true
```

### Error Logging

```typescript
// Enhanced error logging
const logApiError = (service: string, error: any, context?: any) => {
  console.error(`[${service}] API Error:`, {
    message: error.message,
    status: error.status,
    timestamp: new Date().toISOString(),
    context
  });

  // Send to monitoring service (e.g., Sentry)
  // Sentry.captureException(error);
};
```

## 📊 Usage Examples

### 1. Symptom Analysis

```typescript
const analyzeSymptoms = async () => {
  const symptoms = ['fever', 'cough', 'fatigue'];
  const result = await apiServiceManager.generateHealthInsights(symptoms);

  if (result.success) {
    const { possibleConditions, severity, recommendations } = result.data;
    console.log('Possible conditions:', possibleConditions);
    console.log('Severity:', severity);
    console.log('Recommendations:', recommendations);
  }
};
```

### 2. Medical Term Explanation

```typescript
const explainTerm = async () => {
  const result = await apiServiceManager.explainMedicalTerm('hypertension');

  if (result.success) {
    const { explanation, simplified } = result.data;
    console.log('Medical explanation:', explanation);
    console.log('Simplified:', simplified);
  }
};
```

### 3. Send Appointment Reminder

```typescript
const sendReminder = async () => {
  const result = await apiServiceManager.sendAppointmentReminder(
    'user123',
    {
      date: '2024-01-15',
      time: '10:00 AM',
      doctor: 'Dr. Johnson',
      specialty: 'Cardiology'
    },
    {
      phone: '+1234567890',
      email: 'patient@example.com'
    }
  );

  if (result.success) {
    console.log('Reminders sent:', result.data.length);
  }
};
```

### 4. Generate Health Tips

```typescript
const getHealthTips = async () => {
  const result = await apiServiceManager.generateHealthTips('nutrition', 5);

  if (result.success) {
    console.log('Health tips:', result.data);
  }
};
```

## 🔄 Next Steps

### Additional API Integrations

Consider adding these APIs for enhanced functionality:

1. **Payment Processing** (Stripe)
2. **Maps & Location** (Google Maps)
3. **Fitness Tracking** (Fitbit, Apple Health)
4. **Pharmacy APIs** (GoodRx)
5. **Insurance APIs** (Coverage verification)
6. **Lab Results APIs** (Direct integration)
7. **Telemedicine APIs** (Zoom, Teams)

### Performance Optimization

1. **Caching**: Implement result caching for expensive API calls
2. **Batching**: Batch multiple API requests where possible
3. **Lazy Loading**: Load API services only when needed
4. **Connection Pooling**: Optimize HTTP connections

### Security Enhancements

1. **API Key Encryption**: Encrypt API keys in storage
2. **Request Signing**: Sign API requests for additional security
3. **IP Whitelisting**: Restrict API access to specific IPs
4. **Audit Logging**: Log all API interactions for compliance

---

## 📞 Support

For issues with API integrations:

1. Check the troubleshooting section above
2. Verify your API keys and credentials
3. Review API provider documentation
4. Enable debug mode for detailed logs
5. Check your API usage quotas and limits

## 📚 Resources

- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Twilio API Documentation](https://www.twilio.com/docs)
- [SendGrid API Documentation](https://sendgrid.com/docs/api-reference/)
- [Google Cloud Healthcare API](https://cloud.google.com/healthcare/docs)
- [Azure Health Bot](https://docs.microsoft.com/en-us/health-bot/)
- [IBM Watson Health](https://www.ibm.com/watson-health)

---

**Happy integrating! 🚀**
