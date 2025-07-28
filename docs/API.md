# 📊 API Documentation

## Overview

The DoctAI Health Hub API provides comprehensive healthcare services including patient management, AI-powered diagnostics, appointment scheduling, and compliance features. This documentation covers all available endpoints, authentication, and usage examples.

## 📋 Table of Contents

- [🔐 Authentication](#-authentication)
- [🏥 Patient Management](#-patient-management)
- [🤖 AI Diagnostics](#-ai-diagnostics)
- [📅 Appointment Management](#-appointment-management)
- [💊 Treatment Management](#-treatment-management)
- [📊 Analytics & Reporting](#-analytics--reporting)
- [🔐 Compliance & Security](#-compliance--security)
- [📱 Notifications](#-notifications)
- [🛠️ Error Handling](#️-error-handling)
- [📈 Rate Limits](#-rate-limits)

---

## 🔐 Authentication

### API Keys

Most endpoints require authentication via API keys:

```bash
# Add to request headers
Authorization: Bearer YOUR_API_KEY
X-API-Key: YOUR_API_KEY
```

### JWT Tokens

For user-specific operations:

```bash
# Add to request headers
Authorization: Bearer YOUR_JWT_TOKEN
```

### OAuth 2.0

For third-party integrations:

```bash
# OAuth flow
GET /oauth/authorize?client_id=YOUR_CLIENT_ID&redirect_uri=YOUR_REDIRECT_URI&response_type=code&scope=read write

# Exchange code for token
POST /oauth/token
Content-Type: application/x-www-form-urlencoded

grant_type=authorization_code&code=YOUR_CODE&client_id=YOUR_CLIENT_ID&client_secret=YOUR_CLIENT_SECRET
```

## 📋 Base URL

```bash
# Development
https://api.doctai.com/v1

# Staging
https://staging-api.doctai.com/v1

# Production
https://api.doctai.com/v1
```

---

## 🏥 Patient Management

### Get Patient Profile

```http
GET /patients/{patient_id}
```

**Response:**
```json
{
  "id": "pat_123456789",
  "name": "John Doe",
  "email": "john.doe@email.com",
  "phone": "+1234567890",
  "dateOfBirth": "1985-03-15",
  "gender": "male",
  "address": {
    "street": "123 Main St",
    "city": "New York",
    "state": "NY",
    "zipCode": "10001"
  },
  "medicalHistory": {
    "allergies": ["penicillin"],
    "conditions": ["hypertension"],
    "medications": ["lisinopril"]
  },
  "createdAt": "2024-01-15T10:30:00Z",
  "updatedAt": "2024-01-20T14:45:00Z"
}
```

### Update Patient Profile

```http
PUT /patients/{patient_id}
```

**Request Body:**
```json
{
  "name": "John Doe",
  "email": "john.doe@email.com",
  "phone": "+1234567890",
  "address": {
    "street": "123 Main St",
    "city": "New York",
    "state": "NY",
    "zipCode": "10001"
  }
}
```

### Create Patient

```http
POST /patients
```

**Request Body:**
```json
{
  "name": "Jane Smith",
  "email": "jane.smith@email.com",
  "phone": "+1234567891",
  "dateOfBirth": "1990-07-22",
  "gender": "female",
  "address": {
    "street": "456 Oak Ave",
    "city": "Los Angeles",
    "state": "CA",
    "zipCode": "90210"
  }
}
```

### Search Patients

```http
GET /patients/search?query=john&limit=10&offset=0
```

**Response:**
```json
{
  "patients": [
    {
      "id": "pat_123456789",
      "name": "John Doe",
      "email": "john.doe@email.com"
    }
  ],
  "total": 1,
  "limit": 10,
  "offset": 0
}
```

---

## 🤖 AI Diagnostics

### Medical Image Analysis

```http
POST /ai/analyze-image
Content-Type: multipart/form-data
```

**Request Body:**
```json
{
  "image": "base64_encoded_image_or_file",
  "imageType": "xray|mri|ct|skin_lesion|eeg",
  "analysisType": "diagnosis|screening|monitoring",
  "priority": "routine|urgent|emergency"
}
```

**Response:**
```json
{
  "analysisId": "ana_123456789",
  "status": "completed",
  "results": {
    "diagnosis": "Normal chest X-ray",
    "confidence": 0.95,
    "findings": [
      {
        "finding": "No acute cardiopulmonary abnormality",
        "confidence": 0.98,
        "location": "chest"
      }
    ],
    "recommendations": [
      "Follow-up in 6 months for routine screening"
    ]
  },
  "processingTime": 2.5,
  "modelVersion": "v2.1.0"
}
```

### Get Analysis Status

```http
GET /ai/analysis/{analysis_id}
```

### Batch Analysis

```http
POST /ai/batch-analyze
```

**Request Body:**
```json
{
  "images": [
    {
      "id": "img_1",
      "data": "base64_encoded_image_1",
      "type": "xray"
    },
    {
      "id": "img_2",
      "data": "base64_encoded_image_2",
      "type": "mri"
    }
  ],
  "callbackUrl": "https://your-app.com/webhooks/analysis-complete"
}
```

### AI Chat Consultation

```http
POST /ai/chat
```

**Request Body:**
```json
{
  "message": "I have chest pain and shortness of breath",
  "patientId": "pat_123456789",
  "context": {
    "symptoms": ["chest pain", "shortness of breath"],
    "duration": "2 hours",
    "severity": "moderate"
  }
}
```

**Response:**
```json
{
  "response": "Based on your symptoms, I recommend seeking immediate medical attention. Chest pain with shortness of breath could indicate a serious condition.",
  "urgency": "high",
  "recommendations": [
    "Call 911 or go to nearest emergency room",
    "Avoid physical exertion",
    "Take prescribed medications if available"
  ],
  "followUp": {
    "timeline": "immediate",
    "type": "emergency_care"
  }
}
```

---

## 📅 Appointment Management

### Create Appointment

```http
POST /appointments
```

**Request Body:**
```json
{
  "patientId": "pat_123456789",
  "providerId": "prov_987654321",
  "appointmentType": "consultation|follow_up|emergency",
  "scheduledAt": "2024-02-15T14:30:00Z",
  "duration": 30,
  "reason": "Annual checkup",
  "location": {
    "type": "in_person|virtual",
    "address": "123 Medical Center Dr",
    "room": "Exam Room 3"
  }
}
```

### Get Appointments

```http
GET /appointments?patientId=pat_123456789&startDate=2024-02-01&endDate=2024-02-28
```

**Response:**
```json
{
  "appointments": [
    {
      "id": "apt_123456789",
      "patientId": "pat_123456789",
      "providerId": "prov_987654321",
      "appointmentType": "consultation",
      "scheduledAt": "2024-02-15T14:30:00Z",
      "duration": 30,
      "status": "confirmed",
      "location": {
        "type": "in_person",
        "address": "123 Medical Center Dr",
        "room": "Exam Room 3"
      }
    }
  ],
  "total": 1
}
```

### Update Appointment

```http
PUT /appointments/{appointment_id}
```

**Request Body:**
```json
{
  "scheduledAt": "2024-02-16T15:00:00Z",
  "status": "rescheduled",
  "notes": "Patient requested time change"
}
```

### Cancel Appointment

```http
DELETE /appointments/{appointment_id}
```

**Request Body:**
```json
{
  "reason": "Patient request",
  "cancelledBy": "patient"
}
```

---

## 💊 Treatment Management

### Create Treatment Plan

```http
POST /treatments
```

**Request Body:**
```json
{
  "patientId": "pat_123456789",
  "diagnosis": "Hypertension",
  "treatmentType": "medication|lifestyle|surgery",
  "medications": [
    {
      "name": "Lisinopril",
      "dosage": "10mg",
      "frequency": "once daily",
      "duration": "ongoing"
    }
  ],
  "instructions": "Take medication as prescribed, monitor blood pressure daily",
  "followUpDate": "2024-03-15T10:00:00Z"
}
```

### Get Treatment History

```http
GET /treatments/{patient_id}
```

**Response:**
```json
{
  "treatments": [
    {
      "id": "trt_123456789",
      "patientId": "pat_123456789",
      "diagnosis": "Hypertension",
      "treatmentType": "medication",
      "medications": [
        {
          "name": "Lisinopril",
          "dosage": "10mg",
          "frequency": "once daily",
          "status": "active"
        }
      ],
      "startDate": "2024-01-15T00:00:00Z",
      "endDate": null,
      "status": "active"
    }
  ]
}
```

### Update Treatment

```http
PUT /treatments/{treatment_id}
```

**Request Body:**
```json
{
  "medications": [
    {
      "name": "Lisinopril",
      "dosage": "20mg",
      "frequency": "once daily",
      "reason": "Blood pressure not controlled"
    }
  ],
  "notes": "Increased dosage due to persistent hypertension"
}
```

---

## 📊 Analytics & Reporting

### Get Patient Analytics

```http
GET /analytics/patient/{patient_id}?period=30d&metrics=appointments,medications,vitals
```

**Response:**
```json
{
  "patientId": "pat_123456789",
  "period": "30d",
  "metrics": {
    "appointments": {
      "total": 3,
      "completed": 2,
      "cancelled": 1,
      "attendanceRate": 0.67
    },
    "medications": {
      "active": 2,
      "adherence": 0.85,
      "refills": 1
    },
    "vitals": {
      "bloodPressure": {
        "average": "120/80",
        "trend": "stable"
      },
      "weight": {
        "current": "70kg",
        "change": "-2kg"
      }
    }
  }
}
```

### Get Provider Analytics

```http
GET /analytics/provider/{provider_id}?period=90d
```

**Response:**
```json
{
  "providerId": "prov_987654321",
  "period": "90d",
  "metrics": {
    "appointments": {
      "total": 150,
      "completed": 145,
      "cancelled": 5,
      "averageDuration": 25
    },
    "patients": {
      "new": 15,
      "returning": 135,
      "total": 150
    },
    "satisfaction": {
      "averageRating": 4.8,
      "totalReviews": 120
    }
  }
}
```

### Generate Report

```http
POST /analytics/reports
```

**Request Body:**
```json
{
  "reportType": "patient_summary|provider_performance|compliance_audit",
  "filters": {
    "startDate": "2024-01-01",
    "endDate": "2024-01-31",
    "patientIds": ["pat_123456789"],
    "providers": ["prov_987654321"]
  },
  "format": "pdf|csv|json"
}
```

---

## 🔐 Compliance & Security

### Audit Logs

```http
GET /compliance/audit-logs?startDate=2024-01-01&endDate=2024-01-31&userId=user_123
```

**Response:**
```json
{
  "logs": [
    {
      "id": "log_123456789",
      "timestamp": "2024-01-15T10:30:00Z",
      "userId": "user_123",
      "action": "read",
      "resource": "patient",
      "resourceId": "pat_123456789",
      "ipAddress": "192.168.1.100",
      "success": true
    }
  ],
  "total": 1
}
```

### Access Requests

```http
POST /compliance/access-requests
```

**Request Body:**
```json
{
  "userId": "user_123",
  "resource": "patient",
  "resourceId": "pat_123456789",
  "reason": "Emergency medical care required",
  "urgency": "high"
}
```

### Data Disposal

```http
POST /compliance/dispose-data
```

**Request Body:**
```json
{
  "resourceType": "appointment_records",
  "resourceIds": ["apt_123456789"],
  "disposalMethod": "secure_delete",
  "reason": "Retention period expired"
}
```

---

## 📱 Notifications

### Send Notification

```http
POST /notifications/send
```

**Request Body:**
```json
{
  "recipientId": "pat_123456789",
  "type": "appointment_reminder|medication_reminder|test_result",
  "channel": "email|sms|push",
  "title": "Appointment Reminder",
  "message": "Your appointment is scheduled for tomorrow at 2:30 PM",
  "data": {
    "appointmentId": "apt_123456789",
    "providerName": "Dr. Smith"
  }
}
```

### Get Notifications

```http
GET /notifications?recipientId=pat_123456789&status=unread
```

**Response:**
```json
{
  "notifications": [
    {
      "id": "not_123456789",
      "recipientId": "pat_123456789",
      "type": "appointment_reminder",
      "title": "Appointment Reminder",
      "message": "Your appointment is scheduled for tomorrow at 2:30 PM",
      "status": "unread",
      "createdAt": "2024-01-15T10:30:00Z"
    }
  ]
}
```

### Update Notification Status

```http
PUT /notifications/{notification_id}
```

**Request Body:**
```json
{
  "status": "read"
}
```

---

## 🛠️ Error Handling

### Error Response Format

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid request parameters",
    "details": [
      {
        "field": "email",
        "message": "Email format is invalid"
      }
    ],
    "requestId": "req_123456789",
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

### Common Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `VALIDATION_ERROR` | 400 | Request validation failed |
| `UNAUTHORIZED` | 401 | Authentication required |
| `FORBIDDEN` | 403 | Insufficient permissions |
| `NOT_FOUND` | 404 | Resource not found |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests |
| `INTERNAL_ERROR` | 500 | Server error |

### Retry Logic

```bash
# Exponential backoff for retries
# 1st retry: 1 second delay
# 2nd retry: 2 second delay
# 3rd retry: 4 second delay
# Maximum 3 retries
```

---

## 📈 Rate Limits

### Default Limits

- **Authentication**: 100 requests/minute
- **Patient Data**: 1000 requests/minute
- **AI Analysis**: 10 requests/minute
- **Appointments**: 500 requests/minute
- **Notifications**: 100 requests/minute

### Rate Limit Headers

```bash
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1642233600
```

### Rate Limit Exceeded Response

```json
{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded. Try again in 60 seconds.",
    "retryAfter": 60
  }
}
```

---

## 📞 Support

For API support:

1. **Documentation**: Check this guide for endpoint details
2. **Status Page**: [API Status](https://status.doctai.com)
3. **Support Email**: api-support@doctai.com
4. **Developer Portal**: [Developer Portal](https://developers.doctai.com)

## 🔗 Related Resources

- [SDK Documentation](https://github.com/doctai/doctai-sdk)
- [Webhook Guide](https://docs.doctai.com/webhooks)
- [Testing Guide](https://docs.doctai.com/testing)
- [Best Practices](https://docs.doctai.com/best-practices)
