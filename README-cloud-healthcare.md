# Cloud Healthcare API Integration

This document describes the cloud healthcare API integration for DoctAI Health Hub, which provides access to Google Cloud Healthcare, Azure Health Bot, and IBM Watson Health APIs alongside your existing custom ML models.

## 🎯 Overview

The cloud healthcare integration provides:
- **Multiple Provider Support**: Google Cloud Healthcare, Azure Health Bot, IBM Watson Health
- **Seamless Fallback**: Automatic fallback to custom ML models if cloud APIs fail
- **Consensus Analysis**: Get results from multiple providers for higher accuracy
- **Feature Flags**: Easy enable/disable of features via environment variables
- **Backward Compatibility**: Works with existing prediction service

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │    │  Prediction      │    │  Cloud          │
│   Components    │───▶│  Service         │───▶│  Healthcare     │
│                 │    │                  │    │  APIs           │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │  Custom ML       │
                       │  Models          │
                       │  (Fallback)      │
                       └──────────────────┘
```

## 🚀 Quick Start

### 1. Enable Cloud Healthcare

Set the following environment variables in your `.env` file:

```bash
# Enable cloud healthcare APIs
VITE_USE_CLOUD_HEALTHCARE=true

# Enable fallback to custom ML models
VITE_CLOUD_HEALTHCARE_FALLBACK=true

# Choose primary provider
VITE_PRIMARY_CLOUD_PROVIDER=google
```

### 2. Configure Google Cloud Healthcare

```bash
# Google Cloud Healthcare
VITE_ENABLE_GOOGLE_HEALTHCARE=true
VITE_GOOGLE_HEALTHCARE_PROJECT_ID=doctai-project
VITE_GOOGLE_HEALTHCARE_LOCATION=us-central1
VITE_GOOGLE_HEALTHCARE_DATASET_ID=your-dataset-id
VITE_GOOGLE_HEALTHCARE_API_KEY=your-api-key
```

### 3. Configure Azure Health Bot

```bash
# Azure Health Bot
VITE_ENABLE_AZURE_HEALTH_BOT=true
VITE_AZURE_HEALTH_BOT_ENDPOINT=https://your-bot.azurewebsites.net
VITE_AZURE_HEALTH_BOT_API_KEY=your-api-key
```

### 4. Configure IBM Watson Health

```bash
# IBM Watson Health
VITE_ENABLE_WATSON_HEALTH=true
VITE_WATSON_HEALTH_API_KEY=your-api-key
VITE_WATSON_HEALTH_ENDPOINT=https://your-watson-instance.com
VITE_WATSON_HEALTH_VERSION=2023-01-01
```

## 📋 API Reference

### Core Functions

#### `analyzeImage(imageUri: string, imageType: ImageType)`

Analyzes medical images using cloud healthcare APIs with fallback to custom ML models.

```typescript
import { analyzeImage } from '@/services/predictionService';

const result = await analyzeImage(imageUri, 'skin_lesion');
console.log(result.prediction); // 'benign' | 'malignant'
console.log(result.confidence); // 0.85
console.log(result.metadata.provider); // 'google' | 'azure' | 'watson' | 'custom_ml'
```

#### `assessSymptoms(symptoms: string[], patientContext?: any)`

Assesses symptoms using Azure Health Bot or Watson Health.

```typescript
import { assessSymptoms } from '@/services/predictionService';

const assessment = await assessSymptoms(['fever', 'cough'], { age: 30 });
console.log(assessment.severity); // 'low' | 'medium' | 'high' | 'critical'
console.log(assessment.urgency); // 'routine' | 'soon' | 'urgent' | 'emergency'
```

#### `getClinicalInsights(patientData: any)`

Gets clinical insights from Watson Health.

```typescript
import { getClinicalInsights } from '@/services/predictionService';

const insights = await getClinicalInsights({
  age: 45,
  gender: 'male',
  medicalHistory: ['diabetes'],
  currentSymptoms: ['chest pain']
});
```

#### `triageEmergency(symptoms: string[])`

Performs emergency triage using Azure Health Bot.

```typescript
import { triageEmergency } from '@/services/predictionService';

const triage = await triageEmergency(['chest pain', 'shortness of breath']);
console.log(triage.isEmergency); // true
console.log(triage.urgency); // 'emergency'
```

### Utility Functions

#### `getCloudHealthcareStatus()`

Returns the current status of cloud healthcare providers.

```typescript
import { getCloudHealthcareStatus } from '@/services/predictionService';

const status = getCloudHealthcareStatus();
console.log(status.available); // true
console.log(status.providers); // ['google', 'azure', 'watson']
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VITE_USE_CLOUD_HEALTHCARE` | Enable cloud healthcare APIs | `false` |
| `VITE_CLOUD_HEALTHCARE_FALLBACK` | Enable fallback to custom ML | `true` |
| `VITE_ENABLE_CONSENSUS_ANALYSIS` | Enable consensus analysis | `false` |
| `VITE_PRIMARY_CLOUD_PROVIDER` | Primary provider | `google` |
| `VITE_CLOUD_HEALTHCARE_TIMEOUT` | API timeout (ms) | `30000` |
| `VITE_CLOUD_HEALTHCARE_MAX_RETRIES` | Max retry attempts | `3` |
| `
