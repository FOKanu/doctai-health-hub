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
VITE_GOOGLE_HEALTHCARE_PROJECT_ID=your-project-id
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
| `VITE_CLOUD_HEALTHCARE_DEBUG` | Enable debug mode | `false` |

### Provider-Specific Configuration

#### Google Cloud Healthcare
- Requires Google Cloud project with Healthcare API enabled
- Supports DICOM and medical imaging analysis
- HIPAA compliant

#### Azure Health Bot
- Specialized in symptom assessment and triage
- Multi-language support
- Integration with existing healthcare systems

#### IBM Watson Health
- Advanced clinical insights and literature analysis
- Treatment recommendations
- Evidence-based medicine

## 🎛️ Feature Flags

### Consensus Analysis

Enable consensus analysis to get results from multiple providers:

```bash
VITE_ENABLE_CONSENSUS_ANALYSIS=true
```

This will:
- Run analysis on all available providers
- Calculate consensus prediction
- Provide agreement percentage
- Merge findings and recommendations

### Fallback System

The fallback system ensures your app continues working even if cloud APIs fail:

1. **Primary**: Cloud healthcare APIs
2. **Fallback**: Custom ML models
3. **Legacy**: Original prediction service

### Debug Mode

Enable debug mode for detailed logging:

```bash
VITE_CLOUD_HEALTHCARE_DEBUG=true
VITE_CLOUD_HEALTHCARE_LOG_REQUESTS=true
```

## 📊 Monitoring

### CloudHealthcareStatus Component

Use the `CloudHealthcareStatus` component to monitor your cloud healthcare setup:

```typescript
import { CloudHealthcareStatus } from '@/components/CloudHealthcareStatus';

// In your component
<CloudHealthcareStatus />
```

This component shows:
- Overall status
- Available providers
- Configuration validation
- Feature flags
- Performance settings

### Performance Metrics

The system tracks:
- Processing time per provider
- Success/failure rates
- API response times
- Consensus agreement levels

## 🔒 Security & Compliance

### Data Privacy
- All API calls use secure HTTPS
- API keys are stored in environment variables
- No sensitive data is logged in debug mode

### HIPAA Compliance
- Google Cloud Healthcare is HIPAA compliant
- Azure Health Bot supports HIPAA compliance
- Watson Health has healthcare compliance features

### Best Practices
1. Use environment variables for API keys
2. Enable debug mode only in development
3. Monitor API usage and costs
4. Implement rate limiting if needed

## 🚨 Troubleshooting

### Common Issues

#### "No cloud healthcare providers available"
- Check if `VITE_USE_CLOUD_HEALTHCARE=true`
- Verify at least one provider is configured
- Check API keys and endpoints

#### "Cloud healthcare API failed"
- Verify API keys are correct
- Check network connectivity
- Review API quotas and limits
- Enable fallback mode

#### "Configuration validation failed"
- Check required environment variables
- Verify provider enablement flags
- Ensure primary provider is available

### Debug Steps

1. Enable debug mode:
```bash
VITE_CLOUD_HEALTHCARE_DEBUG=true
```

2. Check browser console for detailed logs

3. Use the CloudHealthcareStatus component to verify configuration

4. Test individual providers:
```typescript
const status = getCloudHealthcareStatus();
console.log('Available providers:', status.providers);
```

## 📈 Performance Optimization

### Timeout Settings
```bash
VITE_CLOUD_HEALTHCARE_TIMEOUT=30000  # 30 seconds
VITE_CLOUD_HEALTHCARE_MAX_RETRIES=3
```

### Provider Selection
- Use fastest provider as primary
- Enable consensus only when needed
- Monitor response times per provider

### Caching
- Consider implementing result caching
- Cache clinical insights and recommendations
- Store provider performance metrics

## 🔄 Migration Guide

### From Custom ML Only

1. Set up cloud healthcare APIs
2. Enable fallback mode
3. Test with small batch of images
4. Monitor performance and accuracy
5. Gradually increase cloud API usage

### From Legacy Prediction Service

1. Update to use `analyzeImage()` instead of `analyzePrediction()`
2. Add image type parameter
3. Handle new metadata fields
4. Update UI to show provider information

## 📚 Examples

### Basic Image Analysis
```typescript
const result = await analyzeImage(imageUri, 'skin_lesion');
if (result.metadata.provider === 'google') {
  console.log('Used Google Cloud Healthcare');
} else if (result.metadata.provider === 'custom_ml') {
  console.log('Fell back to custom ML');
}
```

### Symptom Assessment
```typescript
const symptoms = ['fever', 'cough', 'fatigue'];
const assessment = await assessSymptoms(symptoms, { age: 25 });
if (assessment.urgency === 'emergency') {
  // Show emergency alert
}
```

### Clinical Insights
```typescript
const insights = await getClinicalInsights({
  age: 50,
  gender: 'female',
  medicalHistory: ['hypertension'],
  currentSymptoms: ['chest pain']
});
console.log('Risk factors:', insights.riskFactors);
```

## 🤝 Contributing

To add new cloud healthcare providers:

1. Create provider service class
2. Implement required interfaces
3. Add configuration options
4. Update main service
5. Add tests and documentation

## 📞 Support

For issues with cloud healthcare APIs:
- Check provider documentation
- Review API quotas and limits
- Enable debug mode for detailed logs
- Use fallback mode for critical functionality
