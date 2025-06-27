# 🚀 Google Cloud Console Setup Guide for DoctAI Health Hub

This comprehensive guide will walk you through setting up Google Cloud Console and configuring all necessary services for your DoctAI Health Hub application.

## 📋 Table of Contents

1. [Google Cloud Console Setup](#google-cloud-console-setup)
2. [Project Creation & Configuration](#project-creation--configuration)
3. [API Enablement](#api-enablement)
4. [Service Account Setup](#service-account-setup)
5. [Healthcare API Configuration](#healthcare-api-configuration)
6. [Authentication & Security](#authentication--security)
7. [Environment Configuration](#environment-configuration)
8. [Testing & Validation](#testing--validation)
9. [Cost Management](#cost-management)
10. [Troubleshooting](#troubleshooting)

## 🏗️ Google Cloud Console Setup

### Step 1: Access Google Cloud Console

1. **Visit Google Cloud Console**
   - Go to [console.cloud.google.com](https://console.cloud.google.com)
   - Sign in with your Google account
   - If you don't have an account, create one

2. **Accept Terms of Service**
   - Read and accept the Google Cloud Platform Terms of Service
   - Complete the account verification process

### Step 2: Billing Setup

1. **Enable Billing**
   - Click on "Billing" in the left navigation
   - Click "Link a billing account"
   - Add a payment method (credit card required)
   - Set up billing alerts (recommended)

2. **Set Budget Alerts**
   - Go to Billing > Budgets & alerts
   - Create a new budget
   - Set monthly budget limit (e.g., $50)
   - Configure email alerts at 50%, 80%, and 100%

## 🎯 Project Creation & Configuration

### Step 1: Create a New Project

1. **Create Project**
   - Click on the project dropdown at the top
   - Click "New Project"
   - Enter project name: `doctai-health-hub`
   - Enter project ID: `doctai-health-hub-[YOUR-UNIQUE-ID]`
   - Click "Create"

2. **Set Project as Default**
   - Select your new project from the dropdown
   - This will be your active project for all operations

### Step 2: Configure Project Settings

1. **Project Information**
   - Go to IAM & Admin > Settings
   - Note your Project ID and Project Number
   - These will be needed for API configuration

2. **Enable APIs**
   - Go to APIs & Services > Library
   - Search for and enable the following APIs:
     - **Cloud Healthcare API**
     - **Cloud Vision API**
     - **Cloud Storage API**
     - **Cloud Functions API**
     - **Cloud Run API**

## 🔌 API Enablement

### Step 1: Enable Required APIs

```bash
# Enable Cloud Healthcare API
gcloud services enable healthcare.googleapis.com

# Enable Cloud Vision API
gcloud services enable vision.googleapis.com

# Enable Cloud Storage API
gcloud services enable storage.googleapis.com

# Enable Cloud Functions API
gcloud services enable cloudfunctions.googleapis.com

# Enable Cloud Run API
gcloud services enable run.googleapis.com
```

### Step 2: Verify API Status

1. **Check API Status**
   - Go to APIs & Services > Dashboard
   - Verify all APIs show "Enabled" status
   - Note any API quotas or limits

2. **API Quotas**
   - Go to APIs & Services > Quotas
   - Review default quotas for each API
   - Request quota increases if needed

## 🔐 Service Account Setup

### Step 1: Create Service Account

1. **Navigate to IAM**
   - Go to IAM & Admin > Service Accounts
   - Click "Create Service Account"

2. **Configure Service Account**
   - Name: `doctai-healthcare-service`
   - Description: `Service account for DoctAI Health Hub healthcare operations`
   - Click "Create and Continue"

### Step 2: Assign Roles

1. **Healthcare Roles**
   - Click "Select a role"
   - Search for and add these roles:
     - **Healthcare Dataset Admin**
     - **Healthcare DICOM Store Admin**
     - **Healthcare FHIR Store Admin**
     - **Healthcare HL7 V2 Store Admin**

2. **Additional Roles**
   - **Cloud Storage Admin** (for image storage)
   - **Cloud Vision API User** (for image analysis)
   - **Cloud Functions Developer** (for serverless functions)

### Step 3: Create and Download Key

1. **Create Key**
   - Click on your service account
   - Go to "Keys" tab
   - Click "Add Key" > "Create new key"
   - Choose "JSON" format
   - Click "Create"

2. **Secure the Key**
   - Download the JSON key file
   - Store it securely (never commit to git)
   - Add to your `.env` file as `GOOGLE_APPLICATION_CREDENTIALS`

## 🏥 Healthcare API Configuration

### Step 1: Create Healthcare Dataset

1. **Navigate to Healthcare API**
   - Go to APIs & Services > Healthcare API
   - Click "Create Dataset"

2. **Configure Dataset**
   - Dataset ID: `doctai-healthcare-dataset`
   - Location: `us-central1` (or your preferred region)
   - Click "Create"

### Step 2: Create DICOM Stores

1. **Create DICOM Store for X-rays**
   - Go to your dataset
   - Click "Create DICOM Store"
   - Store ID: `chest-xray-store`
   - Click "Create"

2. **Create DICOM Store for CT Scans**
   - Store ID: `ct-scan-store`
   - Click "Create"

3. **Create DICOM Store for MRIs**
   - Store ID: `mri-store`
   - Click "Create"

4. **Create DICOM Store for Skin Lesions**
   - Store ID: `skin-lesion-store`
   - Click "Create"

### Step 3: Configure FHIR Store (Optional)

1. **Create FHIR Store**
   - Go to your dataset
   - Click "Create FHIR Store"
   - Store ID: `patient-data-store`
   - FHIR Version: `R4`
   - Click "Create"

## 🔒 Authentication & Security

### Step 1: Configure Authentication

1. **Application Default Credentials**
   ```bash
   # Set up application default credentials
   gcloud auth application-default login
   ```

2. **Service Account Key**
   ```bash
   # Set the service account key path
   export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account-key.json"
   ```

### Step 2: Security Best Practices

1. **IAM Policies**
   - Go to IAM & Admin > IAM
   - Review and restrict access
   - Use principle of least privilege

2. **API Keys**
   - Go to APIs & Services > Credentials
   - Create API keys with restrictions
   - Set HTTP referrer restrictions
   - Set API restrictions

3. **VPC Configuration**
   - Consider setting up VPC for additional security
   - Configure firewall rules
   - Enable private Google access

## ⚙️ Environment Configuration

### Step 1: Update Your .env File

```bash
# Google Cloud Healthcare Configuration
VITE_USE_CLOUD_HEALTHCARE=true
VITE_ENABLE_GOOGLE_HEALTHCARE=true
VITE_PRIMARY_CLOUD_PROVIDER=google

# Google Cloud Project Details
VITE_GOOGLE_HEALTHCARE_PROJECT_ID=doctai-project
VITE_GOOGLE_HEALTHCARE_LOCATION=us-central1
VITE_GOOGLE_HEALTHCARE_DATASET_ID=doctai-healthcare-dataset

# Service Account Authentication
GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/service-account-key.json

# API Configuration
VITE_GOOGLE_CLOUD_VISION_API_KEY=your-vision-api-key
VITE_GOOGLE_CLOUD_STORAGE_BUCKET=doctai-health-hub-images

# Performance Settings
VITE_CLOUD_HEALTHCARE_TIMEOUT=30000
VITE_CLOUD_HEALTHCARE_MAX_RETRIES=3

# Debug Settings
VITE_CLOUD_HEALTHCARE_DEBUG=true
VITE_CLOUD_HEALTHCARE_LOG_REQUESTS=true
```

### Step 2: Create Cloud Storage Bucket

```bash
# Create storage bucket for images
gsutil mb -l us-central1 gs://doctai-health-hub-images

# Set bucket permissions
gsutil iam ch serviceAccount:doctai-healthcare-service@doctai-health-hub-[YOUR-UNIQUE-ID].iam.gserviceaccount.com:objectAdmin gs://doctai-health-hub-images
```

## 🧪 Testing & Validation

### Step 1: Test Healthcare API Connection

```typescript
// Test script to verify Google Cloud Healthcare setup
import { CloudHealthcareService } from './src/services/cloudHealthcare';

const testGoogleHealthcare = async () => {
  try {
    const config = {
      googleHealthcare: {
        projectId: process.env.VITE_GOOGLE_HEALTHCARE_PROJECT_ID,
        location: process.env.VITE_GOOGLE_HEALTHCARE_LOCATION,
        datasetId: process.env.VITE_GOOGLE_HEALTHCARE_DATASET_ID
      }
    };

    const service = new CloudHealthcareService(config);

    // Test dataset access
    const status = await service.getServiceStatus();
    console.log('Google Healthcare Status:', status);

    return true;
  } catch (error) {
    console.error('Google Healthcare Test Failed:', error);
    return false;
  }
};
```

### Step 2: Test Image Upload

```typescript
// Test image upload to Cloud Storage
import { Storage } from '@google-cloud/storage';

const testImageUpload = async (imageFile: File) => {
  try {
    const storage = new Storage();
    const bucket = storage.bucket(process.env.VITE_GOOGLE_CLOUD_STORAGE_BUCKET);

    const blob = bucket.file(`test-images/${Date.now()}-${imageFile.name}`);
    const blobStream = blob.createWriteStream();

    return new Promise((resolve, reject) => {
      blobStream.on('error', reject);
      blobStream.on('finish', () => {
        resolve(blob.publicUrl());
      });
      blobStream.end(imageFile);
    });
  } catch (error) {
    console.error('Image Upload Test Failed:', error);
    throw error;
  }
};
```

### Step 3: Validate Configuration

```typescript
// Configuration validation script
const validateGoogleCloudConfig = () => {
  const required = [
    'VITE_GOOGLE_HEALTHCARE_PROJECT_ID',
    'VITE_GOOGLE_HEALTHCARE_LOCATION',
    'VITE_GOOGLE_HEALTHCARE_DATASET_ID',
    'GOOGLE_APPLICATION_CREDENTIALS'
  ];

  const missing = required.filter(key => !process.env[key]);

  if (missing.length > 0) {
    console.error('Missing required environment variables:', missing);
    return false;
  }

  console.log('✅ Google Cloud configuration is valid');
  return true;
};
```

## 💰 Cost Management

### Step 1: Set Up Budget Alerts

1. **Create Budget**
   - Go to Billing > Budgets & alerts
   - Click "Create Budget"
   - Set monthly budget: $50
   - Configure alerts at 50%, 80%, 100%

2. **Monitor Usage**
   - Go to Billing > Reports
   - Set up custom reports
   - Monitor API usage costs

### Step 2: Cost Optimization

1. **API Quotas**
   - Set appropriate quotas for each API
   - Monitor usage patterns
   - Optimize API calls

2. **Storage Optimization**
   - Use appropriate storage classes
   - Set up lifecycle policies
   - Compress images when possible

## 🔍 Troubleshooting

### Common Issues

#### 1. "Permission Denied" Errors
**Cause**: Insufficient IAM permissions
**Solution**:
```bash
# Grant additional permissions
gcloud projects add-iam-policy-binding doctai-health-hub-[YOUR-UNIQUE-ID] \
  --member="serviceAccount:doctai-healthcare-service@doctai-health-hub-[YOUR-UNIQUE-ID].iam.gserviceaccount.com" \
  --role="roles/healthcare.datasetAdmin"
```

#### 2. "API Not Enabled" Errors
**Cause**: Required APIs not enabled
**Solution**:
```bash
# Enable required APIs
gcloud services enable healthcare.googleapis.com
gcloud services enable vision.googleapis.com
gcloud services enable storage.googleapis.com
```

#### 3. "Authentication Failed" Errors
**Cause**: Invalid service account credentials
**Solution**:
```bash
# Verify service account setup
gcloud auth application-default login
gcloud config set project doctai-health-hub-[YOUR-UNIQUE-ID]
```

#### 4. "Dataset Not Found" Errors
**Cause**: Incorrect dataset ID or location
**Solution**:
- Verify dataset ID in Google Cloud Console
- Check dataset location matches configuration
- Ensure dataset exists and is accessible

### Debug Commands

```bash
# Check project configuration
gcloud config list

# Verify APIs are enabled
gcloud services list --enabled

# Test authentication
gcloud auth list

# Check service account permissions
gcloud projects get-iam-policy doctai-health-hub-[YOUR-UNIQUE-ID]
```

## 📊 Monitoring & Logging

### Step 1: Enable Cloud Monitoring

1. **Set Up Monitoring**
   - Go to Monitoring > Overview
   - Create workspace for your project
   - Set up dashboards for API usage

2. **Configure Logging**
   - Go to Logging > Logs Explorer
   - Create log-based metrics
   - Set up alerts for errors

### Step 2: Performance Monitoring

```typescript
// Add monitoring to your healthcare service
const monitorHealthcareCall = async (operation: string, call: () => Promise<any>) => {
  const startTime = Date.now();

  try {
    const result = await call();
    const duration = Date.now() - startTime;

    // Log success metrics
    console.log(`Healthcare API ${operation} successful in ${duration}ms`);

    // Send to monitoring service
    // monitoringService.recordSuccess(operation, duration);

    return result;
  } catch (error) {
    const duration = Date.now() - startTime;

    // Log error metrics
    console.error(`Healthcare API ${operation} failed after ${duration}ms:`, error);

    // Send to monitoring service
    // monitoringService.recordError(operation, error, duration);

    throw error;
  }
};
```

## 🚀 Next Steps

### Additional Google Cloud Services

Consider integrating these additional Google Cloud services:

1. **Cloud Functions** - For serverless image processing
2. **Cloud Run** - For containerized healthcare applications
3. **BigQuery** - For healthcare data analytics
4. **Cloud AI Platform** - For custom ML model deployment
5. **Cloud Build** - For CI/CD pipeline automation

### Security Enhancements

1. **VPC Service Controls** - For additional security
2. **Cloud KMS** - For encryption key management
3. **Cloud Armor** - For DDoS protection
4. **Binary Authorization** - For container security

---

## 📞 Support Resources

- [Google Cloud Healthcare API Documentation](https://cloud.google.com/healthcare/docs)
- [Google Cloud Console](https://console.cloud.google.com)
- [Google Cloud Support](https://cloud.google.com/support)
- [Healthcare API Best Practices](https://cloud.google.com/healthcare/docs/best-practices)

---

**Happy configuring! 🎉**
