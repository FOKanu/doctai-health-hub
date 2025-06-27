# ⚡ Quick Google Cloud Setup for DoctAI Health Hub

This is a quick setup guide to get your Google Cloud Console and services configured for DoctAI Health Hub.

## 🚀 Option 1: Automated Setup (Recommended)

### Prerequisites
1. **Install Google Cloud CLI**
   ```bash
   # macOS (using Homebrew)
   brew install google-cloud-sdk

   # Or download from: https://cloud.google.com/sdk/docs/install
   ```

2. **Authenticate with Google Cloud**
   ```bash
   gcloud auth login
   gcloud auth application-default login
   ```

### Run the Setup Script
```bash
# Make sure you're in the project root
cd /Users/francis/code/doctai-health-hub

# Run the automated setup script
./scripts/setup-google-cloud.sh
```

The script will:
- ✅ Create a Google Cloud project
- ✅ Enable required APIs
- ✅ Create service account with proper permissions
- ✅ Set up healthcare dataset and DICOM stores
- ✅ Create storage bucket
- ✅ Update your `.env` file
- ✅ Create test scripts

## 🛠️ Option 2: Manual Setup

### Step 1: Google Cloud Console Setup

1. **Visit Google Cloud Console**
   - Go to [console.cloud.google.com](https://console.cloud.google.com)
   - Sign in with your Google account

2. **Create a New Project**
   - Click on the project dropdown at the top
   - Click "New Project"
   - Name: `DoctAI Health Hub`
   - Project ID: `doctai-health-hub-[YOUR-UNIQUE-ID]`
   - Click "Create"

3. **Enable Billing**
   - Go to Billing in the left navigation
   - Link a billing account
   - Add payment method (required)

### Step 2: Enable Required APIs

Go to **APIs & Services > Library** and enable:

- ✅ **Cloud Healthcare API**
- ✅ **Cloud Vision API**
- ✅ **Cloud Storage API**
- ✅ **Cloud Functions API**
- ✅ **Cloud Run API**

### Step 3: Create Service Account

1. **Go to IAM & Admin > Service Accounts**
2. **Click "Create Service Account"**
3. **Configure:**
   - Name: `doctai-healthcare-service`
   - Description: `Service account for DoctAI Health Hub healthcare operations`

4. **Assign Roles:**
   - Healthcare Dataset Admin
   - Healthcare DICOM Store Admin
   - Cloud Storage Admin
   - Cloud Vision API User

5. **Create and Download Key:**
   - Go to Keys tab
   - Click "Add Key" > "Create new key"
   - Choose JSON format
   - Download the key file

### Step 4: Create Healthcare Dataset

1. **Go to APIs & Services > Healthcare API**
2. **Click "Create Dataset"**
3. **Configure:**
   - Dataset ID: `doctai-healthcare-dataset`
   - Location: `us-central1`

### Step 5: Create DICOM Stores

In your dataset, create these DICOM stores:
- `chest-xray-store`
- `ct-scan-store`
- `mri-store`
- `skin-lesion-store`

### Step 6: Create Storage Bucket

```bash
# Create storage bucket
gsutil mb -l us-central1 gs://doctai-health-hub-images

# Set permissions
gsutil iam ch serviceAccount:doctai-healthcare-service@[YOUR-PROJECT-ID].iam.gserviceaccount.com:objectAdmin gs://doctai-health-hub-images
```

### Step 7: Update Environment Variables

Add these to your `.env` file:

```bash
# Google Cloud Healthcare Configuration
VITE_USE_CLOUD_HEALTHCARE=true
VITE_ENABLE_GOOGLE_HEALTHCARE=true
VITE_PRIMARY_CLOUD_PROVIDER=google

# Google Cloud Project Details
VITE_GOOGLE_HEALTHCARE_PROJECT_ID=doctai-project
VITE_GOOGLE_HEALTHCARE_LOCATION=us-central1
VITE_GOOGLE_HEALTHCARE_DATASET_ID=doctai-healthcare-dataset
VITE_GOOGLE_CLOUD_STORAGE_BUCKET=doctai-health-hub-images

# Service Account Authentication
GOOGLE_APPLICATION_CREDENTIALS=./doctai-healthcare-service-key.json

# Performance Settings
VITE_CLOUD_HEALTHCARE_TIMEOUT=30000
VITE_CLOUD_HEALTHCARE_MAX_RETRIES=3

# Debug Settings
VITE_CLOUD_HEALTHCARE_DEBUG=true
VITE_CLOUD_HEALTHCARE_LOG_REQUESTS=true
```

## 🧪 Testing Your Setup

### Test Google Cloud Connection

```bash
# Test the setup
node scripts/test-google-cloud.js
```

### Test from Your App

1. **Start your development server**
   ```bash
   npm run dev
   ```

2. **Check the CloudHealthcareStatus component**
   - Navigate to your app
   - Look for the cloud healthcare status indicator
   - Should show "Google Cloud Healthcare: Connected"

3. **Test image analysis**
   - Upload a medical image
   - Check if it uses Google Cloud Healthcare API
   - Verify results in the console

## 🔧 Troubleshooting

### Common Issues

#### "Permission Denied"
```bash
# Grant additional permissions
gcloud projects add-iam-policy-binding [YOUR-PROJECT-ID] \
  --member="serviceAccount:doctai-healthcare-service@[YOUR-PROJECT-ID].iam.gserviceaccount.com" \
  --role="roles/healthcare.datasetAdmin"
```

#### "API Not Enabled"
```bash
# Enable required APIs
gcloud services enable healthcare.googleapis.com
gcloud services enable vision.googleapis.com
gcloud services enable storage.googleapis.com
```

#### "Authentication Failed"
```bash
# Verify authentication
gcloud auth application-default login
gcloud config set project [YOUR-PROJECT-ID]
```

### Debug Commands

```bash
# Check project configuration
gcloud config list

# Verify APIs are enabled
gcloud services list --enabled

# Test authentication
gcloud auth list

# Check service account permissions
gcloud projects get-iam-policy [YOUR-PROJECT-ID]
```

## 💰 Cost Management

### Set Up Budget Alerts

1. **Go to Billing > Budgets & alerts**
2. **Create Budget:**
   - Monthly budget: $50
   - Alerts at 50%, 80%, 100%

### Monitor Usage

- Go to **Billing > Reports**
- Set up custom reports
- Monitor API usage costs

## 🔒 Security Best Practices

1. **Never commit service account keys to git**
2. **Use principle of least privilege for IAM roles**
3. **Set up API key restrictions**
4. **Consider VPC for additional security**
5. **Enable audit logging**

## 📚 Additional Resources

- [Detailed Setup Guide](GOOGLE_CLOUD_SETUP_GUIDE.md)
- [Google Cloud Healthcare Documentation](https://cloud.google.com/healthcare/docs)
- [Cloud Healthcare Best Practices](https://cloud.google.com/healthcare/docs/best-practices)

---

## 🎯 Next Steps

After setup, you can:

1. **Test the integration** with sample medical images
2. **Configure additional providers** (Azure, Watson)
3. **Set up monitoring and logging**
4. **Implement cost optimization strategies**
5. **Add security enhancements**

---

**Need help?** Check the detailed setup guide or run the automated script for a hassle-free setup! 🚀
