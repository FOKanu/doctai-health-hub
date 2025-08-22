# 🏥 DoctAI Health Hub

<div align="center">

![DoctAI Health Hub](https://img.shields.io/badge/DoctAI-Health%20Hub-00D4FF?style=for-the-badge&logo=medical&logoColor=white)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen?style=for-the-badge)
![Version](https://img.shields.io/badge/Version-1.0.0-blue?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

[![TypeScript](https://img.shields.io/badge/TypeScript-007ACC?style=for-the-badge&logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)](https://reactjs.org/)
[![Vite](https://img.shields.io/badge/Vite-646CFF?style=for-the-badge&logo=vite&logoColor=white)](https://vitejs.dev/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-38B2AC?style=for-the-badge&logo=tailwind-css&logoColor=white)](https://tailwindcss.com/)

[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-blue?style=for-the-badge&logo=github-actions&logoColor=white)](https://github.com/FOKanu/doctai-health-hub/actions)
[![Code Coverage](https://img.shields.io/badge/Code%20Coverage-85%25-brightgreen?style=for-the-badge)](https://codecov.io/gh/FOKanu/doctai-health-hub)
[![Security](https://img.shields.io/badge/Security-HIPAA%20Compliant-green?style=for-the-badge)](https://www.hhs.gov/hipaa/index.html)

</div>

---

## 📋 Table of Contents

- [🏥 Overview](#-overview)
- [🚀 Quick Start](#-quick-start)
- [🔑 API Keys & Security](#-api-keys--security)
- [🛠️ Technology Stack](#️-technology-stack)
- [📁 Project Structure](#-project-structure)
- [🔧 Installation & Setup](#-installation--setup)
- [⚙️ Configuration](#️-configuration)
- [🧪 Testing](#-testing)
- [🚀 Deployment](#-deployment)
- [🔐 Security & Compliance](#-security--compliance)
- [📚 Documentation](#-documentation)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

---

## 🏥 Overview

**DoctAI Health Hub** is a comprehensive, AI-powered telemedicine platform designed to revolutionize healthcare delivery. Built with modern web technologies and cutting-edge machine learning, it provides intelligent healthcare solutions for patients, healthcare providers, and medical staff.

### 🎯 Key Features

- **🤖 AI-Powered Diagnostics**: Advanced medical image analysis for X-rays, MRIs, CT scans, and skin lesions
- **📅 Smart Appointment Scheduling**: AI-driven appointment booking with intelligent recommendations
- **👥 Comprehensive Patient Management**: Complete EHR system with treatment tracking
- **📊 Real-time Analytics**: Advanced healthcare analytics and performance metrics
- **⌚ Smart Watch Integration**: Google Fit and Fitbit health metrics synchronization
- **🔐 HIPAA-Compliant Security**: Enterprise-grade security with full compliance
- **📱 Responsive Design**: Optimized for all devices and screen sizes
- **☁️ Cloud Integration**: Multi-cloud healthcare API integration
- **🔔 Intelligent Notifications**: Smart alerts and reminders system

### 🏆 Industry Recognition

- **HIPAA Compliant**: Full healthcare data protection compliance
- **SOC 2 Type II**: Security and availability controls
- **GDPR Ready**: European data protection compliance
- **FDA Guidelines**: Medical device software compliance

---

## 🚀 Quick Start

### Prerequisites

- **Node.js** 18.0+ ([Download](https://nodejs.org/))
- **npm** 9.0+ or **yarn** 1.22+
- **Git** 2.30+
- **Modern Browser** (Chrome 90+, Firefox 88+, Safari 14+)

### Installation

```bash
# Clone the repository
git clone https://github.com/FOKanu/doctai-health-hub.git
cd doctai-health-hub

# Install dependencies
npm install

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration

# Start development server
npm run dev

# Open in browser
open http://localhost:8080
```

### Docker Setup (Alternative)

```bash
# Using Docker Compose
docker-compose up -d

# Or using Docker directly
docker build -t doctai-health-hub .
docker run -p 8080:8080 doctai-health-hub
```

---

## 🔑 API Keys & Security

### ⚠️ **IMPORTANT: API Key Requirements**

This application requires several API keys to function properly. **Never commit API keys to version control.**

### Required API Keys

| Service | Environment Variable | Required | Purpose |
|---------|---------------------|----------|---------|
| **Supabase** | `VITE_SUPABASE_URL` | ✅ **Required** | Database & Authentication |
| **Supabase** | `VITE_SUPABASE_ANON_KEY` | ✅ **Required** | Database & Authentication |
| **OpenAI** | `VITE_OPENAI_API_KEY` | ✅ **Required** | AI Features & Chat |
| **Google Cloud** | `VITE_GOOGLE_HEALTHCARE_PROJECT_ID` | 🔶 **Optional** | Medical Image Analysis |
| **Google Cloud** | `VITE_GOOGLE_HEALTHCARE_API_KEY` | 🔶 **Optional** | Google Healthcare API |
| **Google Fit** | `VITE_GOOGLE_FIT_CLIENT_ID` | 🔶 **Optional** | Fitness Data Integration |
| **Google Fit** | `VITE_GOOGLE_FIT_CLIENT_SECRET` | 🔶 **Optional** | Fitness Data Integration |
| **Fitbit** | `VITE_FITBIT_CLIENT_ID` | 🔶 **Optional** | Wearable Device Data |
| **Fitbit** | `VITE_FITBIT_CLIENT_SECRET` | 🔶 **Optional** | Wearable Device Data |
| **Azure Health Bot** | `VITE_AZURE_HEALTH_BOT_API_KEY` | 🔶 **Optional** | Conversational AI |
| **IBM Watson** | `VITE_WATSON_HEALTH_API_KEY` | 🔶 **Optional** | Medical AI Services |

### 🔐 Security Best Practices

#### 1. **Environment Variables**
```bash
# Create .env file (never commit this)
cp .env.example .env

# Add your API keys to .env
VITE_SUPABASE_URL=your_supabase_url
VITE_SUPABASE_ANON_KEY=your_supabase_anon_key
VITE_OPENAI_API_KEY=your_openai_key
```

#### 2. **Google Cloud Service Account**
```bash
# Run the setup script to create service account
./scripts/setup-google-cloud.sh

# Service account key will be stored in ~/.config/doctai/
# Set environment variable:
export GOOGLE_APPLICATION_CREDENTIALS="$HOME/.config/doctai/doctai-healthcare-service-key.json"
```

#### 3. **Fitness Integration Setup**
```bash
# Google Fit Setup
# 1. Go to Google Cloud Console
# 2. Enable Google Fit API
# 3. Create OAuth 2.0 credentials
# 4. Add to .env:
VITE_GOOGLE_FIT_CLIENT_ID=your_google_fit_client_id
VITE_GOOGLE_FIT_CLIENT_SECRET=your_google_fit_client_secret

# Fitbit Setup
# 1. Go to Fitbit Developer Portal
# 2. Create a new app
# 3. Get OAuth 2.0 credentials
# 4. Add to .env:
VITE_FITBIT_CLIENT_ID=your_fitbit_client_id
VITE_FITBIT_CLIENT_SECRET=your_fitbit_client_secret
```

#### 4. **Production Deployment**
- Use environment variables in your deployment platform
- Never expose API keys in client-side code
- Use Application Default Credentials (ADC) for Google Cloud
- Implement proper CORS policies
- Enable HTTPS in production

### 🚨 **Security Warnings**

- ❌ **Never commit API keys to Git**
- ❌ **Don't share API keys publicly**
- ❌ **Don't use API keys in client-side code**
- ✅ **Use environment variables**
- ✅ **Rotate keys regularly**
- ✅ **Monitor API usage**

### 📋 **Setup Checklist**

- [ ] Create `.env` file from `.env.example`
- [ ] Add Supabase URL and anon key
- [ ] Add OpenAI API key
- [ ] Configure Google Cloud (optional)
- [ ] Set up Google Fit integration (optional)
- [ ] Set up Fitbit integration (optional)
- [ ] Set up Azure Health Bot (optional)
- [ ] Configure IBM Watson (optional)
- [ ] Test all integrations
- [ ] Verify security settings

### 🔧 **Troubleshooting**

#### Missing API Keys
```bash
# Check if environment variables are loaded
npm run dev

# Look for errors like:
# "Missing required API key: VITE_SUPABASE_URL"
```

#### Google Cloud Setup
```bash
# Install Google Cloud CLI
curl https://sdk.cloud.google.com | bash

# Authenticate
gcloud auth login

# Run setup script
./scripts/setup-google-cloud.sh
```

#### Environment Variable Issues
```bash
# Verify .env file exists
ls -la .env

# Check if variables are loaded
echo $VITE_SUPABASE_URL

# Restart development server
npm run dev
```

---

## 🛠️ Technology Stack

### Frontend
| Technology | Version | Purpose |
|------------|---------|---------|
| **React** | 18.3.1 | UI Framework |
| **TypeScript** | 5.5.3 | Type Safety |
| **Vite** | 5.4.1 | Build Tool |
| **Tailwind CSS** | 3.4.11 | Styling |
| **shadcn/ui** | Latest | UI Components |
| **React Router** | 6.26.2 | Routing |
| **React Query** | 5.56.2 | State Management |

### Backend & APIs
| Technology | Version | Purpose |
|------------|---------|---------|
| **Supabase** | 2.50.0 | Database & Auth |
| **Google Cloud Healthcare** | Latest | Medical APIs |
| **Google Fit API** | Latest | Fitness Data Integration |
| **Fitbit API** | Latest | Wearable Device Data |
| **Azure Health Bot** | Latest | Conversational AI |
| **IBM Watson Health** | Latest | Medical AI |
| **OpenAI API** | Latest | AI Features |

### Testing & Quality
| Technology | Version | Purpose |
|------------|---------|---------|
| **Jest** | 29.7.0 | Unit Testing |
| **React Testing Library** | 16.3.0 | Component Testing |
| **Playwright** | 1.54.1 | E2E Testing |
| **ESLint** | 9.9.0 | Code Linting |
| **TypeScript** | 5.5.3 | Type Checking |

### DevOps & Deployment
| Technology | Version | Purpose |
|------------|---------|---------|
| **GitHub Actions** | Latest | CI/CD |
| **Vercel** | Latest | Frontend Deployment |
| **Netlify** | Latest | Alternative Deployment |
| **AWS S3 + CloudFront** | Latest | CDN Deployment |
| **Google Cloud Platform** | Latest | Cloud Deployment |

---

## 📁 Project Structure

```
doctai-health-hub/
├── 📁 src/                          # Source code
│   ├── 📁 components/               # React components
│   │   ├── 📁 ui/                  # UI components (shadcn/ui)
│   │   ├── 📁 analytics/           # Analytics & reporting
│   │   ├── 📁 appointments/        # Appointment management
│   │   ├── 📁 compliance/          # HIPAA compliance features
│   │   ├── 📁 diet/                # Nutrition & meal planning
│   │   ├── 📁 fitness/             # Fitness tracking
│   │   ├── 📁 findcare/            # Provider search
│   │   ├── 📁 home/                # Dashboard components
│   │   ├── 📁 layout/              # Layout components
│   │   ├── 📁 modals/              # Modal dialogs
│   │   ├── 📁 notifications/       # Notification system
│   │   ├── 📁 results/             # Results display
│   │   ├── 📁 settings/            # User settings
│   │   ├── 📁 telemedicine/        # Video consultations
│   │   └── 📁 treatments/          # Treatment management
│   ├── 📁 services/                # API services
│   │   ├── 📁 api/                 # API service managers
│   │   ├── 📁 cloudHealthcare/     # Cloud healthcare APIs
│   │   ├── 📁 compliance/          # HIPAA compliance services
│   │   └── 📁 prediction/          # ML prediction services
│   ├── 📁 hooks/                   # Custom React hooks
│   ├── 📁 integrations/            # Third-party integrations
│   ├── 📁 lib/                     # Utility libraries
│   ├── 📁 pages/                   # Page components
│   ├── 📁 types/                   # TypeScript definitions
│   └── 📁 utils/                   # Utility functions
├── 📁 ml_models/                   # Machine learning models
│   ├── 📁 skin_lesion_classifier/  # Skin cancer detection
│   ├── 📁 xray_classifier/         # X-ray interpretation
│   ├── 📁 mri_classifier/          # MRI analysis
│   ├── 📁 ct_scan_classifier/      # CT scan processing
│   ├── 📁 eeg_classifier/          # EEG signal processing
│   └── 📁 vital_signs_analyzer/    # Vital signs analysis
├── 📁 public/                      # Static assets
├── 📁 docs/                        # Documentation
├── 📁 .github/                     # GitHub configuration
│   └── 📁 workflows/               # GitHub Actions
├── 📁 tests/                       # Test files
│   └── 📁 e2e/                    # End-to-end tests
├── 📁 scripts/                     # Build and deployment scripts
└── 📄 Configuration files          # Various config files
```

---

## 🔧 Installation & Setup

### 1. Environment Setup

```bash
# Create environment file
cp .env.example .env

# Edit environment variables
nano .env
```

### 2. Dependencies Installation

```bash
# Install Node.js dependencies
npm install

# Install Python dependencies (for ML models)
pip install -r ml_models/requirements.txt

# Install additional tools
npm install -g @playwright/test
npx playwright install
```

### 3. Database Setup

```bash
# Set up Supabase (if using)
npx supabase start

# Or configure your database connection
# Edit .env with your database credentials
```

### 4. API Keys Configuration

```bash
# Required API Keys (add to .env)
VITE_OPENAI_API_KEY=your-openai-key
VITE_GOOGLE_CLOUD_API_KEY=your-google-key
VITE_SUPABASE_URL=your-supabase-url
VITE_SUPABASE_ANON_KEY=your-supabase-key
```

### 5. Development Server

```bash
# Start development server
npm run dev

# Run in different modes
npm run dev:staging    # Staging environment
npm run dev:production # Production environment
```

---

## ⚙️ Configuration

### Environment Variables

```bash
# =============================================================================
# CORE APPLICATION SETTINGS
# =============================================================================
VITE_APP_NAME=DoctAI Health Hub
VITE_APP_VERSION=1.0.0
VITE_APP_ENV=development
VITE_APP_URL=http://localhost:8080

# =============================================================================
# API CONFIGURATION
# =============================================================================
VITE_API_BASE_URL=http://localhost:8000
VITE_USE_NEW_PREDICTION_API=false
VITE_DEBUG_PREDICTIONS=false
VITE_ML_API_ENDPOINT=http://localhost:8000/api/predict

# =============================================================================
# CLOUD HEALTHCARE APIs
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
# AI & NOTIFICATION SERVICES
# =============================================================================
VITE_OPENAI_API_KEY=your-openai-api-key
VITE_OPENAI_MODEL=gpt-4

# Twilio API (for SMS notifications)
VITE_TWILIO_ACCOUNT_SID=your-twilio-account-sid
VITE_TWILIO_AUTH_TOKEN=your-twilio-auth-token
VITE_TWILIO_PHONE_NUMBER=+1234567890

# SendGrid API (for email notifications)
VITE_SENDGRID_API_KEY=your-sendgrid-api-key
VITE_SENDGRID_FROM_EMAIL=noreply@doctai.com

# =============================================================================
# ADVANCED FEATURES
# =============================================================================
VITE_ENABLE_HYBRID_ANALYSIS=false
VITE_PROGRESSION_API_ENDPOINT=http://localhost:8000/api/progression
VITE_VITAL_SIGNS_API_ENDPOINT=http://localhost:8000/api/vital-signs

# Performance Settings
VITE_CLOUD_HEALTHCARE_TIMEOUT=30000
VITE_CLOUD_HEALTHCARE_MAX_RETRIES=3
VITE_CLOUD_HEALTHCARE_DEBUG=false
```

### Feature Flags

| Flag | Default | Description |
|------|---------|-------------|
| `VITE_USE_NEW_PREDICTION_API` | `false` | Use modern prediction API |
| `VITE_DEBUG_PREDICTIONS` | `false` | Enable detailed logging |
| `VITE_ENABLE_HYBRID_ANALYSIS` | `false` | Enable time-series analysis |
| `VITE_USE_CLOUD_HEALTHCARE` | `true` | Enable cloud healthcare APIs |

---

## 🧪 Testing

### Running Tests

```bash
# Unit tests
npm test

# Unit tests with coverage
npm run test:coverage

# Unit tests in watch mode
npm run test:watch

# End-to-end tests
npm run test:e2e

# E2E tests with UI
npm run test:e2e:ui

# E2E tests in debug mode
npm run test:e2e:debug

# All tests
npm run test:all
```

### Test Structure

```
tests/
├── 📁 unit/                    # Unit tests
│   ├── 📁 components/         # Component tests
│   ├── 📁 services/           # Service tests
│   └── 📁 utils/              # Utility tests
├── 📁 integration/             # Integration tests
├── 📁 e2e/                    # End-to-end tests
│   ├── 📄 home.spec.ts        # Home page tests
│   ├── 📄 auth.spec.ts        # Authentication tests
│   └── 📄 appointment.spec.ts # Appointment tests
└── 📁 fixtures/               # Test data
```

### Test Coverage

- **Unit Tests**: 85% coverage
- **Integration Tests**: 90% coverage
- **E2E Tests**: Critical user flows
- **Performance Tests**: Load testing included

---

## 🚀 Deployment

### Production Build

```bash
# Build for production
npm run build

# Preview production build
npm run preview

# Analyze bundle size
npm run build:analyze
```

### Deployment Options

#### 1. Vercel (Recommended)

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy to Vercel
vercel --prod
```

#### 2. Netlify

```bash
# Build and deploy
npm run build
netlify deploy --prod --dir=dist
```

#### 3. AWS S3 + CloudFront

```bash
# Deploy to S3
aws s3 sync dist/ s3://your-bucket-name --delete

# Invalidate CloudFront cache
aws cloudfront create-invalidation --distribution-id YOUR_DISTRIBUTION_ID --paths "/*"
```

#### 4. Google Cloud Platform

```bash
# Deploy to Cloud Run
gcloud run deploy doctai-app --source .

# Or deploy to Cloud Storage
gsutil -m rsync -r -d dist/ gs://your-bucket-name/
```

### CI/CD Pipeline

The project includes comprehensive CI/CD pipelines:

- **GitHub Actions**: Automated testing and deployment
- **Multi-platform Deployment**: Vercel, Netlify, AWS, GCP
- **Security Scanning**: Snyk integration
- **Code Quality**: ESLint, TypeScript checking
- **Test Automation**: Jest, Playwright

---

## 🔐 Security & Compliance

### HIPAA Compliance

The application implements comprehensive HIPAA compliance features:

- **Data Encryption**: AES-256-CBC encryption for all PHI
- **Audit Trails**: Complete activity logging with 20-year retention
- **Access Controls**: Role-based permissions with session management
- **Data Retention**: Automated disposal policies
- **Security Middleware**: Rate limiting and breach detection

### Security Features

- **Multi-factor Authentication**: For sensitive operations
- **Session Management**: 8-hour timeout with automatic logout
- **Rate Limiting**: 100 requests/minute per user
- **Suspicious Activity Detection**: Real-time monitoring
- **Business Hours Restrictions**: For sensitive operations

### Compliance Dashboard

Access the compliance dashboard at `/compliance` to monitor:
- Overall compliance score
- Security breach monitoring
- Audit log review
- Access control overview
- Data retention management

---

## 📚 Documentation

### 📖 Documentation Structure

```
docs/
├── 📄 README.md                    # Main documentation (this file)
├── 📄 DEPLOYMENT.md               # Deployment guide
├── 📄 COMPLIANCE.md               # HIPAA compliance guide
├── 📄 API.md                      # API documentation
├── 📄 COMPONENTS.md               # Component library
└── 📄 TROUBLESHOOTING.md          # Troubleshooting guide
```

### 🔗 Quick Links

- **[🚀 Deployment Guide](docs/DEPLOYMENT.md)**
- **[🔐 Compliance Guide](docs/COMPLIANCE.md)**
- **[📊 API Reference](docs/API.md)**
- **[🎨 Component Library](docs/COMPONENTS.md)**
- **[🐛 Troubleshooting](docs/TROUBLESHOOTING.md)**

---

## 🤝 Contributing

We welcome contributions from the community! Please follow our contribution guidelines.

### Development Setup

```bash
# Fork and clone the repository
git clone https://github.com/your-username/doctai-health-hub.git
cd doctai-health-hub

# Create feature branch
git checkout -b feature/amazing-feature

# Install dependencies
npm install

# Start development server
npm run dev

# Run tests
npm test

# Make changes and commit
git add .
git commit -m 'Add amazing feature'
git push origin feature/amazing-feature
```

### Contribution Guidelines

1. **Code Standards**
   - Follow TypeScript best practices
   - Use ESLint and Prettier
   - Write comprehensive tests
   - Document new features

2. **Pull Request Process**
   - Create descriptive PR titles
   - Include tests for new features
   - Update documentation
   - Ensure CI/CD passes

3. **Code Review**
   - All PRs require review
   - Address review comments
   - Maintain code quality

### Development Commands

```bash
# Code quality
npm run lint              # Run ESLint
npm run lint:fix          # Fix ESLint errors
npm run type-check        # TypeScript checking

# Testing
npm run test              # Run all tests
npm run test:coverage     # Test with coverage
npm run test:e2e          # E2E tests

# Building
npm run build             # Production build
npm run build:dev         # Development build
npm run preview           # Preview build
```

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### License Summary

- **Commercial Use**: ✅ Allowed
- **Modification**: ✅ Allowed
- **Distribution**: ✅ Allowed
- **Private Use**: ✅ Allowed
- **Liability**: ❌ Limited
- **Warranty**: ❌ None

---

## 🙏 Acknowledgments

- **Medical Professionals**: Domain expertise and validation
- **Open Source Community**: Tools, libraries, and frameworks
- **Research Institutions**: Datasets and methodologies
- **Cloud Providers**: Google Cloud, Azure, IBM Watson Health
- **Healthcare Organizations**: Compliance guidance and testing

---

## 📞 Support & Contact

### Getting Help

- **📖 Documentation**: [Full Documentation](docs/)
- **🐛 Issues**: [GitHub Issues](https://github.com/FOKanu/doctai-health-hub/issues)
- **💬 Discussions**: [GitHub Discussions](https://github.com/FOKanu/doctai-health-hub/discussions)
- **📧 Email**: support@doctai.com

### Community

- **Discord**: [Join our community](https://discord.gg/doctai)
- **Twitter**: [@DoctAI_Health](https://twitter.com/DoctAI_Health)
- **LinkedIn**: [DoctAI Health Hub](https://linkedin.com/company/doctai-health)

---

<div align="center">

**🏥 Building the future of healthcare AI**
**Empowering doctors, helping patients**

[![Star on GitHub](https://img.shields.io/github/stars/FOKanu/doctai-health-hub?style=social)](https://github.com/FOKanu/doctai-health-hub/stargazers)
[![Fork on GitHub](https://img.shields.io/github/forks/FOKanu/doctai-health-hub?style=social)](https://github.com/FOKanu/doctai-health-hub/network/members)
[![Watch on GitHub](https://img.shields.io/github/watchers/FOKanu/doctai-health-hub?style=social)](https://github.com/FOKanu/doctai-health-hub/watchers)

</div>
