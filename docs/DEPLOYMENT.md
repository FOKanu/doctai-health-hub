# 🚀 Deployment Guide

## Overview

This guide covers comprehensive deployment strategies for the DoctAI Health Hub application across multiple platforms and environments. Choose the deployment option that best fits your infrastructure and requirements.

## 📋 Table of Contents

- [🏗️ Architecture](#️-architecture)
- [☁️ Cloud Platforms](#-cloud-platforms)
- [🔧 Environment Setup](#-environment-setup)
- [📦 Build Process](#-build-process)
- [🚀 Deployment Options](#-deployment-options)
- [🔄 CI/CD Pipelines](#-cicd-pipelines)
- [🔐 Security Configuration](#-security-configuration)
- [📊 Monitoring & Logging](#-monitoring--logging)
- [🛠️ Troubleshooting](#-troubleshooting)

---

## 🏗️ Architecture

### Production Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   CDN/Edge      │    │   Load Balancer │    │   Application   │
│   (CloudFront)  │◄──►│   (ALB/NLB)     │◄──►│   (Cloud Run)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Static Assets │    │   API Gateway   │    │   Database      │
│   (S3/Storage)  │    │   (API Gateway) │    │   (Supabase)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Environment Strategy

- **Development**: Local development with hot reload
- **Staging**: Pre-production testing environment
- **Production**: Live application with full monitoring

---

## ☁️ Cloud Platforms

### 1. Vercel (Recommended for Frontend)

**Pros:**
- Zero-config deployment
- Automatic HTTPS
- Global CDN
- Git integration
- Preview deployments

**Cons:**
- Limited backend capabilities
- Vendor lock-in

### 2. AWS (Enterprise)

**Pros:**
- Full control over infrastructure
- Scalable and reliable
- Cost-effective for large scale
- Multiple services integration

**Cons:**
- Complex setup
- Requires DevOps expertise

### 3. Google Cloud Platform

**Pros:**
- Healthcare-focused services
- Strong AI/ML integration
- HIPAA compliance features
- Global infrastructure

**Cons:**
- Learning curve
- Cost for small projects

### 4. Netlify

**Pros:**
- Simple deployment
- Good for static sites
- Built-in forms
- Good free tier

**Cons:**
- Limited backend features
- Less control over infrastructure

---

## 🔧 Environment Setup

### Required Environment Variables

```bash
# =============================================================================
# CORE APPLICATION SETTINGS
# =============================================================================
NODE_ENV=production
VITE_APP_ENV=production
VITE_APP_NAME=DoctAI Health Hub
VITE_APP_VERSION=1.0.0
VITE_APP_URL=https://your-domain.com

# =============================================================================
# API CONFIGURATION
# =============================================================================
VITE_API_BASE_URL=https://api.your-domain.com
VITE_USE_NEW_PREDICTION_API=true
VITE_DEBUG_PREDICTIONS=false
VITE_ML_API_ENDPOINT=https://api.your-domain.com/api/predict

# =============================================================================
# CLOUD HEALTHCARE APIs
# =============================================================================
VITE_USE_CLOUD_HEALTHCARE=true
VITE_CLOUD_HEALTHCARE_FALLBACK=true
VITE_ENABLE_CONSENSUS_ANALYSIS=true
VITE_PRIMARY_CLOUD_PROVIDER=google

# Google Cloud Healthcare
VITE_ENABLE_GOOGLE_HEALTHCARE=true
VITE_GOOGLE_HEALTHCARE_PROJECT_ID=your-project-id
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
# DATABASE & AUTHENTICATION
# =============================================================================
VITE_SUPABASE_URL=your-supabase-url
VITE_SUPABASE_ANON_KEY=your-supabase-anon-key
VITE_SUPABASE_SERVICE_ROLE_KEY=your-supabase-service-role-key

# =============================================================================
# SECURITY & COMPLIANCE
# =============================================================================
HIPAA_ENCRYPTION_KEY=your-secure-encryption-key
HIPAA_KEY_ROTATION_DAYS=90
HIPAA_AUDIT_RETENTION_DAYS=7300
HIPAA_SESSION_TIMEOUT_MINUTES=480
HIPAA_MAX_FAILED_LOGINS=5
HIPAA_REQUIRE_MFA=true

# =============================================================================
# PERFORMANCE SETTINGS
# =============================================================================
VITE_CLOUD_HEALTHCARE_TIMEOUT=30000
VITE_CLOUD_HEALTHCARE_MAX_RETRIES=3
VITE_CLOUD_HEALTHCARE_DEBUG=false
```

---

## 📦 Build Process

### Production Build

```bash
# Build for production
npm run build

# Preview production build
npm run preview

# Analyze bundle size
npm run build:analyze

# Build with different environments
npm run build:staging
npm run build:production
```

### Build Optimization

```bash
# Enable build optimizations
VITE_BUILD_OPTIMIZE=true
VITE_BUILD_MINIFY=true
VITE_BUILD_SOURCEMAP=false

# Bundle analysis
npm run build:analyze
```

---

## 🚀 Deployment Options

### 1. Vercel (Recommended)

**Setup:**
```bash
# Install Vercel CLI
npm i -g vercel

# Login to Vercel
vercel login

# Deploy to Vercel
vercel --prod

# Or link existing project
vercel link
```

**Required Secrets:**
- `VERCEL_TOKEN`: Vercel API token
- `VERCEL_ORG_ID`: Vercel organization ID
- `VERCEL_PROJECT_ID`: Vercel project ID

**Pros:**
- Zero configuration
- Automatic HTTPS
- Global CDN
- Preview deployments
- Built-in analytics

### 2. Netlify

**Setup:**
```bash
# Install Netlify CLI
npm i -g netlify-cli

# Login to Netlify
netlify login

# Deploy to Netlify
netlify deploy --prod --dir=dist
```

**Required Secrets:**
- `NETLIFY_AUTH_TOKEN`: Netlify API token
- `NETLIFY_SITE_ID`: Netlify site ID

**Pros:**
- Free tier available
- Form handling
- Serverless functions
- Branch deployments

### 3. AWS S3 + CloudFront

**Setup:**
```bash
# Configure AWS CLI
aws configure

# Create S3 bucket
aws s3 mb s3://your-bucket-name

# Enable static website hosting
aws s3 website s3://your-bucket-name --index-document index.html --error-document index.html

# Deploy to S3
aws s3 sync dist/ s3://your-bucket-name --delete

# Invalidate CloudFront cache
aws cloudfront create-invalidation --distribution-id YOUR_DISTRIBUTION_ID --paths "/*"
```

**Required Secrets:**
- `AWS_ACCESS_KEY_ID`: AWS access key
- `AWS_SECRET_ACCESS_KEY`: AWS secret key
- `AWS_REGION`: AWS region (e.g., us-east-1)
- `AWS_S3_BUCKET`: S3 bucket name
- `AWS_CLOUDFRONT_DISTRIBUTION_ID`: CloudFront distribution ID
- `AWS_CLOUDFRONT_DOMAIN`: CloudFront domain

**Pros:**
- Highly scalable
- Global distribution
- Cost-effective
- Full control

### 4. Google Cloud Platform

**Setup:**
```bash
# Install gcloud CLI
gcloud init

# Deploy to Cloud Run
gcloud run deploy doctai-app --source .

# Or deploy to Cloud Storage
gsutil -m rsync -r -d dist/ gs://your-bucket-name/
```

**Required Secrets:**
- `GCP_SA_KEY`: Base64-encoded service account key
- `GCP_PROJECT_ID`: GCP project ID
- `GCP_REGION`: GCP region (e.g., us-central1)
- `GCP_USE_STORAGE`: Set to 'true' for Cloud Storage deployment
- `GCP_STORAGE_BUCKET`: Cloud Storage bucket name

**Pros:**
- Integration with other GCP services
- Cloud Run for serverless
- Cloud Storage for static hosting
- Global load balancing

---

## 🔄 CI/CD Pipelines

### GitHub Actions Workflows

#### Main CI Pipeline (`ci.yml`)

Triggers on:
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop` branches

Jobs:
1. **Test & Quality Checks**
   - Linting with ESLint
   - Type checking with TypeScript
   - Unit tests with Jest
   - Coverage reporting

2. **Build Verification**
   - Application build
   - Artifact upload

3. **E2E Tests**
   - Playwright tests across multiple browsers
   - Test result upload

4. **Security Scan**
   - npm audit
   - Snyk security scanning

5. **Bundle Analysis**
   - Build size analysis
   - Performance metrics

### Deployment Workflows

#### Vercel Deployment (`deploy-vercel.yml`)

**Required Secrets:**
- `VERCEL_TOKEN`: Vercel API token
- `VERCEL_ORG_ID`: Vercel organization ID
- `VERCEL_PROJECT_ID`: Vercel project ID

#### Netlify Deployment (`deploy-netlify.yml`)

**Required Secrets:**
- `NETLIFY_AUTH_TOKEN`: Netlify API token
- `NETLIFY_SITE_ID`: Netlify site ID

#### AWS S3 + CloudFront (`deploy-aws.yml`)

**Required Secrets:**
- `AWS_ACCESS_KEY_ID`: AWS access key
- `AWS_SECRET_ACCESS_KEY`: AWS secret key
- `AWS_REGION`: AWS region
- `AWS_S3_BUCKET`: S3 bucket name
- `AWS_CLOUDFRONT_DISTRIBUTION_ID`: CloudFront distribution ID
- `AWS_CLOUDFRONT_DOMAIN`: CloudFront domain

#### Google Cloud Platform (`deploy-gcp.yml`)

**Required Secrets:**
- `GCP_SA_KEY`: Base64-encoded service account key
- `GCP_PROJECT_ID`: GCP project ID
- `GCP_REGION`: GCP region
- `GCP_USE_STORAGE`: Set to 'true' for Cloud Storage deployment
- `GCP_STORAGE_BUCKET`: Cloud Storage bucket name

---

## 🔐 Security Configuration

### Secrets Management

1. **Never commit secrets to version control**
2. **Use GitHub Secrets for sensitive data**
3. **Rotate secrets regularly**
4. **Use least privilege principle**

### Security Scanning

The pipeline includes:
- **npm audit**: Dependency vulnerability scanning
- **Snyk**: Advanced security scanning
- **CodeQL**: GitHub's semantic code analysis

### Best Practices

1. **Environment Separation**
   - Use different environments for staging and production
   - Implement proper access controls

2. **Monitoring**
   - Set up alerts for failed deployments
   - Monitor application performance
   - Track security vulnerabilities

3. **Backup Strategy**
   - Regular database backups
   - Configuration backups
   - Disaster recovery plan

---

## 📊 Monitoring & Logging

### Application Monitoring

```bash
# Enable monitoring
VITE_ENABLE_MONITORING=true
VITE_SENTRY_DSN=your-sentry-dsn
VITE_GOOGLE_ANALYTICS_ID=your-ga-id
```

### Logging Configuration

```bash
# Logging levels
VITE_LOG_LEVEL=info
VITE_ENABLE_DEBUG_LOGGING=false
VITE_ENABLE_PERFORMANCE_LOGGING=true
```

### Performance Monitoring

- **Core Web Vitals**: Monitor LCP, FID, CLS
- **Error Tracking**: Sentry integration
- **Analytics**: Google Analytics
- **Uptime Monitoring**: Status page integration

---

## 🛠️ Troubleshooting

### Common Issues

#### Build Failures
```bash
# Clear cache and reinstall dependencies
rm -rf node_modules package-lock.json
npm install

# Check for TypeScript errors
npm run type-check

# Verify build locally
npm run build
```

#### Test Failures
```bash
# Clear Jest cache
npm test -- --clearCache

# Run tests with verbose output
npm test -- --verbose

# Check test coverage
npm run test:coverage
```

#### Deployment Issues

**Vercel:**
```bash
# Check Vercel CLI status
vercel whoami

# Verify project linking
vercel ls
```

**Netlify:**
```bash
# Check Netlify CLI status
netlify status

# Verify site configuration
netlify sites:list
```

**AWS:**
```bash
# Verify AWS credentials
aws sts get-caller-identity

# Check S3 bucket access
aws s3 ls s3://your-bucket-name
```

**GCP:**
```bash
# Verify gcloud authentication
gcloud auth list

# Check project configuration
gcloud config get-value project
```

### Performance Optimization

1. **Bundle Analysis**
   - Monitor bundle size
   - Identify large dependencies
   - Optimize imports

2. **Caching Strategy**
   - Implement proper cache headers
   - Use CDN effectively
   - Optimize static assets

3. **Monitoring**
   - Set up performance monitoring
   - Track Core Web Vitals
   - Monitor error rates

### Getting Required Tokens

#### Snyk Token:
1. Go to [snyk.io](https://snyk.io)
2. Sign up/login
3. Go to Account Settings > API Token
4. Copy the token

#### Codecov Token:
1. Go to [codecov.io](https://codecov.io)
2. Sign up/login with GitHub
3. Go to Settings > Repository Upload Token
4. Copy the token

#### Vercel Tokens:
1. Go to [vercel.com](https://vercel.com)
2. Sign up/login
3. Go to Settings > Tokens
4. Create new token
5. Get Project ID and Org ID from project settings

#### AWS Credentials:
1. Go to AWS Console
2. Create IAM user with S3 and CloudFront permissions
3. Generate access keys
4. Create S3 bucket and CloudFront distribution

### Quick Fix Checklist

- [ ] Add SNYK_TOKEN to GitHub secrets
- [ ] Add CODECOV_TOKEN to GitHub secrets
- [ ] Add AWS credentials (if using AWS deployment)
- [ ] Add Vercel tokens (if using Vercel deployment)
- [ ] Add Netlify tokens (if using Netlify deployment)
- [ ] Add GCP credentials (if using GCP deployment)
- [ ] Run tests locally to verify they pass
- [ ] Check that build works locally

---

## 📞 Support

For issues with deployment:

1. Check the GitHub Actions logs
2. Verify environment variables
3. Test locally first
4. Review security configurations
5. Consult platform-specific documentation

## 🔄 Continuous Improvement

Regular tasks:
- Update dependencies
- Review security scans
- Optimize build times
- Monitor performance metrics
- Update documentation
