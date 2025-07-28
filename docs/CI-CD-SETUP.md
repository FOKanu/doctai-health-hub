# 🚀 CI/CD Pipeline Setup Guide

This document provides comprehensive instructions for setting up and configuring the CI/CD pipeline for DoctAI Health Hub.

## 📋 Table of Contents

1. [Overview](#overview)
2. [Testing Framework](#testing-framework)
3. [GitHub Actions Workflows](#github-actions-workflows)
4. [Deployment Platforms](#deployment-platforms)
5. [Environment Variables](#environment-variables)
6. [Security Considerations](#security-considerations)
7. [Troubleshooting](#troubleshooting)

## 🎯 Overview

The CI/CD pipeline includes:

- **Automated Testing**: Unit tests, E2E tests, and code quality checks
- **Build Verification**: Automated build and artifact management
- **Security Scanning**: Vulnerability assessment and security audits
- **Multi-Platform Deployment**: Support for Vercel, Netlify, AWS, and GCP
- **Bundle Analysis**: Performance monitoring and optimization

## 🧪 Testing Framework

### Unit Testing (Jest + React Testing Library)

```bash
# Run all tests
npm test

# Run tests in watch mode
npm run test:watch

# Run tests with coverage
npm run test:coverage

# Run specific test file
npm test -- --testPathPattern=App.test.tsx
```

### E2E Testing (Playwright)

```bash
# Run all E2E tests
npm run test:e2e

# Run E2E tests with UI
npm run test:e2e:ui

# Run E2E tests in debug mode
npm run test:e2e:debug

# Run specific E2E test
npx playwright test home.spec.ts
```

### Code Quality Checks

```bash
# Run linting
npm run lint

# Run type checking
npm run type-check

# Run all CI checks
npm run ci
```

## 🔄 GitHub Actions Workflows

### Main CI Pipeline (`ci.yml`)

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

**Setup:**
1. Create a Vercel account
2. Install Vercel CLI: `npm i -g vercel`
3. Link your project: `vercel link`
4. Get the required IDs from Vercel dashboard

#### Netlify Deployment (`deploy-netlify.yml`)

**Required Secrets:**
- `NETLIFY_AUTH_TOKEN`: Netlify API token
- `NETLIFY_SITE_ID`: Netlify site ID

**Setup:**
1. Create a Netlify account
2. Get API token from user settings
3. Get site ID from site settings

#### AWS S3 + CloudFront (`deploy-aws.yml`)

**Required Secrets:**
- `AWS_ACCESS_KEY_ID`: AWS access key
- `AWS_SECRET_ACCESS_KEY`: AWS secret key
- `AWS_REGION`: AWS region (e.g., us-east-1)
- `AWS_S3_BUCKET`: S3 bucket name
- `AWS_CLOUDFRONT_DISTRIBUTION_ID`: CloudFront distribution ID
- `AWS_CLOUDFRONT_DOMAIN`: CloudFront domain

**Setup:**
1. Create S3 bucket for static hosting
2. Configure CloudFront distribution
3. Set up IAM user with appropriate permissions

#### Google Cloud Platform (`deploy-gcp.yml`)

**Required Secrets:**
- `GCP_SA_KEY`: Base64-encoded service account key
- `GCP_PROJECT_ID`: GCP project ID
- `GCP_REGION`: GCP region (e.g., us-central1)
- `GCP_USE_STORAGE`: Set to 'true' for Cloud Storage deployment
- `GCP_STORAGE_BUCKET`: Cloud Storage bucket name

**Setup:**
1. Create GCP project
2. Enable required APIs (Cloud Run, Cloud Build, Cloud Storage)
3. Create service account with appropriate roles
4. Download and encode service account key

## 🌐 Deployment Platforms

### Vercel (Recommended)

**Pros:**
- Zero configuration
- Automatic HTTPS
- Global CDN
- Preview deployments
- Built-in analytics

**Setup:**
```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel --prod
```

### Netlify

**Pros:**
- Free tier available
- Form handling
- Serverless functions
- Branch deployments

**Setup:**
```bash
# Install Netlify CLI
npm i -g netlify-cli

# Deploy
netlify deploy --prod
```

### AWS S3 + CloudFront

**Pros:**
- Highly scalable
- Global distribution
- Cost-effective
- Full control

**Setup:**
```bash
# Configure AWS CLI
aws configure

# Deploy to S3
aws s3 sync dist/ s3://your-bucket-name --delete

# Invalidate CloudFront
aws cloudfront create-invalidation --distribution-id YOUR_DISTRIBUTION_ID --paths "/*"
```

### Google Cloud Platform

**Pros:**
- Integration with other GCP services
- Cloud Run for serverless
- Cloud Storage for static hosting
- Global load balancing

**Setup:**
```bash
# Install gcloud CLI
gcloud init

# Deploy to Cloud Run
gcloud run deploy doctai-app --source .
```

## 🔐 Environment Variables

### Required for All Deployments

```bash
# Application
NODE_ENV=production
VITE_APP_ENV=production

# Google Maps API
VITE_GOOGLE_MAPS_API_KEY=your-google-maps-api-key

# Supabase
VITE_SUPABASE_URL=your-supabase-url
VITE_SUPABASE_ANON_KEY=your-supabase-anon-key

# OpenAI
VITE_OPENAI_API_KEY=your-openai-api-key

# Azure Health Bot
VITE_ENABLE_AZURE_HEALTH_BOT=true
VITE_AZURE_HEALTH_BOT_ENDPOINT=your-azure-endpoint
VITE_AZURE_HEALTH_BOT_API_KEY=your-azure-api-key
```

### Platform-Specific Variables

#### Vercel
```bash
VERCEL_TOKEN=your-vercel-token
VERCEL_ORG_ID=your-org-id
VERCEL_PROJECT_ID=your-project-id
```

#### Netlify
```bash
NETLIFY_AUTH_TOKEN=your-netlify-token
NETLIFY_SITE_ID=your-site-id
```

#### AWS
```bash
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_REGION=us-east-1
AWS_S3_BUCKET=your-bucket-name
AWS_CLOUDFRONT_DISTRIBUTION_ID=your-distribution-id
AWS_CLOUDFRONT_DOMAIN=your-domain.com
```

#### GCP
```bash
GCP_SA_KEY=base64-encoded-service-account-key
GCP_PROJECT_ID=your-project-id
GCP_REGION=us-central1
GCP_USE_STORAGE=true
GCP_STORAGE_BUCKET=your-bucket-name
```

## 🔒 Security Considerations

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

## 📞 Support

For issues with the CI/CD pipeline:

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
