# 🛠️ Troubleshooting Guide

## Overview

This comprehensive troubleshooting guide covers common issues, solutions, and best practices for the DoctAI Health Hub application. It includes development, deployment, and production issues.

## 📋 Table of Contents

- [🚨 Critical Issues](#-critical-issues)
- [🔧 Development Issues](#-development-issues)
- [🚀 Deployment Issues](#-deployment-issues)
- [🧪 Testing Issues](#-testing-issues)
- [🔐 Security Issues](#-security-issues)
- [📊 Performance Issues](#-performance-issues)
- [🐛 Common Errors](#-common-errors)
- [📞 Support Resources](#-support-resources)

---

## 🚨 Critical Issues

### 1. Syntax Error in RiskAssessmentsScreen.tsx

**Problem**: Malformed case statement causing compilation failure

**Solution**:
```typescript
// ❌ Before (Broken)
case 'case 'risk':
  const ':
{
  const riskOrder = { high: 3, medium: 2, low: 1 };

// ✅ After (Fixed)
case 'risk': {
  const riskOrder = { high: 3, medium: 2, low: 1 };
  return riskOrder[b.riskLevel] - riskOrder[a.riskLevel];
}
```

**Impact**: This was blocking the entire application from running
**Status**: ✅ **RESOLVED**

### 2. TypeScript Type Safety Improvements

**Problem**: 166 TypeScript errors affecting code quality

**Fixes Applied**:
- Replaced `any` types with proper TypeScript types
- Fixed empty interfaces
- Corrected import statements
- Added proper type definitions

**Results**:
- **Before**: 166 errors/warnings
- **After**: 122 errors/warnings
- **Improvement**: 26% reduction

### 3. Development Server Issues

**Problem**: Development server failing to start

**Solution**:
```bash
# Clear cache and reinstall dependencies
rm -rf node_modules package-lock.json
npm install

# Check for TypeScript errors
npm run type-check

# Verify build locally
npm run build
```

---

## 🔧 Development Issues

### Build Failures

**Problem**: Build failing due to dependencies or configuration

**Solution**:
```bash
# Clear cache and reinstall dependencies
rm -rf node_modules package-lock.json
npm install

# Check for TypeScript errors
npm run type-check

# Verify build locally
npm run build

# Check for linting errors
npm run lint

# Fix linting errors automatically
npm run lint:fix
```

### TypeScript Errors

**Common Issues and Solutions**:

1. **`@typescript-eslint/no-explicit-any`**
   ```typescript
   // ❌ Before
   const data: any = response.data;

   // ✅ After
   interface ApiResponse {
     data: PatientData;
   }
   const data: ApiResponse = response.data;
   ```

2. **`prefer-const`**
   ```typescript
   // ❌ Before
   let patient = { name: 'John' };

   // ✅ After
   const patient = { name: 'John' };
   ```

3. **`no-case-declarations`**
   ```typescript
   // ❌ Before
   case 'risk':
     const riskOrder = { high: 3, medium: 2, low: 1 };
     return riskOrder[b.riskLevel] - riskOrder[a.riskLevel];

   // ✅ After
   case 'risk': {
     const riskOrder = { high: 3, medium: 2, low: 1 };
     return riskOrder[b.riskLevel] - riskOrder[a.riskLevel];
   }
   ```

### Import/Export Issues

**Problem**: Module resolution errors

**Solution**:
```typescript
// ❌ Before
import { Button } from './components/ui/button';

// ✅ After
import { Button } from '@/components/ui/button';
```

### Environment Variables

**Problem**: Missing or incorrect environment variables

**Solution**:
```bash
# Check if .env file exists
ls -la .env

# Create .env file from example
cp .env.example .env

# Verify required variables
cat .env | grep VITE_
```

---

## 🚀 Deployment Issues

### Missing Secrets (Most Common)

**Problem**: Deployments failing due to missing GitHub secrets

**Solution**: Add the following secrets to your GitHub repository:

#### Required for All Deployments:
```bash
# Go to: Settings > Secrets and variables > Actions
# Add these secrets:

# Security Scanning
SNYK_TOKEN=your-snyk-token
CODECOV_TOKEN=your-codecov-token

# AWS Deployment
AWS_ACCESS_KEY_ID=your-aws-access-key
AWS_SECRET_ACCESS_KEY=your-aws-secret-key
AWS_REGION=us-east-1
AWS_S3_BUCKET=your-s3-bucket-name
AWS_CLOUDFRONT_DISTRIBUTION_ID=your-cloudfront-distribution-id
AWS_CLOUDFRONT_DOMAIN=your-cloudfront-domain.com

# Vercel Deployment
VERCEL_TOKEN=your-vercel-token
VERCEL_ORG_ID=your-vercel-org-id
VERCEL_PROJECT_ID=your-vercel-project-id

# Netlify Deployment
NETLIFY_AUTH_TOKEN=your-netlify-token
NETLIFY_SITE_ID=your-netlify-site-id

# GCP Deployment
GCP_SA_KEY=base64-encoded-service-account-key
GCP_PROJECT_ID=your-gcp-project-id
GCP_REGION=us-central1
GCP_STORAGE_BUCKET=your-gcp-bucket-name
```

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

### Platform-Specific Issues

#### Vercel Deployment Issues
```bash
# Check Vercel CLI status
vercel whoami

# Verify project linking
vercel ls

# Check deployment logs
vercel logs
```

#### Netlify Deployment Issues
```bash
# Check Netlify CLI status
netlify status

# Verify site configuration
netlify sites:list

# Check deployment logs
netlify logs
```

#### AWS Deployment Issues
```bash
# Verify AWS credentials
aws sts get-caller-identity

# Check S3 bucket access
aws s3 ls s3://your-bucket-name

# Test CloudFront invalidation
aws cloudfront create-invalidation --distribution-id YOUR_DISTRIBUTION_ID --paths "/*"
```

#### GCP Deployment Issues
```bash
# Verify gcloud authentication
gcloud auth list

# Check project configuration
gcloud config get-value project

# Test Cloud Storage access
gsutil ls gs://your-bucket-name
```

---

## 🧪 Testing Issues

### Test Failures

**Problem**: Tests timing out or failing

**Solution**:
```bash
# Clear Jest cache
npm test -- --clearCache

# Run tests with verbose output
npm test -- --verbose

# Check test coverage
npm run test:coverage

# Run specific test file
npm test -- --testPathPattern=App.test.tsx
```

### E2E Test Issues

**Problem**: Playwright tests failing

**Solution**:
```bash
# Install Playwright browsers
npx playwright install

# Run E2E tests with UI
npm run test:e2e:ui

# Run E2E tests in debug mode
npm run test:e2e:debug

# Run specific E2E test
npx playwright test home.spec.ts
```

### Coverage Issues

**Problem**: Low test coverage

**Solution**:
```bash
# Generate coverage report
npm run test:coverage

# Check coverage thresholds
# Update jest.config.js if needed
```

---

## 🔐 Security Issues

### Authentication Problems

**Problem**: Users unable to log in

**Solution**:
```bash
# Check Supabase configuration
VITE_SUPABASE_URL=your-supabase-url
VITE_SUPABASE_ANON_KEY=your-supabase-anon-key

# Verify Supabase is running
npx supabase status
```

### HIPAA Compliance Issues

**Problem**: Compliance dashboard showing errors

**Solution**:
```bash
# Check compliance environment variables
HIPAA_ENCRYPTION_KEY=your-secure-encryption-key
HIPAA_KEY_ROTATION_DAYS=90
HIPAA_AUDIT_RETENTION_DAYS=7300

# Verify audit logging is enabled
# Check database schema for compliance tables
```

### API Security Issues

**Problem**: API requests being blocked

**Solution**:
```bash
# Check rate limiting configuration
# Verify API keys are valid
# Check CORS settings
```

---

## 📊 Performance Issues

### Bundle Size Issues

**Problem**: Large bundle size affecting load times

**Solution**:
```bash
# Analyze bundle size
npm run build:analyze

# Check for large dependencies
npm ls --depth=0

# Optimize imports
# Use dynamic imports for large components
```

### Memory Leaks

**Problem**: Application memory usage increasing over time

**Solution**:
```typescript
// Use React.memo for expensive components
const ExpensiveComponent = React.memo(({ data }) => {
  return <div>{/* Component content */}</div>;
});

// Use useMemo for expensive calculations
const expensiveValue = useMemo(() => {
  return calculateExpensiveValue(data);
}, [data]);

// Use useCallback for event handlers
const handleClick = useCallback((id) => {
  handleItemClick(id);
}, [handleItemClick]);
```

### Slow API Responses

**Problem**: API calls taking too long

**Solution**:
```typescript
// Implement caching
const { data, loading } = useQuery(['patient', patientId],
  () => fetchPatient(patientId),
  { staleTime: 5 * 60 * 1000 } // 5 minutes
);

// Use optimistic updates
const mutation = useMutation(updatePatient, {
  onMutate: async (newPatient) => {
    // Optimistically update cache
  }
});
```

---

## 🐛 Common Errors

### Error: Cannot find module

**Problem**: Module resolution errors

**Solution**:
```bash
# Check tsconfig.json paths
{
  "compilerOptions": {
    "baseUrl": ".",
    "paths": {
      "@/*": ["src/*"]
    }
  }
}

# Verify file exists
ls -la src/components/ui/button.tsx

# Clear module cache
rm -rf node_modules/.cache
```

### Error: React Hook dependencies

**Problem**: Missing dependencies in useEffect

**Solution**:
```typescript
// ❌ Before
useEffect(() => {
  fetchData(patientId);
}, []); // Missing dependency

// ✅ After
useEffect(() => {
  fetchData(patientId);
}, [patientId]); // Include all dependencies
```

### Error: Type 'any' is not assignable

**Problem**: TypeScript strict mode errors

**Solution**:
```typescript
// ❌ Before
const data: any = response.data;

// ✅ After
interface ApiResponse {
  data: PatientData;
}
const data: ApiResponse = response.data;
```

### Error: Maximum call stack size exceeded

**Problem**: Infinite recursion

**Solution**:
```typescript
// ❌ Before
const Component = () => {
  const [data, setData] = useState(fetchData());
  return <div>{data}</div>;
};

// ✅ After
const Component = () => {
  const [data, setData] = useState(null);

  useEffect(() => {
    const loadData = async () => {
      const result = await fetchData();
      setData(result);
    };
    loadData();
  }, []);

  return <div>{data}</div>;
};
```

---

## 📞 Support Resources

### Getting Help

1. **Check the logs**: Look at the specific error messages
2. **Test locally**: Run the same commands on your machine
3. **Verify configuration**: Ensure all required settings are correct
4. **Check documentation**: Review the relevant documentation
5. **Search issues**: Look for similar issues in GitHub

### Support Channels

- **📖 Documentation**: [Full Documentation](docs/)
- **🐛 Issues**: [GitHub Issues](https://github.com/FOKanu/doctai-health-hub/issues)
- **💬 Discussions**: [GitHub Discussions](https://github.com/FOKanu/doctai-health-hub/discussions)
- **📧 Email**: support@doctai.com
- **Discord**: [Join our community](https://discord.gg/doctai)

### Quick Fix Checklist

- [ ] Add SNYK_TOKEN to GitHub secrets
- [ ] Add CODECOV_TOKEN to GitHub secrets
- [ ] Add AWS credentials (if using AWS deployment)
- [ ] Add Vercel tokens (if using Vercel deployment)
- [ ] Add Netlify tokens (if using Netlify deployment)
- [ ] Add GCP credentials (if using GCP deployment)
- [ ] Run tests locally to verify they pass
- [ ] Check that build works locally
- [ ] Verify environment variables are set
- [ ] Check for TypeScript errors
- [ ] Run linting and fix issues
- [ ] Test the application locally

### Debugging Steps

1. **Check GitHub Actions Logs**
   - Go to your repository on GitHub
   - Click "Actions" tab
   - Click on the failed workflow
   - Check the specific job that failed
   - Look for error messages

2. **Test Locally**
   ```bash
   # Run the same commands locally
   npm ci
   npm run lint
   npm run type-check
   npm test
   npm run build
   ```

3. **Check Secrets**
   ```bash
   # Verify secrets are set in GitHub
   # Go to: Settings > Secrets and variables > Actions
   # Ensure all required secrets are present
   ```

4. **Verify Dependencies**
   ```bash
   # Check if all packages are installed
   npm ls --depth=0

   # Update dependencies if needed
   npm update
   ```

---

## 🔄 Continuous Improvement

### Regular Maintenance Tasks

- **Update dependencies**: Keep packages up to date
- **Review security scans**: Monitor for vulnerabilities
- **Optimize build times**: Improve CI/CD performance
- **Monitor performance metrics**: Track application performance
- **Update documentation**: Keep docs current

### Performance Monitoring

- **Core Web Vitals**: Monitor LCP, FID, CLS
- **Error Tracking**: Use Sentry for error monitoring
- **Analytics**: Track user behavior and performance
- **Uptime Monitoring**: Ensure application availability

### Security Best Practices

- **Regular security audits**: Conduct periodic assessments
- **Dependency scanning**: Monitor for vulnerabilities
- **Access control reviews**: Verify permissions regularly
- **Compliance monitoring**: Ensure HIPAA compliance

---

**Note**: This troubleshooting guide covers the most common issues. For specific problems not covered here, please check the GitHub issues or contact support.
