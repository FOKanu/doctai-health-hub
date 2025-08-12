// Cloud Healthcare Configuration
export const CLOUD_HEALTHCARE_CONFIG = {
  // Feature flags
  ENABLE_CLOUD_HEALTHCARE: import.meta.env.VITE_USE_CLOUD_HEALTHCARE === 'true',
  ENABLE_FALLBACK: import.meta.env.VITE_CLOUD_HEALTHCARE_FALLBACK === 'true',
  ENABLE_CONSENSUS: import.meta.env.VITE_ENABLE_CONSENSUS_ANALYSIS === 'true',

  // Provider-specific flags
  ENABLE_GOOGLE: import.meta.env.VITE_ENABLE_GOOGLE_HEALTHCARE === 'true',
  ENABLE_AZURE: import.meta.env.VITE_ENABLE_AZURE_HEALTH_BOT === 'true',
  ENABLE_WATSON: import.meta.env.VITE_ENABLE_WATSON_HEALTH === 'true',

  // Primary provider selection
  PRIMARY_PROVIDER: import.meta.env.VITE_PRIMARY_CLOUD_PROVIDER || 'google',

  // Google Cloud Healthcare
  GOOGLE: {
    PROJECT_ID: import.meta.env.VITE_GOOGLE_HEALTHCARE_PROJECT_ID,
    LOCATION: import.meta.env.VITE_GOOGLE_HEALTHCARE_LOCATION || 'us-central1',
    DATASET_ID: import.meta.env.VITE_GOOGLE_HEALTHCARE_DATASET_ID,
    STORAGE_BUCKET: import.meta.env.VITE_GOOGLE_CLOUD_STORAGE_BUCKET,
    API_KEY: import.meta.env.VITE_GOOGLE_HEALTHCARE_API_KEY,
    ENABLED: import.meta.env.VITE_ENABLE_GOOGLE_HEALTHCARE === 'true' &&
             !!import.meta.env.VITE_GOOGLE_HEALTHCARE_PROJECT_ID
  },

  // Azure Health Bot
  AZURE: {
    ENDPOINT: import.meta.env.VITE_AZURE_HEALTH_BOT_ENDPOINT,
    API_KEY: import.meta.env.VITE_AZURE_HEALTH_BOT_API_KEY,
    ENABLED: import.meta.env.VITE_ENABLE_AZURE_HEALTH_BOT === 'true' &&
             !!import.meta.env.VITE_AZURE_HEALTH_BOT_ENDPOINT
  },

  // IBM Watson Health
  WATSON: {
    API_KEY: import.meta.env.VITE_WATSON_HEALTH_API_KEY,
    ENDPOINT: import.meta.env.VITE_WATSON_HEALTH_ENDPOINT,
    VERSION: import.meta.env.VITE_WATSON_HEALTH_VERSION || '2023-01-01',
    ENABLED: import.meta.env.VITE_ENABLE_WATSON_HEALTH === 'true' &&
             !!import.meta.env.VITE_WATSON_HEALTH_API_KEY
  },

  // Performance settings
  TIMEOUT_MS: parseInt(import.meta.env.VITE_CLOUD_HEALTHCARE_TIMEOUT || '30000'),
  MAX_RETRIES: parseInt(import.meta.env.VITE_CLOUD_HEALTHCARE_MAX_RETRIES || '3'),

  // Debug settings
  DEBUG_MODE: import.meta.env.VITE_CLOUD_HEALTHCARE_DEBUG === 'true',
  LOG_REQUESTS: import.meta.env.VITE_CLOUD_HEALTHCARE_LOG_REQUESTS === 'true'
} as const;

// Helper functions
export const isCloudHealthcareEnabled = () => {
  return CLOUD_HEALTHCARE_CONFIG.ENABLE_CLOUD_HEALTHCARE &&
         (CLOUD_HEALTHCARE_CONFIG.GOOGLE.ENABLED ||
          CLOUD_HEALTHCARE_CONFIG.AZURE.ENABLED ||
          CLOUD_HEALTHCARE_CONFIG.WATSON.ENABLED);
};

export const getAvailableProviders = () => {
  const providers: string[] = [];

  if (CLOUD_HEALTHCARE_CONFIG.GOOGLE.ENABLED) providers.push('google');
  if (CLOUD_HEALTHCARE_CONFIG.AZURE.ENABLED) providers.push('azure');
  if (CLOUD_HEALTHCARE_CONFIG.WATSON.ENABLED) providers.push('watson');

  return providers;
};

export const getPrimaryProvider = () => {
  const available = getAvailableProviders();
  const primary = CLOUD_HEALTHCARE_CONFIG.PRIMARY_PROVIDER;

  // If primary provider is available, use it
  if (available.includes(primary)) {
    return primary;
  }

  // Otherwise, use the first available provider
  return available[0] || null;
};

// Configuration validation
export const validateCloudHealthcareConfig = () => {
  const errors: string[] = [];

  if (CLOUD_HEALTHCARE_CONFIG.ENABLE_CLOUD_HEALTHCARE) {
    if (!CLOUD_HEALTHCARE_CONFIG.GOOGLE.ENABLED &&
        !CLOUD_HEALTHCARE_CONFIG.AZURE.ENABLED &&
        !CLOUD_HEALTHCARE_CONFIG.WATSON.ENABLED) {
      errors.push('Cloud healthcare is enabled but no providers are configured');
    }

    if (CLOUD_HEALTHCARE_CONFIG.PRIMARY_PROVIDER &&
        !getAvailableProviders().includes(CLOUD_HEALTHCARE_CONFIG.PRIMARY_PROVIDER)) {
      errors.push(`Primary provider '${CLOUD_HEALTHCARE_CONFIG.PRIMARY_PROVIDER}' is not available`);
    }
  }

  return {
    isValid: errors.length === 0,
    errors
  };
};

// Environment variable template for documentation
export const ENV_VARS_TEMPLATE = `
# Cloud Healthcare API Configuration
# Set VITE_USE_CLOUD_HEALTHCARE=true to enable cloud healthcare APIs

# Feature Flags
VITE_USE_CLOUD_HEALTHCARE=false
VITE_CLOUD_HEALTHCARE_FALLBACK=true
VITE_ENABLE_CONSENSUS_ANALYSIS=false
VITE_PRIMARY_CLOUD_PROVIDER=google

# Provider Enablement
VITE_ENABLE_GOOGLE_HEALTHCARE=false
VITE_ENABLE_AZURE_HEALTH_BOT=false
VITE_ENABLE_WATSON_HEALTH=false

# Google Cloud Healthcare (set in .env, do not commit secrets)
VITE_GOOGLE_HEALTHCARE_PROJECT_ID=
VITE_GOOGLE_HEALTHCARE_LOCATION=us-central1
VITE_GOOGLE_HEALTHCARE_DATASET_ID=
VITE_GOOGLE_CLOUD_STORAGE_BUCKET=
VITE_GOOGLE_HEALTHCARE_API_KEY=

# Azure Health Bot
VITE_AZURE_HEALTH_BOT_ENDPOINT=https://your-bot.azurewebsites.net
VITE_AZURE_HEALTH_BOT_API_KEY=your-api-key

# IBM Watson Health
VITE_WATSON_HEALTH_API_KEY=your-api-key
VITE_WATSON_HEALTH_ENDPOINT=https://your-watson-instance.com
VITE_WATSON_HEALTH_VERSION=2023-01-01

# Performance Settings
VITE_CLOUD_HEALTHCARE_TIMEOUT=30000
VITE_CLOUD_HEALTHCARE_MAX_RETRIES=3

# Debug Settings
VITE_CLOUD_HEALTHCARE_DEBUG=false
VITE_CLOUD_HEALTHCARE_LOG_REQUESTS=false
`;
