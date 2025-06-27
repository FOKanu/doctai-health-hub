// Test script for Google Cloud Healthcare setup
import { CloudHealthcareService } from '../src/services/cloudHealthcare/index.js';
import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

// Load environment variables from .env file
function loadEnvFile() {
  try {
    const __filename = fileURLToPath(import.meta.url);
    const __dirname = dirname(__filename);
    const envPath = join(__dirname, '..', '.env');
    const envContent = readFileSync(envPath, 'utf8');

    const envVars = {};
    envContent.split('\n').forEach(line => {
      const trimmedLine = line.trim();
      if (trimmedLine && !trimmedLine.startsWith('#')) {
        const [key, ...valueParts] = trimmedLine.split('=');
        if (key && valueParts.length > 0) {
          envVars[key] = valueParts.join('=');
        }
      }
    });

    return envVars;
  } catch (error) {
    console.error('Error loading .env file:', error.message);
    return {};
  }
}

async function testGoogleHealthcare() {
    try {
        console.log('🧪 Testing Google Cloud Healthcare setup...');

        // Load environment variables
        const envVars = loadEnvFile();

        const config = {
            googleHealthcare: {
                projectId: 'doctai-project',
                location: envVars.VITE_GOOGLE_HEALTHCARE_LOCATION,
                datasetId: envVars.VITE_GOOGLE_HEALTHCARE_DATASET_ID
            }
        };

        console.log('📋 Configuration loaded:');
        console.log('  Project ID:', config.googleHealthcare.projectId);
        console.log('  Location:', config.googleHealthcare.location);
        console.log('  Dataset ID:', config.googleHealthcare.datasetId);

        const service = new CloudHealthcareService(config);

        // Test service status
        const status = await service.getServiceStatus();
        console.log('\n✅ Google Healthcare Status:', status);

        // Test connection
        const connectionTest = await service.testGoogleHealthcareConnection();
        console.log('\n🔗 Connection Test:', connectionTest);

        // Get available providers
        const providers = service.getAvailableProviders();
        console.log('\n📡 Available Providers:', providers);

        if (status.configured && connectionTest.success) {
            console.log('\n🎉 Google Cloud Healthcare setup is working correctly!');
            return true;
        } else {
            console.log('\n❌ Google Cloud Healthcare setup has issues:');
            if (status.errors.length > 0) {
                console.log('  Errors:', status.errors);
            }
            return false;
        }
    } catch (error) {
        console.error('❌ Google Healthcare Test Failed:', error);
        return false;
    }
}

// Run test if called directly
if (import.meta.url === `file://${process.argv[1]}`) {
    testGoogleHealthcare();
}

export { testGoogleHealthcare };
