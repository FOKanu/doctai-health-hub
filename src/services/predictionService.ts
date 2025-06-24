import { modernPredictionService } from './prediction/modernPredictionService';
import { analyzePredictionLegacy } from './prediction/legacyPredictionService';
import { USE_NEW_PREDICTION_API, DEBUG_PREDICTIONS, PredictionResult } from './prediction/types';
import { CloudHealthcareService, CloudAnalysisResult, ImageType } from './cloudHealthcare';

// Re-export types for backward compatibility
export type { PredictionResult, ModernPredictionResult, ImageType } from './prediction/types';
export { savePredictionToSupabase } from './prediction/databaseService';
export { modernPredictionService as predictionService } from './prediction/modernPredictionService';

// Feature flags for cloud healthcare integration
const USE_CLOUD_HEALTHCARE = import.meta.env.VITE_USE_CLOUD_HEALTHCARE === 'true';
const CLOUD_HEALTHCARE_FALLBACK = import.meta.env.VITE_CLOUD_HEALTHCARE_FALLBACK === 'true';
const ENABLE_CONSENSUS_ANALYSIS = import.meta.env.VITE_ENABLE_CONSENSUS_ANALYSIS === 'true';

// Initialize cloud healthcare service if enabled
let cloudHealthcareService: CloudHealthcareService | null = null;

if (USE_CLOUD_HEALTHCARE) {
  const cloudConfig = {
    googleHealthcare: {
      projectId: import.meta.env.VITE_GOOGLE_HEALTHCARE_PROJECT_ID,
      location: import.meta.env.VITE_GOOGLE_HEALTHCARE_LOCATION || 'us-central1',
      datasetId: import.meta.env.VITE_GOOGLE_HEALTHCARE_DATASET_ID,
      apiKey: import.meta.env.VITE_GOOGLE_HEALTHCARE_API_KEY
    },
    azureHealthBot: {
      endpoint: import.meta.env.VITE_AZURE_HEALTH_BOT_ENDPOINT,
      apiKey: import.meta.env.VITE_AZURE_HEALTH_BOT_API_KEY
    },
    watsonHealth: {
      apiKey: import.meta.env.VITE_WATSON_HEALTH_API_KEY,
      endpoint: import.meta.env.VITE_WATSON_HEALTH_ENDPOINT,
      version: import.meta.env.VITE_WATSON_HEALTH_VERSION || '2023-01-01'
    }
  };

  // Only initialize if at least one provider is configured
  if (cloudConfig.googleHealthcare?.projectId ||
      cloudConfig.azureHealthBot?.endpoint ||
      cloudConfig.watsonHealth?.apiKey) {
    cloudHealthcareService = new CloudHealthcareService(cloudConfig);
  }
}

/**
 * Enhanced analyze function that can use cloud healthcare APIs
 * Falls back to custom ML models if cloud APIs are unavailable
 */
export const analyzeImage = async (imageUri: string, imageType: ImageType = 'skin_lesion'): Promise<PredictionResult> => {
  try {
    // Convert imageUri to File for cloud healthcare APIs
    const response = await fetch(imageUri);
    const blob = await response.blob();
    const file = new File([blob], 'image.jpg', { type: 'image/jpeg' });

    // Try cloud healthcare APIs first if enabled
    if (USE_CLOUD_HEALTHCARE && cloudHealthcareService) {
      console.log('🌐 Using cloud healthcare APIs');

      try {
        let cloudResult: CloudAnalysisResult;

        if (ENABLE_CONSENSUS_ANALYSIS) {
          // Get consensus from multiple providers
          const consensusResult = await cloudHealthcareService.getConsensusAnalysis(file, imageType);
          cloudResult = consensusResult.consensus;

          if (DEBUG_PREDICTIONS) {
            console.log('Consensus analysis:', {
              agreement: consensusResult.agreement,
              individualResults: consensusResult.individualResults.length,
              consensus: cloudResult
            });
          }
        } else {
          // Use single provider
          cloudResult = await cloudHealthcareService.analyzeImage(file, imageType);
        }

        // Convert cloud result to legacy format
        const benignProbability = cloudResult.prediction.toLowerCase().includes('benign')
          ? cloudResult.confidence
          : 1 - cloudResult.confidence;

        const malignantProbability = 1 - benignProbability;
        const prediction: 'benign' | 'malignant' = benignProbability > 0.5 ? 'benign' : 'malignant';

        return {
          prediction,
          confidence: cloudResult.confidence,
          probabilities: {
            benign: benignProbability,
            malignant: malignantProbability
          },
          timestamp: new Date().toISOString(),
          imageId: crypto.randomUUID(),
          // Add cloud-specific metadata
          metadata: {
            provider: cloudResult.provider,
            findings: cloudResult.findings,
            recommendations: cloudResult.recommendations,
            riskLevel: cloudResult.riskLevel,
            processingTime: cloudResult.metadata.processingTime
          }
        };
      } catch (cloudError) {
        console.warn('Cloud healthcare API failed:', cloudError);

        // If fallback is disabled, throw the error
        if (!CLOUD_HEALTHCARE_FALLBACK) {
          throw cloudError;
        }

        // Otherwise, fall back to custom ML models
        console.log('🔄 Falling back to custom ML models');
      }
    }

    // Use custom ML models (existing logic)
    if (USE_NEW_PREDICTION_API) {
      console.log('🚀 Using new prediction API (custom ML)');
      try {
        const modernResult = await modernPredictionService.analyzeImage(
          file,
          imageType,
          crypto.randomUUID()
        );

        const benignProbability = modernResult.classLabels.includes('Benign')
          ? modernResult.probabilities[modernResult.classLabels.indexOf('Benign')]
          : (modernResult.riskLevel === 'low' ? 0.8 : 0.2);

        const malignantProbability = 1 - benignProbability;
        const prediction: 'benign' | 'malignant' = benignProbability > 0.5 ? 'benign' : 'malignant';

        return {
          prediction,
          confidence: modernResult.confidence,
          probabilities: {
            benign: benignProbability,
            malignant: malignantProbability
          },
          timestamp: modernResult.createdAt,
          imageId: modernResult.imageId,
          metadata: {
            provider: 'custom_ml',
            modelVersion: modernResult.modelVersion,
            riskLevel: modernResult.riskLevel
          }
        };
      } catch (error) {
        console.warn('New API failed, falling back to legacy:', error);
        return analyzePredictionLegacy(imageUri);
      }
    } else {
      console.log('📱 Using legacy prediction API (custom ML)');
      return analyzePredictionLegacy(imageUri);
    }
  } catch (error) {
    console.error('All prediction methods failed:', error);
    throw new Error(`Image analysis failed: ${error.message}`);
  }
};

/**
 * @deprecated Use analyzeImage() instead. This function will be removed in a future version.
 * This is a compatibility adapter for the old API.
 */
export const analyzePrediction = async (imageUri: string): Promise<PredictionResult> => {
  console.warn(
    '⚠️  analyzePrediction() is deprecated and will be removed in a future version. ' +
    'Please use analyzeImage() instead for better functionality and support for multiple image types.'
  );

  // Use the enhanced analyze function
  return analyzeImage(imageUri, 'skin_lesion');
};

/**
 * New function to get symptom assessment from cloud healthcare APIs
 */
export const assessSymptoms = async (symptoms: string[], patientContext?: any) => {
  if (!cloudHealthcareService) {
    throw new Error('Cloud healthcare service not available for symptom assessment');
  }

  return await cloudHealthcareService.assessSymptoms(symptoms, patientContext);
};

/**
 * New function to get clinical insights from cloud healthcare APIs
 */
export const getClinicalInsights = async (patientData: any) => {
  if (!cloudHealthcareService) {
    throw new Error('Cloud healthcare service not available for clinical insights');
  }

  return await cloudHealthcareService.getClinicalInsights(patientData);
};

/**
 * New function to get emergency triage from cloud healthcare APIs
 */
export const triageEmergency = async (symptoms: string[]) => {
  if (!cloudHealthcareService) {
    throw new Error('Cloud healthcare service not available for emergency triage');
  }

  return await cloudHealthcareService.triageEmergency(symptoms);
};

/**
 * Utility function to check cloud healthcare availability
 */
export const getCloudHealthcareStatus = () => {
  if (!cloudHealthcareService) {
    return {
      available: false,
      providers: [],
      reason: 'Cloud healthcare not configured or disabled'
    };
  }

  return {
    available: true,
    providers: cloudHealthcareService.getAvailableProviders(),
    primaryProvider: import.meta.env.VITE_PRIMARY_CLOUD_PROVIDER || 'google',
    fallbackEnabled: CLOUD_HEALTHCARE_FALLBACK,
    consensusEnabled: ENABLE_CONSENSUS_ANALYSIS
  };
};
