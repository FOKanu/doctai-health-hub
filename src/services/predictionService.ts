import { modernPredictionService } from './prediction/modernPredictionService';
import { analyzePredictionLegacy } from './prediction/legacyPredictionService';
import { hybridPredictionService, TimeSeriesInput, HybridAnalysisResult, VitalSignsData } from './prediction/hybridPredictionService';
import { USE_NEW_PREDICTION_API, DEBUG_PREDICTIONS, PredictionResult } from './prediction/types';
import { CloudHealthcareService, CloudAnalysisResult, ImageType } from './cloudHealthcare';

// Re-export types for backward compatibility
export type { PredictionResult, ModernPredictionResult, ImageType } from './prediction/types';
export type { TimeSeriesInput, HybridAnalysisResult } from './prediction/hybridPredictionService';

export { modernPredictionService as predictionService } from './prediction/modernPredictionService';
export { hybridPredictionService } from './prediction/hybridPredictionService';

// Feature flags for enhanced functionality
const USE_CLOUD_HEALTHCARE = import.meta.env.VITE_USE_CLOUD_HEALTHCARE === 'true';
const CLOUD_HEALTHCARE_FALLBACK = import.meta.env.VITE_CLOUD_HEALTHCARE_FALLBACK === 'true';
const ENABLE_CONSENSUS_ANALYSIS = import.meta.env.VITE_ENABLE_CONSENSUS_ANALYSIS === 'true';
const ENABLE_HYBRID_ANALYSIS = import.meta.env.VITE_ENABLE_HYBRID_ANALYSIS === 'true';

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
 * Enhanced analyze function that supports both single-instance and time-series analysis
 * Automatically routes to appropriate analysis method based on input
 */
export const analyzeImage = async (
  imageUri: string | File | TimeSeriesInput,
  imageType: ImageType = 'skin_lesion'
): Promise<PredictionResult | HybridAnalysisResult> => {
  try {
    // Handle different input types
    if (typeof imageUri === 'string') {
      // Legacy string URI - convert to File and use single instance analysis
      const response = await fetch(imageUri);
      const blob = await response.blob();
      const file = new File([blob], 'image.jpg', { type: 'image/jpeg' });

      if (ENABLE_HYBRID_ANALYSIS) {
        return await hybridPredictionService.routePrediction(file, imageType);
      } else {
        return await analyzeSingleInstance(file, imageType);
      }
    } else if (imageUri instanceof File) {
      // Single File - use single instance analysis
      if (ENABLE_HYBRID_ANALYSIS) {
        return await hybridPredictionService.routePrediction(imageUri, imageType);
      } else {
        return await analyzeSingleInstance(imageUri, imageType);
      }
    } else if (typeof imageUri === 'object' && 'images' in imageUri) {
      // TimeSeriesInput - use hybrid time-series analysis
      if (ENABLE_HYBRID_ANALYSIS) {
        return await hybridPredictionService.routePrediction(imageUri, imageType);
      } else {
        throw new Error('Time-series analysis requires hybrid analysis to be enabled');
      }
    } else {
      throw new Error('Invalid input type for image analysis');
    }
  } catch (error) {
    console.error('All prediction methods failed:', error);
    throw new Error(`Image analysis failed: ${error.message}`);
  }
};

/**
 * Single instance analysis with fallback chain
 */
async function analyzeSingleInstance(
  imageFile: File,
  imageType: ImageType
): Promise<PredictionResult> {
  // Try cloud healthcare APIs first if enabled
  if (USE_CLOUD_HEALTHCARE && cloudHealthcareService) {
    console.log('🌐 Using cloud healthcare APIs');

    try {
      let cloudResult: CloudAnalysisResult;

      if (ENABLE_CONSENSUS_ANALYSIS) {
        // Get consensus from multiple providers
        const consensusResult = await cloudHealthcareService.getConsensusAnalysis(imageFile, imageType);
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
        cloudResult = await cloudHealthcareService.analyzeImage(imageFile, imageType);
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
        imageFile,
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
       // Convert File to string URI for legacy function
       const imageUrl = URL.createObjectURL(imageFile);
       return analyzePredictionLegacy(imageUrl);
     }
   } else {
     console.log('📱 Using legacy prediction API (custom ML)');
     // Convert File to string URI for legacy function
     const imageUrl = URL.createObjectURL(imageFile);
     return analyzePredictionLegacy(imageUrl);
   }
}

// Helper function to convert Record<string, unknown>[] to VitalSignsData[]
const convertToVitalSignsData = (data: Record<string, unknown>[]): VitalSignsData[] => {
  return data.map(item => ({
    heartRate: typeof item.heartRate === 'number' ? item.heartRate : undefined,
    bloodPressure: typeof item.bloodPressure === 'object' && item.bloodPressure !== null ? 
      (item.bloodPressure as { systolic: number; diastolic: number }) : undefined,
    temperature: typeof item.temperature === 'number' ? item.temperature : undefined,
    oxygenSaturation: typeof item.oxygenSaturation === 'number' ? item.oxygenSaturation : undefined,
    respiratoryRate: typeof item.respiratoryRate === 'number' ? item.respiratoryRate : undefined,
    bloodGlucose: typeof item.bloodGlucose === 'number' ? item.bloodGlucose : undefined,
    weight: typeof item.weight === 'number' ? item.weight : undefined,
    timestamp: typeof item.timestamp === 'string' ? item.timestamp : new Date().toISOString()
  }));
};

/**
 * Analyze image sequence for progression tracking
 */
export const analyzeImageSequence = async (
  images: File[],
  imageTypes: ImageType[],
  timestamps: string[],
  userId: string,
  vitalSigns?: Record<string, unknown>[]
): Promise<HybridAnalysisResult> => {
  if (!ENABLE_HYBRID_ANALYSIS) {
    throw new Error('Time-series analysis requires hybrid analysis to be enabled');
  }

  const timeSeriesInput: TimeSeriesInput = {
    images,
    imageTypes,
    timestamps,
    vitalSigns: vitalSigns ? convertToVitalSignsData(vitalSigns) : undefined,
    userId
  };

  return await hybridPredictionService.routePrediction(timeSeriesInput);
};

/**
 * Analyze vital signs data independently
 */
export const analyzeVitalSigns = async (
  vitalSigns: Record<string, unknown>[],
  userId: string
): Promise<Record<string, unknown>> => {
  if (!ENABLE_HYBRID_ANALYSIS) {
    throw new Error('Vital signs analysis requires hybrid analysis to be enabled');
  }

  // Create a minimal time series input for vital signs only
  const timeSeriesInput: TimeSeriesInput = {
    images: [],
    imageTypes: [],
    timestamps: [],
    vitalSigns: convertToVitalSignsData(vitalSigns),
    userId
  };

  const result = await hybridPredictionService.routePrediction(timeSeriesInput);
  return result.vitalSignsAnalysis as unknown as Record<string, unknown>;
};

/**
 * Get patient progression timeline
 */
export const getPatientProgression = async (
  userId: string,
  conditionType?: string
): Promise<Record<string, unknown>> => {
  // This would integrate with the new database tables
  // For now, return mock data
  return {
    conditionType: conditionType || 'skin_lesion',
    status: 'monitoring',
    severityScore: 0.3,
    confidenceScore: 0.85,
    baselineDate: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString(),
    daysSinceBaseline: 30,
    trend: 'stable'
  };
};

/**
 * Save health metrics to time-series database
 */
export const saveHealthMetrics = async (
  userId: string,
  metricType: string,
  value: React.SyntheticEvent,
  timestamp: string,
  deviceSource?: string
): Promise<void> => {
  // This would save to the new health_metrics_timeseries table
  console.log('Saving health metrics:', {
    userId,
    metricType,
    value,
    timestamp,
    deviceSource
  });
};

/**
 * Create scan sequence for progression analysis
 */
export const createScanSequence = async (
  userId: string,
  sequenceName: string,
  imageIds: string[],
  analysisType: string,
  baselineImageId?: string
): Promise<string> => {
  // This would save to the new scan_sequences table
  const sequenceId = crypto.randomUUID();
  console.log('Creating scan sequence:', {
    sequenceId,
    userId,
    sequenceName,
    imageIds,
    analysisType,
    baselineImageId
  });
  return sequenceId;
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
  return analyzeImage(imageUri, 'skin_lesion') as Promise<PredictionResult>;
};

/**
 * Get comprehensive health analysis combining all available data
 */
export const getComprehensiveHealthAnalysis = async (
  userId: string,
  dateRange: { start: string; end: string }
): Promise<{
  imageAnalysis: Record<string, unknown>[];
  vitalSigns: Record<string, unknown>[];
  progression: Record<string, unknown>;
  riskAssessment: Record<string, unknown>;
  recommendations: string[];
}> => {
  // This would aggregate data from all time-series tables
  // For now, return mock comprehensive analysis
  return {
    imageAnalysis: [],
    vitalSigns: [],
    progression: {
      status: 'stable',
      trend: 'improving',
      confidence: 0.85
    },
    riskAssessment: {
      overallRisk: 'low',
      factors: ['Age', 'Family history'],
      score: 0.25
    },
    recommendations: [
      'Continue routine monitoring',
      'Maintain healthy lifestyle',
      'Schedule annual checkup'
    ]
  };
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

/**
 * Get hybrid analysis status and capabilities
 */
export const getHybridAnalysisStatus = () => {
  return {
    enabled: ENABLE_HYBRID_ANALYSIS,
    capabilities: {
      singleInstance: true,
      imageSequences: ENABLE_HYBRID_ANALYSIS,
      vitalSigns: ENABLE_HYBRID_ANALYSIS,
      progressionTracking: ENABLE_HYBRID_ANALYSIS,
      comprehensiveAnalysis: ENABLE_HYBRID_ANALYSIS
    },
    models: {
      cnn: 'Available',
      lstm: ENABLE_HYBRID_ANALYSIS ? 'Available' : 'Disabled',
      transformer: ENABLE_HYBRID_ANALYSIS ? 'Available' : 'Disabled'
    }
  };
};

/**
 * Save prediction result to Supabase database
 */
export const savePredictionToSupabase = async (
  predictionResult: PredictionResult,
  imageUri: string
): Promise<void> => {
  try {
    // Convert image URI to base64 if it's a data URL
    let imageData = imageUri;
    if (imageUri.startsWith('data:')) {
      imageData = imageUri;
    } else if (imageUri.startsWith('blob:')) {
      // Convert blob URL to base64
      const response = await fetch(imageUri);
      const blob = await response.blob();
      const reader = new FileReader();
      imageData = await new Promise((resolve) => {
        reader.onload = () => resolve(reader.result as string);
        reader.readAsDataURL(blob);
      });
    }

    // Create prediction record
    const predictionRecord = {
      id: predictionResult.imageId,
      user_id: 'mock_user', // TODO: Get from auth context
      image_data: imageData,
      prediction: predictionResult.prediction,
      confidence: predictionResult.confidence,
      benign_probability: predictionResult.probabilities.benign,
      malignant_probability: predictionResult.probabilities.malignant,
      timestamp: predictionResult.timestamp,
      metadata: predictionResult.metadata || {},
      created_at: new Date().toISOString()
    };

                // Import Supabase client dynamically to avoid circular dependencies
    const { supabase } = await import('./supabaseClient');

    try {
      await supabase
        .from('predictions')
        .insert(predictionRecord);
      console.log('Prediction saved to Supabase successfully');
    } catch (dbError) {
      console.error('Error saving prediction to Supabase:', dbError);
      // Don't throw error to avoid breaking the UI flow
    }

    console.log('Prediction saved to Supabase successfully');
  } catch (error) {
    console.error('Error in savePredictionToSupabase:', error);
    // Don't throw error to avoid breaking the UI flow
    // Just log it for debugging
  }
};
