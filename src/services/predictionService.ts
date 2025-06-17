// Feature flag for controlled rollout - using Vite's import.meta.env instead of process.env
const USE_NEW_PREDICTION_API = import.meta.env?.VITE_USE_NEW_PREDICTION_API === 'true' || false;
const DEBUG_PREDICTIONS = import.meta.env?.VITE_DEBUG_PREDICTIONS === 'true' || false;

export interface PredictionResult {
  prediction: 'benign' | 'malignant';
  confidence: number;
  probabilities: {
    benign: number;
    malignant: number;
  };
  timestamp: string;
  imageId?: string;
}

// New modern prediction result interface (added alongside existing interface)
export interface ModernPredictionResult {
  id: string;
  imageId: string;
  imageType: 'skin_lesion' | 'ct_scan' | 'mri' | 'xray' | 'eeg';
  modelName: string;
  predictedClass: number;
  confidence: number;
  probabilities: number[];
  classLabels: string[];
  recommendations: string[];
  riskLevel: 'low' | 'medium' | 'high';
  createdAt: string;
}

// Modern prediction service class (new addition, doesn't affect existing code)
class ModernPredictionService {
  private apiEndpoint: string;

  constructor() {
    // Use Vite's import.meta.env with safe fallback
    this.apiEndpoint = import.meta.env?.VITE_ML_API_ENDPOINT || 'http://localhost:8000/api/predict';
  }

  /**
   * Modern analyze image method with support for multiple image types
   */
  async analyzeImage(
    imageFile: File,
    imageType: 'skin_lesion' | 'ct_scan' | 'mri' | 'xray' | 'eeg',
    imageId: string
  ): Promise<ModernPredictionResult> {
    if (DEBUG_PREDICTIONS) {
      console.log('🔬 Modern API: Analyzing image', { imageType, imageId });
    }

    // Simulate modern ML analysis
    await new Promise(resolve => setTimeout(resolve, 2000));

    const mockClassLabels = {
      'skin_lesion': ['Benign', 'Melanoma', 'Basal Cell Carcinoma', 'Squamous Cell Carcinoma'],
      'ct_scan': ['Normal', 'Tumor', 'Infection', 'Fracture'],
      'mri': ['Normal', 'Tumor', 'Lesion'],
      'xray': ['Normal', 'Pneumonia', 'COVID-19'],
      'eeg': ['Normal', 'Seizure', 'Sleep', 'Artifact']
    };

    const labels = mockClassLabels[imageType];
    const predictedClass = Math.floor(Math.random() * labels.length);
    const confidence = 0.7 + Math.random() * 0.3; // 70-100% confidence

    // Generate mock probabilities
    const probabilities = labels.map((_, i) =>
      i === predictedClass ? confidence : (1 - confidence) / (labels.length - 1)
    );

    const riskLevel = predictedClass === 0 ? 'low' : predictedClass === 1 ? 'medium' : 'high';

    const result = {
      id: crypto.randomUUID(),
      imageId,
      imageType,
      modelName: `${imageType}_classifier_v1`,
      predictedClass,
      confidence,
      probabilities,
      classLabels: labels,
      recommendations: [
        'Consult with a medical professional',
        'Monitor for changes',
        'Follow up in 3 months'
      ],
      riskLevel,
      createdAt: new Date().toISOString()
    };

    if (DEBUG_PREDICTIONS) {
      console.log('🔬 Modern API: Result', result);
    }

    return result;
  }
}

// Export the modern prediction service instance (new, doesn't break existing code)
export const predictionService = new ModernPredictionService();

// Store the original function implementation for safe fallback
const originalAnalyzePrediction = async (imageUri: string): Promise<PredictionResult> => {
  if (DEBUG_PREDICTIONS) {
    console.log('📱 Legacy API: Analyzing image');
  }

  // Simulate AI analysis - in a real app, this would call an ML API
  await new Promise(resolve => setTimeout(resolve, 2000)); // Simulate processing time

  // Mock prediction results
  const benignProbability = Math.random();
  const malignantProbability = 1 - benignProbability;
  const prediction = benignProbability > 0.5 ? 'benign' : 'malignant';

  const result = {
    prediction,
    confidence: Math.max(benignProbability, malignantProbability),
    probabilities: {
      benign: benignProbability,
      malignant: malignantProbability
    },
    timestamp: new Date().toISOString()
  };

  if (DEBUG_PREDICTIONS) {
    console.log('📱 Legacy API: Result', result);
  }

  return result;
};

/**
 * Feature flag controlled analyze function
 * This function conditionally switches between legacy and modern APIs
 */
export const analyzeImage = async (imageUri: string): Promise<PredictionResult> => {
  if (USE_NEW_PREDICTION_API) {
    console.log('🚀 Using new prediction API');
    try {
      // Convert imageUri to File for modern API
      const response = await fetch(imageUri);
      const blob = await response.blob();
      const file = new File([blob], 'image.jpg', { type: 'image/jpeg' });

      // Use modern API
      const modernResult = await predictionService.analyzeImage(
        file,
        'skin_lesion',
        crypto.randomUUID()
      );

      // Adapt to legacy format
      const benignProbability = modernResult.classLabels.includes('Benign')
        ? modernResult.probabilities[modernResult.classLabels.indexOf('Benign')]
        : (modernResult.riskLevel === 'low' ? 0.8 : 0.2);

      const malignantProbability = 1 - benignProbability;
      const prediction = benignProbability > 0.5 ? 'benign' : 'malignant';

      return {
        prediction,
        confidence: modernResult.confidence,
        probabilities: {
          benign: benignProbability,
          malignant: malignantProbability
        },
        timestamp: modernResult.createdAt,
        imageId: modernResult.imageId
      };
    } catch (error) {
      console.warn('New API failed, falling back to legacy:', error);
      return originalAnalyzePrediction(imageUri);
    }
  } else {
    console.log('📱 Using legacy prediction API');
    return originalAnalyzePrediction(imageUri);
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

  // Use the feature flag controlled function
  return analyzeImage(imageUri);
};

export const savePredictionToSupabase = async (
  predictionResult: PredictionResult,
  imageUri: string
): Promise<void> => {
  if (DEBUG_PREDICTIONS) {
    console.log('💾 Saving prediction to database:', predictionResult);
  }

  // Mock save to database - in a real app, this would save to Supabase
  console.log('Saving prediction to database:', predictionResult);
  console.log('Image URI:', imageUri);

  // Simulate database save
  await new Promise(resolve => setTimeout(resolve, 500));

  // In a real implementation, you would:
  // 1. Upload the image to Supabase Storage
  // 2. Save the prediction results to a database table
  // 3. Link the image and prediction data
};
