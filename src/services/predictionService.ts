
import { modernPredictionService } from './prediction/modernPredictionService';
import { analyzePredictionLegacy } from './prediction/legacyPredictionService';
import { USE_NEW_PREDICTION_API, DEBUG_PREDICTIONS, PredictionResult } from './prediction/types';

// Re-export types for backward compatibility
export type { PredictionResult, ModernPredictionResult, ImageType } from './prediction/types';
export { savePredictionToSupabase } from './prediction/databaseService';
export { modernPredictionService as predictionService } from './prediction/modernPredictionService';

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
      const modernResult = await modernPredictionService.analyzeImage(
        file,
        'skin_lesion',
        crypto.randomUUID()
      );

      // Adapt to legacy format
      const benignProbability = modernResult.classLabels.includes('Benign')
        ? modernResult.probabilities[modernResult.classLabels.indexOf('Benign')]
        : (modernResult.riskLevel === 'low' ? 0.8 : 0.2);

      const malignantProbability = 1 - benignProbability;
      // Ensure prediction is properly typed
      const prediction: 'benign' | 'malignant' = benignProbability > 0.5 ? 'benign' : 'malignant';

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
      return analyzePredictionLegacy(imageUri);
    }
  } else {
    console.log('📱 Using legacy prediction API');
    return analyzePredictionLegacy(imageUri);
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
