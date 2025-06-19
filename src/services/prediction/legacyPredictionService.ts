
import { PredictionResult, DEBUG_PREDICTIONS } from './types';

/**
 * Legacy prediction service for backward compatibility
 */
export const analyzePredictionLegacy = async (imageUri: string): Promise<PredictionResult> => {
  if (DEBUG_PREDICTIONS) {
    console.log('📱 Legacy API: Analyzing image');
  }

  // Simulate AI analysis - in a real app, this would call an ML API
  await new Promise(resolve => setTimeout(resolve, 2000)); // Simulate processing time

  // Mock prediction results
  const benignProbability = Math.random();
  const malignantProbability = 1 - benignProbability;
  // Ensure prediction is properly typed
  const prediction: 'benign' | 'malignant' = benignProbability > 0.5 ? 'benign' : 'malignant';

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
