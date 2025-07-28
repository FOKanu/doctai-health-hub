import { LegacyPredictionResult, ModernPredictionResult } from '../types/types';

/**
 * Converts a modern prediction result to the legacy format
 * This ensures backward compatibility with existing components
 */
export const adaptToLegacyFormat = (modernResult: ModernPredictionResult): LegacyPredictionResult => {
  // Find the benign probability from the modern result
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
};

/**
 * Converts a legacy prediction result to a modern format
 * This is useful for upgrading legacy data to the new system
 */
export const adaptToModernFormat = (
  legacyResult: LegacyPredictionResult,
  imageType: 'skin_lesion' | 'ct_scan' | 'mri' | 'xray' | 'eeg' = 'skin_lesion'
): ModernPredictionResult => {
  const classLabels = ['Benign', 'Malignant'];
  const predictedClass = legacyResult.prediction === 'benign' ? 0 : 1;
  const probabilities = [
    legacyResult.probabilities.benign,
    legacyResult.probabilities.malignant
  ];

  const riskLevel = legacyResult.prediction === 'benign' ? 'low' : 'high';

  return {
    id: crypto.randomUUID(),
    imageId: legacyResult.imageId || crypto.randomUUID(),
    imageType,
    modelName: `legacy_${imageType}_classifier`,
    predictedClass,
    confidence: legacyResult.confidence,
    probabilities,
    classLabels,
    recommendations: [
      legacyResult.prediction === 'benign'
        ? 'Continue routine monitoring'
        : 'Consult with a medical professional immediately',
      'Monitor for changes',
      'Follow up as recommended'
    ],
    riskLevel,
    createdAt: legacyResult.timestamp
  };
};

/**
 * Validates that a prediction result has the expected legacy format
 */
export const isLegacyPredictionResult = (result: unknown): result is LegacyPredictionResult => {
  return (
    result &&
    typeof result === 'object' &&
    typeof result.prediction === 'string' &&
    (result.prediction === 'benign' || result.prediction === 'malignant') &&
    typeof result.confidence === 'number' &&
    result.probabilities &&
    typeof result.probabilities.benign === 'number' &&
    typeof result.probabilities.malignant === 'number' &&
    typeof result.timestamp === 'string'
  );
};

/**
 * Validates that a prediction result has the expected modern format
 */
export const isModernPredictionResult = (result: unknown): result is ModernPredictionResult => {
  return (
    result &&
    typeof result === 'object' &&
    typeof result.id === 'string' &&
    typeof result.imageId === 'string' &&
    typeof result.imageType === 'string' &&
    typeof result.modelName === 'string' &&
    typeof result.predictedClass === 'number' &&
    typeof result.confidence === 'number' &&
    Array.isArray(result.probabilities) &&
    Array.isArray(result.classLabels) &&
    Array.isArray(result.recommendations) &&
    typeof result.riskLevel === 'string' &&
    typeof result.createdAt === 'string'
  );
};
