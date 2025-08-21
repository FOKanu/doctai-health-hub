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
    result !== null &&
    'prediction' in result &&
    typeof (result as any).prediction === 'string' &&
    ((result as any).prediction === 'benign' || (result as any).prediction === 'malignant') &&
    'confidence' in result &&
    typeof (result as any).confidence === 'number' &&
    'probabilities' in result &&
    (result as any).probabilities &&
    typeof (result as any).probabilities.benign === 'number' &&
    typeof (result as any).probabilities.malignant === 'number' &&
    'timestamp' in result &&
    typeof (result as any).timestamp === 'string'
  );
};

/**
 * Validates that a prediction result has the expected modern format
 */
export const isModernPredictionResult = (result: unknown): result is ModernPredictionResult => {
  return (
    result &&
    typeof result === 'object' &&
    result !== null &&
    'id' in result &&
    typeof (result as any).id === 'string' &&
    'imageId' in result &&
    typeof (result as any).imageId === 'string' &&
    'imageType' in result &&
    typeof (result as any).imageType === 'string' &&
    'modelName' in result &&
    typeof (result as any).modelName === 'string' &&
    'predictedClass' in result &&
    typeof (result as any).predictedClass === 'number' &&
    'confidence' in result &&
    typeof (result as any).confidence === 'number' &&
    'probabilities' in result &&
    Array.isArray((result as any).probabilities) &&
    'classLabels' in result &&
    Array.isArray((result as any).classLabels) &&
    'recommendations' in result &&
    Array.isArray((result as any).recommendations) &&
    'riskLevel' in result &&
    typeof (result as any).riskLevel === 'string' &&
    'createdAt' in result &&
    typeof (result as any).createdAt === 'string'
  );
};
