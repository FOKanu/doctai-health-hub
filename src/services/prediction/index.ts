
// Main exports
export { analyzeImage, analyzePrediction, predictionService, savePredictionToSupabase } from '../predictionService';

// Type exports
export type { PredictionResult, ModernPredictionResult, ImageType } from './types';

// Service exports for advanced usage
export { modernPredictionService } from './modernPredictionService';
export { analyzePredictionLegacy } from './legacyPredictionService';
