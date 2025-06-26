
// Feature flags for controlled rollout
export const USE_NEW_PREDICTION_API = import.meta.env?.VITE_USE_NEW_PREDICTION_API === 'true' || false;
export const DEBUG_PREDICTIONS = import.meta.env?.VITE_DEBUG_PREDICTIONS === 'true' || false;

// Legacy prediction result interface (backward compatible)
export interface PredictionResult {
  prediction: 'benign' | 'malignant';
  confidence: number;
  probabilities: {
    benign: number;
    malignant: number;
  };
  timestamp: string;
  imageId?: string;
  metadata?: {
    provider?: string;
    modelVersion?: string;
    findings?: string[];
    recommendations?: string[];
    riskLevel?: 'low' | 'medium' | 'high';
    processingTime?: number;
  };
}

// Modern prediction result interface
export interface ModernPredictionResult {
  id: string;
  imageId: string;
  imageType: 'skin_lesion' | 'ct_scan' | 'mri' | 'xray' | 'eeg';
  modelName: string;
  modelVersion?: string;
  predictedClass: number;
  confidence: number;
  probabilities: number[];
  classLabels: string[];
  recommendations: string[];
  riskLevel: 'low' | 'medium' | 'high';
  createdAt: string;
}

// Supported image types (includes 'general' for compatibility)
export type ImageType = 'skin_lesion' | 'ct_scan' | 'mri' | 'xray' | 'eeg' | 'general';
