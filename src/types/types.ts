// Legacy prediction result interface - stable and backward compatible
export interface LegacyPredictionResult {
  prediction: 'benign' | 'malignant';
  confidence: number;
  probabilities: {
    benign: number;
    malignant: number;
  };
  timestamp: string;
  imageId?: string;
}

// Modern prediction result interface
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

// Supported image types
export type ImageType = 'skin_lesion' | 'ct_scan' | 'mri' | 'xray' | 'eeg';

// Risk levels
export type RiskLevel = 'low' | 'medium' | 'high';
