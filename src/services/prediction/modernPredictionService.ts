
import { ModernPredictionResult, ImageType, DEBUG_PREDICTIONS } from './types';

export class ModernPredictionService {
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
    imageType: ImageType,
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

    // Ensure riskLevel is properly typed
    const riskLevel: 'low' | 'medium' | 'high' = predictedClass === 0 ? 'low' : predictedClass === 1 ? 'medium' : 'high';

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

// Export singleton instance
export const modernPredictionService = new ModernPredictionService();
