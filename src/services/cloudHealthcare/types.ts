// Cloud Healthcare API Types
export interface CloudHealthcareConfig {
  googleHealthcare?: {
    projectId: string;
    location: string;
    datasetId: string;
    apiKey?: string;
  };
  azureHealthBot?: {
    endpoint: string;
    apiKey: string;
  };
  watsonHealth?: {
    apiKey: string;
    endpoint: string;
    version: string;
  };
}

export interface CloudAnalysisResult {
  provider: 'google' | 'azure' | 'watson';
  prediction: string;
  confidence: number;
  findings: string[];
  recommendations: string[];
  riskLevel: 'low' | 'medium' | 'high';
  metadata: {
    modelVersion: string;
    processingTime: number;
    imageQuality: number;
  };
  rawResponse?: Record<string, unknown>;
}

export interface SymptomAssessmentResult {
  provider: 'azure' | 'watson';
  assessment: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  recommendations: string[];
  urgency: 'routine' | 'soon' | 'urgent' | 'emergency';
  possibleConditions: Array<{
    condition: string;
    probability: number;
    description: string;
  }>;
}

export interface ClinicalInsightResult {
  provider: 'watson' | 'google';
  insights: string[];
  riskFactors: string[];
  preventiveMeasures: string[];
  followUpRecommendations: string[];
  evidenceLevel: 'low' | 'medium' | 'high';
}

export type ImageType = 'skin_lesion' | 'ct_scan' | 'mri' | 'xray' | 'eeg' | 'general';

export interface CloudHealthcareRequest {
  image: File;
  imageType: ImageType;
  patientContext?: {
    age?: number;
    gender?: string;
    medicalHistory?: string[];
    currentSymptoms?: string[];
  };
  priority?: 'low' | 'normal' | 'high';
}
