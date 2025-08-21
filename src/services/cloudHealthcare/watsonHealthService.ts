import { CloudAnalysisResult, ClinicalInsightResult, CloudHealthcareRequest, SymptomAssessmentResult } from './types';

export class WatsonHealthService {
  private apiKey: string;
  private endpoint: string;
  private version: string;

  constructor(config: { apiKey: string; endpoint: string; version: string }) {
    this.apiKey = config.apiKey;
    this.endpoint = config.endpoint;
    this.version = config.version;
  }

  async analyzeImage(request: CloudHealthcareRequest): Promise<CloudAnalysisResult> {
    const startTime = Date.now();

    try {
      const base64Image = await this.fileToBase64(request.image);

      const response = await this.callWatsonAPI({
        type: 'image_analysis',
        image: base64Image,
        imageType: request.imageType,
        patientContext: request.patientContext
      });

      const processingTime = Date.now() - startTime;

      return {
        provider: 'watson',
        prediction: typeof response.diagnosis === 'string' ? response.diagnosis : 'unknown',
        confidence: typeof response.confidence === 'number' ? response.confidence : 0.5,
        findings: Array.isArray(response.findings) && response.findings.every(item => typeof item === 'string') ? response.findings : [],
        recommendations: Array.isArray(response.recommendations) && response.recommendations.every(item => typeof item === 'string') ? response.recommendations : [],
        riskLevel: this.calculateRiskLevel(
          typeof response.confidence === 'number' ? response.confidence : 0.5,
          typeof response.diagnosis === 'string' ? response.diagnosis : 'unknown'
        ),
        metadata: {
          modelVersion: typeof response.modelVersion === 'string' ? response.modelVersion : '1.0',
          processingTime,
          imageQuality: typeof response.imageQuality === 'number' ? response.imageQuality : 0.8
        },
        rawResponse: response
      };
    } catch (error) {
      console.error('Watson Health API error:', error);
      throw new Error(`Watson Health analysis failed: ${error.message}`);
    }
  }

  async getClinicalInsights(patientData: {
    age?: number;
    gender?: string;
    medicalHistory?: string[];
    currentSymptoms?: string[];
    testResults?: Record<string, unknown>;
  }): Promise<ClinicalInsightResult> {
    try {
      const response = await this.callWatsonAPI({
        type: 'clinical_insights',
        patientData
      });

      return {
        provider: 'watson',
        insights: Array.isArray(response.insights) && response.insights.every(item => typeof item === 'string') ? response.insights : [],
        riskFactors: Array.isArray(response.riskFactors) && response.riskFactors.every(item => typeof item === 'string') ? response.riskFactors : [],
        preventiveMeasures: Array.isArray(response.preventiveMeasures) && response.preventiveMeasures.every(item => typeof item === 'string') ? response.preventiveMeasures : [],
        followUpRecommendations: Array.isArray(response.followUpRecommendations) && response.followUpRecommendations.every(item => typeof item === 'string') ? response.followUpRecommendations : [],
        evidenceLevel: this.mapEvidenceLevel(typeof response.evidenceLevel === 'string' ? response.evidenceLevel : 'medium')
      };
    } catch (error) {
      console.error('Watson Health clinical insights error:', error);
      throw new Error(`Clinical insights failed: ${error.message}`);
    }
  }

  async assessSymptoms(symptoms: string[], patientContext?: Record<string, unknown>): Promise<SymptomAssessmentResult> {
    try {
      const response = await this.callWatsonAPI({
        type: 'symptom_assessment',
        symptoms,
        patientContext
      });

      return {
        provider: 'watson',
        assessment: typeof response.assessment === 'string' ? response.assessment : 'Unable to assess symptoms',
        severity: this.mapSeverity(typeof response.severity === 'string' ? response.severity : 'moderate'),
        recommendations: Array.isArray(response.recommendations) && response.recommendations.every(item => typeof item === 'string') ? response.recommendations : [],
        urgency: this.mapUrgency(typeof response.urgency === 'string' ? response.urgency : 'routine'),
        possibleConditions: Array.isArray(response.possibleConditions) ? response.possibleConditions : []
      };
    } catch (error) {
      console.error('Watson Health symptom assessment error:', error);
      throw new Error(`Symptom assessment failed: ${error.message}`);
    }
  }

  async getTreatmentRecommendations(diagnosis: string, patientContext?: Record<string, unknown>): Promise<{
    treatments: string[];
    medications: string[];
    lifestyle: string[];
    followUp: string[];
  }> {
    try {
      const response = await this.callWatsonAPI({
        type: 'treatment_recommendations',
        diagnosis,
        patientContext
      });

      return {
        treatments: Array.isArray(response.treatments) && response.treatments.every(item => typeof item === 'string') ? response.treatments : [],
        medications: Array.isArray(response.medications) && response.medications.every(item => typeof item === 'string') ? response.medications : [],
        lifestyle: Array.isArray(response.lifestyle) && response.lifestyle.every(item => typeof item === 'string') ? response.lifestyle : [],
        followUp: Array.isArray(response.followUp) && response.followUp.every(item => typeof item === 'string') ? response.followUp : []
      };
    } catch (error) {
      console.error('Watson Health treatment recommendations error:', error);
      throw new Error(`Treatment recommendations failed: ${error.message}`);
    }
  }

  async analyzeMedicalLiterature(query: string): Promise<{
    relevantStudies: Array<{
      title: string;
      authors: string[];
      year: number;
      summary: string;
      relevance: number;
    }>;
    keyFindings: string[];
    evidenceLevel: 'low' | 'medium' | 'high';
  }> {
    try {
      const response = await this.callWatsonAPI({
        type: 'literature_analysis',
        query
      });

      return {
        relevantStudies: Array.isArray(response.relevantStudies) ? response.relevantStudies : [],
        keyFindings: Array.isArray(response.keyFindings) && response.keyFindings.every(item => typeof item === 'string') ? response.keyFindings : [],
        evidenceLevel: this.mapEvidenceLevel(typeof response.evidenceLevel === 'string' ? response.evidenceLevel : 'medium')
      };
    } catch (error) {
      console.error('Watson Health literature analysis error:', error);
      throw new Error(`Literature analysis failed: ${error.message}`);
    }
  }

  private async callWatsonAPI(params: {
    type: string;
    [key: string]: unknown;
  }): Promise<Record<string, unknown>> {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${this.apiKey}`,
      'X-Watson-Version': this.version
    };

    const requestBody = {
      ...params,
      timestamp: new Date().toISOString()
    };

    const response = await fetch(`${this.endpoint}/api/watson`, {
      method: 'POST',
      headers,
      body: JSON.stringify(requestBody)
    });

    if (!response.ok) {
      throw new Error(`Watson Health API error: ${response.status} ${response.statusText}`);
    }

    return await response.json();
  }

  private async fileToBase64(file: File): Promise<string> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onload = () => {
        const base64 = reader.result as string;
        resolve(base64.split(',')[1]);
      };
      reader.onerror = error => reject(error);
    });
  }

  private mapSeverity(watsonSeverity: string): 'low' | 'medium' | 'high' | 'critical' {
    const severityMap: Record<string, 'low' | 'medium' | 'high' | 'critical'> = {
      'low': 'low',
      'mild': 'low',
      'moderate': 'medium',
      'medium': 'medium',
      'high': 'high',
      'severe': 'high',
      'critical': 'critical',
      'emergency': 'critical'
    };

    return severityMap[watsonSeverity.toLowerCase()] || 'medium';
  }

  private mapUrgency(watsonUrgency: string): 'routine' | 'soon' | 'urgent' | 'emergency' {
    const urgencyMap: Record<string, 'routine' | 'soon' | 'urgent' | 'emergency'> = {
      'routine': 'routine',
      'non_urgent': 'routine',
      'soon': 'soon',
      'urgent': 'urgent',
      'emergency': 'emergency',
      'immediate': 'emergency'
    };

    return urgencyMap[watsonUrgency.toLowerCase()] || 'routine';
  }

  private mapEvidenceLevel(watsonEvidenceLevel: string): 'low' | 'medium' | 'high' {
    const evidenceMap: Record<string, 'low' | 'medium' | 'high'> = {
      'low': 'low',
      'weak': 'low',
      'moderate': 'medium',
      'medium': 'medium',
      'strong': 'high',
      'high': 'high'
    };

    return evidenceMap[watsonEvidenceLevel.toLowerCase()] || 'medium';
  }

  private calculateRiskLevel(confidence: number, diagnosis: string): 'low' | 'medium' | 'high' {
    if (diagnosis.toLowerCase().includes('benign') && confidence > 0.8) {
      return 'low';
    }

    if (diagnosis.toLowerCase().includes('malignant') && confidence > 0.7) {
      return 'high';
    }

    if (confidence > 0.6) {
      return 'medium';
    }

    return 'high';
  }
}
