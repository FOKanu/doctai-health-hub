import { SymptomAssessmentResult, CloudAnalysisResult, CloudHealthcareRequest } from './types';

export class AzureHealthBotService {
  private endpoint: string;
  private apiKey: string;

  constructor(config: { endpoint: string; apiKey: string }) {
    this.endpoint = config.endpoint;
    this.apiKey = config.apiKey;
  }

  async assessSymptoms(symptoms: string[], patientContext?: Record<string, unknown>): Promise<SymptomAssessmentResult> {
    try {
      const response = await this.callAzureHealthBotAPI({
        type: 'symptom_assessment',
        symptoms,
        patientContext
      });

      return {
        provider: 'azure',
        assessment: response.assessment || 'Unable to assess symptoms',
        severity: this.mapSeverity(response.severity),
        recommendations: response.recommendations || [],
        urgency: this.mapUrgency(response.urgency),
        possibleConditions: response.possibleConditions || []
      };
    } catch (error) {
      console.error('Azure Health Bot API error:', error);
      throw new Error(`Symptom assessment failed: ${error.message}`);
    }
  }

  async analyzeImage(request: CloudHealthcareRequest): Promise<CloudAnalysisResult> {
    const startTime = Date.now();

    try {
      const base64Image = await this.fileToBase64(request.image);

      const response = await this.callAzureHealthBotAPI({
        type: 'image_analysis',
        image: base64Image,
        imageType: request.imageType,
        patientContext: request.patientContext
      });

      const processingTime = Date.now() - startTime;

      return {
        provider: 'azure',
        prediction: response.diagnosis || 'unknown',
        confidence: response.confidence || 0.5,
        findings: response.findings || [],
        recommendations: response.recommendations || [],
        riskLevel: this.calculateRiskLevel(response.confidence, response.diagnosis),
        metadata: {
          modelVersion: response.modelVersion || '1.0',
          processingTime,
          imageQuality: response.imageQuality || 0.8
        },
        rawResponse: response
      };
    } catch (error) {
      console.error('Azure Health Bot image analysis error:', error);
      throw new Error(`Azure Health Bot analysis failed: ${error.message}`);
    }
  }

  async getMedicalAdvice(query: string, patientContext?: Record<string, unknown>): Promise<string[]> {
    try {
      const response = await this.callAzureHealthBotAPI({
        type: 'medical_advice',
        query,
        patientContext
      });

      return response.advice || [];
    } catch (error) {
      console.error('Azure Health Bot advice error:', error);
      throw new Error(`Medical advice failed: ${error.message}`);
    }
  }

  async triageEmergency(symptoms: string[]): Promise<{
    isEmergency: boolean;
    urgency: 'routine' | 'soon' | 'urgent' | 'emergency';
    recommendations: string[];
  }> {
    try {
      const response = await this.callAzureHealthBotAPI({
        type: 'emergency_triage',
        symptoms
      });

      return {
        isEmergency: response.isEmergency || false,
        urgency: this.mapUrgency(response.urgency),
        recommendations: response.recommendations || []
      };
    } catch (error) {
      console.error('Azure Health Bot triage error:', error);
      throw new Error(`Emergency triage failed: ${error.message}`);
    }
  }

  private async callAzureHealthBotAPI(params: {
    type: string;
    [key: string]: unknown;
  }): Promise<Record<string, unknown>> {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${this.apiKey}`,
    };

    const requestBody = {
      ...params,
      timestamp: new Date().toISOString(),
      language: 'en' // Can be made configurable
    };

    const response = await fetch(`${this.endpoint}/api/healthbot`, {
      method: 'POST',
      headers,
      body: JSON.stringify(requestBody)
    });

    if (!response.ok) {
      throw new Error(`Azure Health Bot API error: ${response.status} ${response.statusText}`);
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

  private mapSeverity(azureSeverity: string): 'low' | 'medium' | 'high' | 'critical' {
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

    return severityMap[azureSeverity.toLowerCase()] || 'medium';
  }

  private mapUrgency(azureUrgency: string): 'routine' | 'soon' | 'urgent' | 'emergency' {
    const urgencyMap: Record<string, 'routine' | 'soon' | 'urgent' | 'emergency'> = {
      'routine': 'routine',
      'non_urgent': 'routine',
      'soon': 'soon',
      'urgent': 'urgent',
      'emergency': 'emergency',
      'immediate': 'emergency'
    };

    return urgencyMap[azureUrgency.toLowerCase()] || 'routine';
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
