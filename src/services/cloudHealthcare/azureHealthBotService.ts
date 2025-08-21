import { SymptomAssessmentResult, CloudAnalysisResult, CloudHealthcareRequest } from './types';

// Type guards for API responses
const isString = (value: unknown): value is string => typeof value === 'string';
const isNumber = (value: unknown): value is number => typeof value === 'number';
const isBoolean = (value: unknown): value is boolean => typeof value === 'boolean';
const isStringArray = (value: unknown): value is string[] => Array.isArray(value) && value.every(item => typeof item === 'string');

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
        assessment: isString(response.assessment) ? response.assessment : 'Unable to assess symptoms',
        severity: this.mapSeverity(isString(response.severity) ? response.severity : 'moderate'),
        recommendations: isStringArray(response.recommendations) ? response.recommendations : [],
        urgency: this.mapUrgency(isString(response.urgency) ? response.urgency : 'routine'),
        possibleConditions: this.parsePossibleConditions(response.possibleConditions)
      };
    } catch (error) {
      console.error('Azure Health Bot API error:', error);
      throw new Error(`Symptom assessment failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
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
        prediction: isString(response.diagnosis) ? response.diagnosis : 'unknown',
        confidence: isNumber(response.confidence) ? response.confidence : 0.5,
        findings: isStringArray(response.findings) ? response.findings : [],
        recommendations: isStringArray(response.recommendations) ? response.recommendations : [],
        riskLevel: this.calculateRiskLevel(
          isNumber(response.confidence) ? response.confidence : 0.5,
          isString(response.diagnosis) ? response.diagnosis : 'unknown'
        ),
        metadata: {
          modelVersion: isString(response.modelVersion) ? response.modelVersion : '1.0',
          processingTime,
          imageQuality: isNumber(response.imageQuality) ? response.imageQuality : 0.8
        },
        rawResponse: response
      };
    } catch (error) {
      console.error('Azure Health Bot image analysis error:', error);
      throw new Error(`Azure Health Bot analysis failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  async getMedicalAdvice(query: string, patientContext?: Record<string, unknown>): Promise<string[]> {
    try {
      const response = await this.callAzureHealthBotAPI({
        type: 'medical_advice',
        query,
        patientContext
      });

      return isStringArray(response.advice) ? response.advice : [];
    } catch (error) {
      console.error('Azure Health Bot advice error:', error);
      throw new Error(`Medical advice failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
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
        isEmergency: isBoolean(response.isEmergency) ? response.isEmergency : false,
        urgency: this.mapUrgency(isString(response.urgency) ? response.urgency : 'routine'),
        recommendations: isStringArray(response.recommendations) ? response.recommendations : []
      };
    } catch (error) {
      console.error('Azure Health Bot triage error:', error);
      throw new Error(`Emergency triage failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
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

  private parsePossibleConditions(conditions: unknown): Array<{
    condition: string;
    probability: number;
    description: string;
  }> {
    if (!Array.isArray(conditions)) {
      return [];
    }

    return conditions
      .filter((condition): condition is Record<string, unknown> => 
        typeof condition === 'object' && condition !== null
      )
      .map(condition => ({
        condition: isString(condition.condition) ? condition.condition : 'Unknown condition',
        probability: isNumber(condition.probability) ? condition.probability : 0,
        description: isString(condition.description) ? condition.description : ''
      }));
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
