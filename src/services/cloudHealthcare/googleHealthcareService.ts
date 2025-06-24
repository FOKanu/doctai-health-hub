import { CloudAnalysisResult, CloudHealthcareRequest, ImageType } from './types';

export class GoogleHealthcareService {
  private projectId: string;
  private location: string;
  private datasetId: string;
  private apiKey?: string;

  constructor(config: {
    projectId: string;
    location: string;
    datasetId: string;
    apiKey?: string;
  }) {
    this.projectId = config.projectId;
    this.location = config.location;
    this.datasetId = config.datasetId;
    this.apiKey = config.apiKey;
  }

  async analyzeImage(request: CloudHealthcareRequest): Promise<CloudAnalysisResult> {
    const startTime = Date.now();

    try {
      // Convert image to base64 for API
      const base64Image = await this.fileToBase64(request.image);

      // Determine the appropriate model based on image type
      const modelEndpoint = this.getModelEndpoint(request.imageType);

      // Make API call to Google Cloud Healthcare
      const response = await this.callGoogleHealthcareAPI({
        image: base64Image,
        imageType: request.imageType,
        modelEndpoint,
        patientContext: request.patientContext
      });

      const processingTime = Date.now() - startTime;

      return {
        provider: 'google',
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
      console.error('Google Healthcare API error:', error);
      throw new Error(`Google Healthcare analysis failed: ${error.message}`);
    }
  }

  private getModelEndpoint(imageType: ImageType): string {
    const baseUrl = `https://healthcare.googleapis.com/v1/projects/${this.projectId}/locations/${this.location}/datasets/${this.datasetId}`;

    switch (imageType) {
      case 'skin_lesion':
        return `${baseUrl}/dicomStores/skin-lesion-models/instances`;
      case 'xray':
        return `${baseUrl}/dicomStores/chest-xray-models/instances`;
      case 'ct_scan':
        return `${baseUrl}/dicomStores/ct-scan-models/instances`;
      case 'mri':
        return `${baseUrl}/dicomStores/mri-models/instances`;
      default:
        return `${baseUrl}/dicomStores/general-models/instances`;
    }
  }

  private async callGoogleHealthcareAPI(params: {
    image: string;
    imageType: ImageType;
    modelEndpoint: string;
    patientContext?: any;
  }): Promise<any> {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };

    if (this.apiKey) {
      headers['Authorization'] = `Bearer ${this.apiKey}`;
    }

    const requestBody = {
      image: params.image,
      imageType: params.imageType,
      patientContext: params.patientContext,
      // Add any additional parameters needed for Google Healthcare API
    };

    const response = await fetch(params.modelEndpoint, {
      method: 'POST',
      headers,
      body: JSON.stringify(requestBody)
    });

    if (!response.ok) {
      throw new Error(`Google Healthcare API error: ${response.status} ${response.statusText}`);
    }

    return await response.json();
  }

  private async fileToBase64(file: File): Promise<string> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onload = () => {
        const base64 = reader.result as string;
        // Remove data URL prefix
        resolve(base64.split(',')[1]);
      };
      reader.onerror = error => reject(error);
    });
  }

  private calculateRiskLevel(confidence: number, diagnosis: string): 'low' | 'medium' | 'high' {
    // High confidence benign = low risk
    if (diagnosis.toLowerCase().includes('benign') && confidence > 0.8) {
      return 'low';
    }

    // High confidence malignant = high risk
    if (diagnosis.toLowerCase().includes('malignant') && confidence > 0.7) {
      return 'high';
    }

    // Medium confidence or unclear diagnosis = medium risk
    if (confidence > 0.6) {
      return 'medium';
    }

    return 'high'; // Default to high risk for low confidence
  }

  // Additional methods for specific medical imaging tasks
  async analyzeChestXray(image: File): Promise<CloudAnalysisResult> {
    return this.analyzeImage({
      image,
      imageType: 'xray',
      priority: 'normal'
    });
  }

  async analyzeSkinLesion(image: File): Promise<CloudAnalysisResult> {
    return this.analyzeImage({
      image,
      imageType: 'skin_lesion',
      priority: 'normal'
    });
  }

  async analyzeCTScan(image: File): Promise<CloudAnalysisResult> {
    return this.analyzeImage({
      image,
      imageType: 'ct_scan',
      priority: 'normal'
    });
  }
}
