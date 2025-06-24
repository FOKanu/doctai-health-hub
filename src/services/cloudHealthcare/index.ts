import { GoogleHealthcareService } from './googleHealthcareService';
import { AzureHealthBotService } from './azureHealthBotService';
import { WatsonHealthService } from './watsonHealthService';
import {
  CloudAnalysisResult,
  SymptomAssessmentResult,
  ClinicalInsightResult,
  CloudHealthcareRequest,
  CloudHealthcareConfig,
  ImageType
} from './types';

// Feature flags for enabling/disabling providers
const CLOUD_PROVIDERS_CONFIG = {
  GOOGLE_HEALTHCARE: import.meta.env.VITE_ENABLE_GOOGLE_HEALTHCARE === 'true',
  AZURE_HEALTH_BOT: import.meta.env.VITE_ENABLE_AZURE_HEALTH_BOT === 'true',
  WATSON_HEALTH: import.meta.env.VITE_ENABLE_WATSON_HEALTH === 'true',
  USE_CLOUD_FALLBACK: import.meta.env.VITE_USE_CLOUD_FALLBACK === 'true',
  PRIMARY_PROVIDER: import.meta.env.VITE_PRIMARY_CLOUD_PROVIDER || 'google'
} as const;

export class CloudHealthcareService {
  private googleHealthcare?: GoogleHealthcareService;
  private azureHealthBot?: AzureHealthBotService;
  private watsonHealth?: WatsonHealthService;
  private config: CloudHealthcareConfig;

  constructor(config: CloudHealthcareConfig) {
    this.config = config;
    this.initializeProviders();
  }

  private initializeProviders() {
    if (CLOUD_PROVIDERS_CONFIG.GOOGLE_HEALTHCARE && this.config.googleHealthcare) {
      this.googleHealthcare = new GoogleHealthcareService(this.config.googleHealthcare);
    }

    if (CLOUD_PROVIDERS_CONFIG.AZURE_HEALTH_BOT && this.config.azureHealthBot) {
      this.azureHealthBot = new AzureHealthBotService(this.config.azureHealthBot);
    }

    if (CLOUD_PROVIDERS_CONFIG.WATSON_HEALTH && this.config.watsonHealth) {
      this.watsonHealth = new WatsonHealthService(this.config.watsonHealth);
    }
  }

  /**
   * Main method for analyzing medical images
   * Compatible with existing prediction service interface
   */
  async analyzeImage(image: File, imageType: ImageType, patientContext?: any): Promise<CloudAnalysisResult> {
    const request: CloudHealthcareRequest = {
      image,
      imageType,
      patientContext,
      priority: 'normal'
    };

    // Try primary provider first
    try {
      const primaryResult = await this.tryPrimaryProvider(request);
      if (primaryResult) {
        return primaryResult;
      }
    } catch (error) {
      console.warn('Primary cloud provider failed:', error);
    }

    // If fallback is enabled, try other providers
    if (CLOUD_PROVIDERS_CONFIG.USE_CLOUD_FALLBACK) {
      try {
        const fallbackResult = await this.tryFallbackProviders(request);
        if (fallbackResult) {
          return fallbackResult;
        }
      } catch (error) {
        console.warn('All cloud providers failed:', error);
      }
    }

    throw new Error('No cloud healthcare providers available');
  }

  /**
   * Get consensus analysis from multiple providers
   */
  async getConsensusAnalysis(image: File, imageType: ImageType, patientContext?: any): Promise<{
    consensus: CloudAnalysisResult;
    individualResults: CloudAnalysisResult[];
    agreement: number;
  }> {
    const request: CloudHealthcareRequest = {
      image,
      imageType,
      patientContext,
      priority: 'normal'
    };

    const results: CloudAnalysisResult[] = [];
    const promises: Promise<CloudAnalysisResult | null>[] = [];

    // Start all available providers
    if (this.googleHealthcare) {
      promises.push(this.googleHealthcare.analyzeImage(request).catch(() => null));
    }
    if (this.azureHealthBot) {
      promises.push(this.azureHealthBot.analyzeImage(request).catch(() => null));
    }
    if (this.watsonHealth) {
      promises.push(this.watsonHealth.analyzeImage(request).catch(() => null));
    }

    // Wait for all results
    const allResults = await Promise.allSettled(promises);
    allResults.forEach(result => {
      if (result.status === 'fulfilled' && result.value) {
        results.push(result.value);
      }
    });

    if (results.length === 0) {
      throw new Error('No cloud providers returned results');
    }

    // Calculate consensus
    const consensus = this.calculateConsensus(results);
    const agreement = this.calculateAgreement(results);

    return {
      consensus,
      individualResults: results,
      agreement
    };
  }

  /**
   * Symptom assessment using Azure Health Bot or Watson
   */
  async assessSymptoms(symptoms: string[], patientContext?: any): Promise<SymptomAssessmentResult> {
    // Try Azure Health Bot first (specialized in symptom assessment)
    if (this.azureHealthBot) {
      try {
        return await this.azureHealthBot.assessSymptoms(symptoms, patientContext);
      } catch (error) {
        console.warn('Azure Health Bot symptom assessment failed:', error);
      }
    }

    // Fallback to Watson Health
    if (this.watsonHealth) {
      try {
        return await this.watsonHealth.assessSymptoms(symptoms, patientContext);
      } catch (error) {
        console.warn('Watson Health symptom assessment failed:', error);
      }
    }

    throw new Error('No symptom assessment providers available');
  }

  /**
   * Get clinical insights from Watson Health
   */
  async getClinicalInsights(patientData: any): Promise<ClinicalInsightResult> {
    if (!this.watsonHealth) {
      throw new Error('Watson Health not configured for clinical insights');
    }

    return await this.watsonHealth.getClinicalInsights(patientData);
  }

  /**
   * Emergency triage using Azure Health Bot
   */
  async triageEmergency(symptoms: string[]): Promise<{
    isEmergency: boolean;
    urgency: 'routine' | 'soon' | 'urgent' | 'emergency';
    recommendations: string[];
  }> {
    if (!this.azureHealthBot) {
      throw new Error('Azure Health Bot not configured for emergency triage');
    }

    return await this.azureHealthBot.triageEmergency(symptoms);
  }

  /**
   * Get medical advice from Azure Health Bot
   */
  async getMedicalAdvice(query: string, patientContext?: any): Promise<string[]> {
    if (!this.azureHealthBot) {
      throw new Error('Azure Health Bot not configured for medical advice');
    }

    return await this.azureHealthBot.getMedicalAdvice(query, patientContext);
  }

  /**
   * Get treatment recommendations from Watson Health
   */
  async getTreatmentRecommendations(diagnosis: string, patientContext?: any) {
    if (!this.watsonHealth) {
      throw new Error('Watson Health not configured for treatment recommendations');
    }

    return await this.watsonHealth.getTreatmentRecommendations(diagnosis, patientContext);
  }

  // Private helper methods
  private async tryPrimaryProvider(request: CloudHealthcareRequest): Promise<CloudAnalysisResult | null> {
    switch (CLOUD_PROVIDERS_CONFIG.PRIMARY_PROVIDER) {
      case 'google':
        return this.googleHealthcare ? await this.googleHealthcare.analyzeImage(request) : null;
      case 'azure':
        return this.azureHealthBot ? await this.azureHealthBot.analyzeImage(request) : null;
      case 'watson':
        return this.watsonHealth ? await this.watsonHealth.analyzeImage(request) : null;
      default:
        return null;
    }
  }

  private async tryFallbackProviders(request: CloudHealthcareRequest): Promise<CloudAnalysisResult | null> {
    const providers = [
      { name: 'google', service: this.googleHealthcare },
      { name: 'azure', service: this.azureHealthBot },
      { name: 'watson', service: this.watsonHealth }
    ];

    for (const provider of providers) {
      if (provider.service && provider.name !== CLOUD_PROVIDERS_CONFIG.PRIMARY_PROVIDER) {
        try {
          return await provider.service.analyzeImage(request);
        } catch (error) {
          console.warn(`${provider.name} fallback failed:`, error);
        }
      }
    }

    return null;
  }

  private calculateConsensus(results: CloudAnalysisResult[]): CloudAnalysisResult {
    if (results.length === 1) {
      return results[0];
    }

    // Calculate average confidence
    const avgConfidence = results.reduce((sum, r) => sum + r.confidence, 0) / results.length;

    // Get most common prediction
    const predictions = results.map(r => r.prediction);
    const predictionCounts = predictions.reduce((acc, pred) => {
      acc[pred] = (acc[pred] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    const consensusPrediction = Object.entries(predictionCounts)
      .sort(([,a], [,b]) => b - a)[0][0];

    // Merge findings and recommendations
    const allFindings = results.flatMap(r => r.findings);
    const allRecommendations = results.flatMap(r => r.recommendations);

    // Calculate risk level based on consensus
    const riskLevels = results.map(r => r.riskLevel);
    const consensusRiskLevel = this.calculateConsensusRiskLevel(riskLevels);

    return {
      provider: 'google' as const, // Use a valid provider type instead of 'consensus'
      prediction: consensusPrediction,
      confidence: avgConfidence,
      findings: [...new Set(allFindings)], // Remove duplicates
      recommendations: [...new Set(allRecommendations)], // Remove duplicates
      riskLevel: consensusRiskLevel,
      metadata: {
        modelVersion: 'consensus',
        processingTime: Math.max(...results.map(r => r.metadata.processingTime)),
        imageQuality: results[0].metadata.imageQuality
      }
    };
  }

  private calculateAgreement(results: CloudAnalysisResult[]): number {
    if (results.length <= 1) return 1.0;

    const predictions = results.map(r => r.prediction);
    const uniquePredictions = new Set(predictions);

    if (uniquePredictions.size === 1) return 1.0;

    const mostCommonPrediction = predictions.reduce((acc, pred) => {
      acc[pred] = (acc[pred] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    const maxCount = Math.max(...Object.values(mostCommonPrediction));
    return maxCount / predictions.length;
  }

  private calculateConsensusRiskLevel(riskLevels: ('low' | 'medium' | 'high')[]): 'low' | 'medium' | 'high' {
    const riskScores = riskLevels.map(risk => {
      switch (risk) {
        case 'low': return 1;
        case 'medium': return 2;
        case 'high': return 3;
        default: return 2;
      }
    });

    const avgRiskScore = riskScores.reduce((sum, score) => sum + score, 0) / riskScores.length;

    if (avgRiskScore <= 1.5) return 'low';
    if (avgRiskScore <= 2.5) return 'medium';
    return 'high';
  }

  // Utility methods
  getAvailableProviders(): string[] {
    const providers: string[] = [];
    if (this.googleHealthcare) providers.push('google');
    if (this.azureHealthBot) providers.push('azure');
    if (this.watsonHealth) providers.push('watson');
    return providers;
  }

  isProviderAvailable(provider: string): boolean {
    switch (provider) {
      case 'google': return !!this.googleHealthcare;
      case 'azure': return !!this.azureHealthBot;
      case 'watson': return !!this.watsonHealth;
      default: return false;
    }
  }
}

// Export types for use in other services
export type {
  CloudAnalysisResult,
  SymptomAssessmentResult,
  ClinicalInsightResult,
  CloudHealthcareRequest,
  CloudHealthcareConfig,
  ImageType
};
