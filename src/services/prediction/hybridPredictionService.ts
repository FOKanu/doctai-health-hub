/**
 * Hybrid Prediction Service
 * Routes between single-instance CNN analysis and time-series sequential analysis
 * based on input data type and quantity
 */

import { modernPredictionService } from './modernPredictionService';
import { analyzePredictionLegacy } from './legacyPredictionService';
import {
  PredictionResult,
  ModernPredictionResult,
  ImageType,
  DEBUG_PREDICTIONS
} from './types';

// Time-series analysis types
export interface TimeSeriesInput {
  images: File[];
  imageTypes: ImageType[];
  timestamps: string[];
  vitalSigns?: VitalSignsData[];
  userId: string;
}

export interface VitalSignsData {
  heartRate?: number;
  bloodPressure?: { systolic: number; diastolic: number };
  temperature?: number;
  oxygenSaturation?: number;
  respiratoryRate?: number;
  bloodGlucose?: number;
  weight?: number;
  timestamp: string;
}

export interface ProgressionAnalysis {
  sequenceId: string;
  progressionScore: number;
  trend: 'improving' | 'stable' | 'worsening';
  confidence: number;
  findings: string[];
  recommendations: string[];
  riskLevel: 'low' | 'medium' | 'high';
  timeline: {
    baselineDate: string;
    currentStatus: string;
    daysSinceBaseline: number;
  };
}

export interface VitalSignsAnalysis {
  healthScore: number;
  trend: 'improving' | 'stable' | 'declining';
  anomalies: Array<{
    type: 'warning' | 'critical';
    description: string;
    confidence: number;
    timestamp: string;
  }>;
  predictions: {
    nextWeekHealthScore: number;
    riskFactors: string[];
  };
}

export interface HybridAnalysisResult {
  type: 'single' | 'sequence' | 'comprehensive';
  singleResults?: PredictionResult[];
  progressionAnalysis?: ProgressionAnalysis;
  vitalSignsAnalysis?: VitalSignsAnalysis;
  combinedRiskScore: number;
  recommendations: string[];
  metadata: {
    processingTime: number;
    modelsUsed: string[];
    dataQuality: 'high' | 'medium' | 'low';
  };
}

export class HybridPredictionService {
  private apiEndpoint: string;
  private progressionEndpoint: string;
  private vitalSignsEndpoint: string;

  constructor() {
    this.apiEndpoint = import.meta.env?.VITE_ML_API_ENDPOINT || 'http://localhost:8000/api/predict';
    this.progressionEndpoint = import.meta.env?.VITE_PROGRESSION_API_ENDPOINT || 'http://localhost:8000/api/progression';
    this.vitalSignsEndpoint = import.meta.env?.VITE_VITAL_SIGNS_API_ENDPOINT || 'http://localhost:8000/api/vital-signs';
  }

  /**
   * Main routing function that determines analysis type and delegates accordingly
   */
  async routePrediction(
    data: File | File[] | TimeSeriesInput,
    imageType?: ImageType
  ): Promise<HybridAnalysisResult> {
    const startTime = Date.now();

    try {
      // Determine analysis type based on input
      if (this.isSingleInstance(data)) {
        return await this.singleInstanceAnalysis(data as File, imageType);
      } else if (this.isImageSequence(data)) {
        return await this.timeSeriesAnalysis(data as TimeSeriesInput);
      } else {
        throw new Error('Invalid input data format');
      }
    } catch (error) {
      console.error('Hybrid prediction error:', error);
      throw error;
    } finally {
      if (DEBUG_PREDICTIONS) {
        const processingTime = Date.now() - startTime;
        console.log(`🔀 Hybrid analysis completed in ${processingTime}ms`);
      }
    }
  }

  /**
   * Check if input is single instance (single image)
   */
  private isSingleInstance(data: unknown): data is File {
    return data instanceof File;
  }

  /**
   * Check if input is image sequence with time-series data
   */
  private isImageSequence(data: unknown): data is TimeSeriesInput {
    return (
      data &&
      Array.isArray(data.images) &&
      data.images.length > 1 &&
      typeof data.userId === 'string'
    );
  }

  /**
   * Single instance analysis using existing CNN models
   */
  private async singleInstanceAnalysis(
    image: File,
    imageType: ImageType = 'skin_lesion'
  ): Promise<HybridAnalysisResult> {
    if (DEBUG_PREDICTIONS) {
      console.log('🔬 Single instance analysis:', { imageType });
    }

    const imageId = crypto.randomUUID();
    const result = await modernPredictionService.analyzeImage(image, imageType, imageId);

    // Convert to legacy format for compatibility
    const legacyResult: PredictionResult = {
      prediction: result.riskLevel === 'low' ? 'benign' : 'malignant',
      confidence: result.confidence,
      probabilities: {
        benign: result.probabilities[0] || 0.5,
        malignant: result.probabilities[1] || 0.5
      },
      timestamp: result.createdAt,
      imageId: result.imageId,
      metadata: {
        provider: 'hybrid_single',
        modelVersion: result.modelVersion,
        riskLevel: result.riskLevel
      }
    };

    return {
      type: 'single',
      singleResults: [legacyResult],
      combinedRiskScore: this.calculateRiskScore([legacyResult]),
      recommendations: result.recommendations,
      metadata: {
        processingTime: Date.now(),
        modelsUsed: [result.modelName],
        dataQuality: 'high'
      }
    };
  }

  /**
   * Time-series analysis for image sequences and vital signs
   */
  private async timeSeriesAnalysis(data: TimeSeriesInput): Promise<HybridAnalysisResult> {
    if (DEBUG_PREDICTIONS) {
      console.log('📈 Time-series analysis:', {
        imageCount: data.images.length,
        hasVitalSigns: !!data.vitalSigns
      });
    }

    const results: HybridAnalysisResult = {
      type: data.vitalSigns ? 'comprehensive' : 'sequence',
      singleResults: [],
      combinedRiskScore: 0,
      recommendations: [],
      metadata: {
        processingTime: Date.now(),
        modelsUsed: [],
        dataQuality: 'high'
      }
    };

    // Analyze individual images first
    const singleResults: PredictionResult[] = [];
    for (let i = 0; i < data.images.length; i++) {
      const imageType = data.imageTypes[i] || 'skin_lesion';
      const imageId = crypto.randomUUID();

      try {
        const result = await modernPredictionService.analyzeImage(
          data.images[i],
          imageType,
          imageId
        );

        const legacyResult: PredictionResult = {
          prediction: result.riskLevel === 'low' ? 'benign' : 'malignant',
          confidence: result.confidence,
          probabilities: {
            benign: result.probabilities[0] || 0.5,
            malignant: result.probabilities[1] || 0.5
          },
          timestamp: data.timestamps[i] || new Date().toISOString(),
          imageId: result.imageId,
          metadata: {
            provider: 'hybrid_sequence',
            modelVersion: result.modelVersion,
            riskLevel: result.riskLevel,
            sequenceIndex: i
          }
        };

        singleResults.push(legacyResult);
        results.metadata.modelsUsed.push(result.modelName);
      } catch (error) {
        console.error(`Error analyzing image ${i}:`, error);
      }
    }

    results.singleResults = singleResults;

    // Analyze progression if we have multiple images
    if (data.images.length > 1) {
      try {
        results.progressionAnalysis = await this.analyzeProgression(data, singleResults);
        results.metadata.modelsUsed.push('progression_tracker');
      } catch (error) {
        console.error('Error analyzing progression:', error);
      }
    }

    // Analyze vital signs if provided
    if (data.vitalSigns && data.vitalSigns.length > 0) {
      try {
        results.vitalSignsAnalysis = await this.analyzeVitalSigns(data.vitalSigns);
        results.metadata.modelsUsed.push('vital_signs_analyzer');
      } catch (error) {
        console.error('Error analyzing vital signs:', error);
      }
    }

    // Calculate combined risk score
    results.combinedRiskScore = this.calculateCombinedRiskScore(results);

    // Generate comprehensive recommendations
    results.recommendations = this.generateComprehensiveRecommendations(results);

    return results;
  }

  /**
   * Analyze progression from image sequence
   */
  private async analyzeProgression(
    data: TimeSeriesInput,
    singleResults: PredictionResult[]
  ): Promise<ProgressionAnalysis> {
    if (DEBUG_PREDICTIONS) {
      console.log('🔄 Analyzing progression...');
    }

    // Mock progression analysis - in real implementation, this would call the LSTM model
    const progressionScore = this.calculateProgressionScore(singleResults);
    const trend = this.determineTrend(singleResults);
    const confidence = this.calculateProgressionConfidence(singleResults);

    return {
      sequenceId: crypto.randomUUID(),
      progressionScore,
      trend,
      confidence,
      findings: [
        `Analyzed ${data.images.length} images over ${this.calculateTimeSpan(data.timestamps)} days`,
        `Overall progression score: ${(progressionScore * 100).toFixed(1)}%`,
        `Trend direction: ${trend}`
      ],
      recommendations: this.generateProgressionRecommendations(trend, progressionScore),
      riskLevel: this.calculateProgressionRiskLevel(progressionScore, trend),
      timeline: {
        baselineDate: data.timestamps[0] || new Date().toISOString(),
        currentStatus: trend,
        daysSinceBaseline: this.calculateDaysSinceBaseline(data.timestamps[0])
      }
    };
  }

  /**
   * Analyze vital signs data
   */
  private async analyzeVitalSigns(vitalSigns: VitalSignsData[]): Promise<VitalSignsAnalysis> {
    if (DEBUG_PREDICTIONS) {
      console.log('💓 Analyzing vital signs...');
    }

    // Mock vital signs analysis - in real implementation, this would call the Transformer model
    const healthScore = this.calculateHealthScore(vitalSigns);
    const trend = this.determineVitalSignsTrend(vitalSigns);
    const anomalies = this.detectVitalSignsAnomalies(vitalSigns);

    return {
      healthScore,
      trend,
      anomalies,
      predictions: {
        nextWeekHealthScore: this.predictNextWeekHealthScore(healthScore, trend),
        riskFactors: this.identifyRiskFactors(vitalSigns)
      }
    };
  }

  /**
   * Calculate progression score from single results
   */
  private calculateProgressionScore(results: PredictionResult[]): number {
    if (results.length < 2) return 0.5;

    const riskScores = results.map(r => r.probabilities.malignant);
    const trend = this.calculateTrendSlope(riskScores);

    // Normalize to 0-1 range
    return Math.max(0, Math.min(1, 0.5 + trend * 0.5));
  }

  /**
   * Determine trend from progression results
   */
  private determineTrend(results: PredictionResult[]): 'improving' | 'stable' | 'worsening' {
    if (results.length < 2) return 'stable';

    const riskScores = results.map(r => r.probabilities.malignant);
    const slope = this.calculateTrendSlope(riskScores);

    if (slope > 0.1) return 'worsening';
    if (slope < -0.1) return 'improving';
    return 'stable';
  }

  /**
   * Calculate trend slope using linear regression
   */
  private calculateTrendSlope(values: number[]): number {
    const n = values.length;
    const x = Array.from({ length: n }, (_, i) => i);

    const sumX = x.reduce((a, b) => a + b, 0);
    const sumY = values.reduce((a, b) => a + b, 0);
    const sumXY = x.reduce((a, b, i) => a + b * values[i], 0);
    const sumXX = x.reduce((a, b) => a + b * b, 0);

    const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
    return slope;
  }

  /**
   * Calculate progression confidence
   */
  private calculateProgressionConfidence(results: PredictionResult[]): number {
    const confidences = results.map(r => r.confidence);
    return confidences.reduce((a, b) => a + b, 0) / confidences.length;
  }

  /**
   * Calculate progression risk level
   */
  private calculateProgressionRiskLevel(
    score: number,
    trend: 'improving' | 'stable' | 'worsening'
  ): 'low' | 'medium' | 'high' {
    if (trend === 'worsening' && score > 0.7) return 'high';
    if (trend === 'improving' && score < 0.3) return 'low';
    return 'medium';
  }

  /**
   * Calculate health score from vital signs
   */
  private calculateHealthScore(vitalSigns: VitalSignsData[]): number {
    if (vitalSigns.length === 0) return 0.5;

    let totalScore = 0;
    let validMetrics = 0;

    vitalSigns.forEach(vs => {
      if (vs.heartRate) {
        const hrScore = this.scoreHeartRate(vs.heartRate);
        totalScore += hrScore;
        validMetrics++;
      }
      if (vs.temperature) {
        const tempScore = this.scoreTemperature(vs.temperature);
        totalScore += tempScore;
        validMetrics++;
      }
      // Add more vital signs scoring...
    });

    return validMetrics > 0 ? totalScore / validMetrics : 0.5;
  }

  /**
   * Score individual vital signs
   */
  private scoreHeartRate(hr: number): number {
    if (hr >= 60 && hr <= 100) return 1.0;
    if (hr >= 50 && hr <= 110) return 0.8;
    if (hr >= 40 && hr <= 120) return 0.6;
    return 0.3;
  }

  private scoreTemperature(temp: number): number {
    if (temp >= 97.8 && temp <= 99.1) return 1.0;
    if (temp >= 97.0 && temp <= 99.5) return 0.8;
    if (temp >= 96.0 && temp <= 100.0) return 0.6;
    return 0.3;
  }

  /**
   * Determine vital signs trend
   */
  private determineVitalSignsTrend(vitalSigns: VitalSignsData[]): 'improving' | 'stable' | 'declining' {
    if (vitalSigns.length < 2) return 'stable';

    const scores = vitalSigns.map(vs => this.calculateHealthScore([vs]));
    const slope = this.calculateTrendSlope(scores);

    if (slope > 0.05) return 'improving';
    if (slope < -0.05) return 'declining';
    return 'stable';
  }

  /**
   * Detect anomalies in vital signs
   */
  private detectVitalSignsAnomalies(vitalSigns: VitalSignsData[]): Array<{
    type: 'warning' | 'critical';
    description: string;
    confidence: number;
    timestamp: string;
  }> {
    const anomalies: Array<{
      type: 'warning' | 'critical';
      description: string;
      confidence: number;
      timestamp: string;
    }> = [];

    vitalSigns.forEach(vs => {
      if (vs.heartRate && (vs.heartRate < 50 || vs.heartRate > 120)) {
        anomalies.push({
          type: vs.heartRate < 40 || vs.heartRate > 140 ? 'critical' : 'warning',
          description: `Abnormal heart rate: ${vs.heartRate} bpm`,
          confidence: 0.8,
          timestamp: vs.timestamp
        });
      }
      // Add more anomaly detection...
    });

    return anomalies;
  }

  /**
   * Predict next week health score
   */
  private predictNextWeekHealthScore(
    currentScore: number,
    trend: 'improving' | 'stable' | 'declining'
  ): number {
    const trendFactor = trend === 'improving' ? 0.05 : trend === 'declining' ? -0.05 : 0;
    return Math.max(0, Math.min(1, currentScore + trendFactor));
  }

  /**
   * Identify risk factors from vital signs
   */
  private identifyRiskFactors(vitalSigns: VitalSignsData[]): string[] {
    const riskFactors: string[] = [];

    vitalSigns.forEach(vs => {
      if (vs.heartRate && vs.heartRate > 100) {
        riskFactors.push('Elevated heart rate');
      }
      if (vs.temperature && vs.temperature > 99.5) {
        riskFactors.push('Elevated temperature');
      }
    });

    return [...new Set(riskFactors)]; // Remove duplicates
  }

  /**
   * Calculate combined risk score from all analyses
   */
  private calculateCombinedRiskScore(result: HybridAnalysisResult): number {
    let totalScore = 0;
    let weightSum = 0;

    // Single results weight
    if (result.singleResults && result.singleResults.length > 0) {
      const avgRisk = result.singleResults.reduce((sum, r) => sum + r.probabilities.malignant, 0) / result.singleResults.length;
      totalScore += avgRisk * 0.4;
      weightSum += 0.4;
    }

    // Progression analysis weight
    if (result.progressionAnalysis) {
      const progressionRisk = result.progressionAnalysis.progressionScore;
      totalScore += progressionRisk * 0.4;
      weightSum += 0.4;
    }

    // Vital signs analysis weight
    if (result.vitalSignsAnalysis) {
      const vitalSignsRisk = 1 - result.vitalSignsAnalysis.healthScore;
      totalScore += vitalSignsRisk * 0.2;
      weightSum += 0.2;
    }

    return weightSum > 0 ? totalScore / weightSum : 0.5;
  }

  /**
   * Generate comprehensive recommendations
   */
  private generateComprehensiveRecommendations(result: HybridAnalysisResult): string[] {
    const recommendations: string[] = [];

    // Base recommendations from single results
    if (result.singleResults) {
      const highRiskCount = result.singleResults.filter(r => r.probabilities.malignant > 0.7).length;
      if (highRiskCount > 0) {
        recommendations.push(`Immediate medical consultation recommended for ${highRiskCount} concerning findings`);
      }
    }

    // Progression-based recommendations
    if (result.progressionAnalysis) {
      if (result.progressionAnalysis.trend === 'worsening') {
        recommendations.push('Schedule follow-up appointment within 2 weeks');
      } else if (result.progressionAnalysis.trend === 'improving') {
        recommendations.push('Continue current treatment plan');
      }
    }

    // Vital signs-based recommendations
    if (result.vitalSignsAnalysis) {
      if (result.vitalSignsAnalysis.anomalies.length > 0) {
        recommendations.push('Monitor vital signs closely and report any concerning changes');
      }
      if (result.vitalSignsAnalysis.trend === 'declining') {
        recommendations.push('Consider lifestyle modifications and consult healthcare provider');
      }
    }

    // Overall risk-based recommendations
    if (result.combinedRiskScore > 0.7) {
      recommendations.push('High-risk assessment - immediate medical attention advised');
    } else if (result.combinedRiskScore > 0.4) {
      recommendations.push('Moderate risk - regular monitoring recommended');
    } else {
      recommendations.push('Low risk - continue routine health monitoring');
    }

    return recommendations;
  }

  /**
   * Generate progression-specific recommendations
   */
  private generateProgressionRecommendations(
    trend: 'improving' | 'stable' | 'worsening',
    score: number
  ): string[] {
    const recommendations: string[] = [];

    if (trend === 'worsening') {
      recommendations.push('Schedule immediate follow-up with specialist');
      recommendations.push('Consider additional diagnostic testing');
      if (score > 0.8) {
        recommendations.push('High progression risk - urgent medical evaluation needed');
      }
    } else if (trend === 'improving') {
      recommendations.push('Continue current treatment plan');
      recommendations.push('Maintain regular monitoring schedule');
    } else {
      recommendations.push('Continue monitoring for any changes');
      recommendations.push('Schedule routine follow-up in 3 months');
    }

    return recommendations;
  }

  /**
   * Calculate risk score from single results
   */
  private calculateRiskScore(results: PredictionResult[]): number {
    if (results.length === 0) return 0.5;
    return results.reduce((sum, r) => sum + r.probabilities.malignant, 0) / results.length;
  }

  /**
   * Calculate time span between timestamps
   */
  private calculateTimeSpan(timestamps: string[]): number {
    if (timestamps.length < 2) return 0;
    const start = new Date(timestamps[0]);
    const end = new Date(timestamps[timestamps.length - 1]);
    return Math.ceil((end.getTime() - start.getTime()) / (1000 * 60 * 60 * 24));
  }

  /**
   * Calculate days since baseline
   */
  private calculateDaysSinceBaseline(baselineDate: string): number {
    const baseline = new Date(baselineDate);
    const now = new Date();
    return Math.ceil((now.getTime() - baseline.getTime()) / (1000 * 60 * 60 * 24));
  }
}

// Export singleton instance
export const hybridPredictionService = new HybridPredictionService();
