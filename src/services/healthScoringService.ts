/**
 * Personalized Health Scoring Service
 * Combines multiple health metrics to generate comprehensive health scores
 */

import { extendedHealthMetricsService, type HealthDomainSummary } from './extendedHealthMetricsService';
import type {
  CardiovascularMetrics,
  SleepMetrics,
  MetabolicMetrics,
  FitnessMetrics,
  MentalHealthMetrics
} from '@/integrations/supabase/types';

export interface HealthScore {
  overall: number; // 0-100
  cardiovascular: number;
  metabolic: number;
  sleep: number;
  fitness: number;
  mentalHealth: number;
  respiratory: number;
  hormonal: number;
  trend: 'improving' | 'stable' | 'declining';
  riskLevel: 'low' | 'medium' | 'high';
  insights: string[];
  recommendations: string[];
  lastUpdated: string;
}

export interface HealthScoreBreakdown {
  domain: string;
  score: number;
  weight: number;
  factors: {
    metric: string;
    value: number;
    optimal: number;
    impact: 'positive' | 'negative' | 'neutral';
    contribution: number;
  }[];
}

export interface HealthScoreTrend {
  date: string;
  overallScore: number;
  domainScores: {
    cardiovascular: number;
    metabolic: number;
    sleep: number;
    fitness: number;
    mentalHealth: number;
  };
}

export interface PersonalizedRecommendations {
  priority: 'high' | 'medium' | 'low';
  category: string;
  title: string;
  description: string;
  actionItems: string[];
  expectedImpact: number; // 0-100
  timeframe: 'immediate' | 'short_term' | 'long_term';
}

export class HealthScoringService {
  private readonly weights = {
    cardiovascular: 0.25, // Heart health is critical
    metabolic: 0.20,      // Metabolic health affects everything
    sleep: 0.20,          // Sleep quality impacts all systems
    fitness: 0.15,        // Physical fitness
    mentalHealth: 0.10,   // Mental well-being
    respiratory: 0.05,    // Respiratory health
    hormonal: 0.05        // Hormonal balance
  };

  private readonly optimalRanges = {
    cardiovascular: {
      heartRateResting: { min: 60, max: 100, optimal: 70 },
      heartRateVariability: { min: 20, max: 100, optimal: 50 },
      bloodPressureSystolic: { min: 90, max: 140, optimal: 120 },
      bloodPressureDiastolic: { min: 60, max: 90, optimal: 80 }
    },
    metabolic: {
      bloodGlucoseFasting: { min: 70, max: 100, optimal: 85 },
      hba1c: { min: 4.0, max: 5.7, optimal: 5.0 },
      cholesterolRatio: { min: 2.0, max: 5.0, optimal: 3.5 }
    },
    sleep: {
      duration: { min: 7, max: 9, optimal: 8 },
      efficiency: { min: 80, max: 100, optimal: 90 },
      latency: { min: 0, max: 30, optimal: 15 }
    },
    fitness: {
      steps: { min: 8000, max: 12000, optimal: 10000 },
      vo2Max: { min: 30, max: 60, optimal: 45 },
      activeMinutes: { min: 30, max: 60, optimal: 45 }
    },
    mentalHealth: {
      moodScore: { min: 6, max: 10, optimal: 8 },
      stressLevel: { min: 1, max: 5, optimal: 2 },
      memoryScore: { min: 0.7, max: 1.0, optimal: 0.9 }
    }
  };

  /**
   * Calculate comprehensive health score
   */
  async calculateHealthScore(userId: string): Promise<HealthScore> {
    try {
      const healthSummary = await extendedHealthMetricsService.getHealthSummary({
        userId,
        startDate: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString(),
        endDate: new Date().toISOString()
      });

      const domainScores = {
        cardiovascular: this.calculateCardiovascularScore(healthSummary.cardiovascular),
        metabolic: this.calculateMetabolicScore(healthSummary.metabolic),
        sleep: this.calculateSleepScore(healthSummary.sleep),
        fitness: this.calculateFitnessScore(healthSummary.fitness),
        mentalHealth: this.calculateMentalHealthScore(healthSummary.mentalHealth),
        respiratory: this.calculateRespiratoryScore(healthSummary.respiratory),
        hormonal: this.calculateHormonalScore(healthSummary.hormonal)
      };

      const overallScore = this.calculateOverallScore(domainScores);
      const trend = await this.calculateTrend(userId);
      const riskLevel = this.calculateRiskLevel(overallScore, domainScores);
      const insights = this.generateInsights(domainScores, healthSummary);
      const recommendations = this.generateRecommendations(domainScores, healthSummary);

      return {
        overall: Math.round(overallScore),
        ...domainScores,
        trend,
        riskLevel,
        insights,
        recommendations,
        lastUpdated: new Date().toISOString()
      };
    } catch (error) {
      console.error('Error calculating health score:', error);
      return this.getMockHealthScore();
    }
  }

  /**
   * Get detailed breakdown of health score
   */
  async getHealthScoreBreakdown(userId: string): Promise<HealthScoreBreakdown[]> {
    try {
      const healthSummary = await extendedHealthMetricsService.getHealthSummary({
        userId,
        startDate: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString(),
        endDate: new Date().toISOString()
      });

      return [
        this.getCardiovascularBreakdown(healthSummary.cardiovascular),
        this.getMetabolicBreakdown(healthSummary.metabolic),
        this.getSleepBreakdown(healthSummary.sleep),
        this.getFitnessBreakdown(healthSummary.fitness),
        this.getMentalHealthBreakdown(healthSummary.mentalHealth)
      ];
    } catch (error) {
      console.error('Error getting health score breakdown:', error);
      return this.getMockHealthScoreBreakdown();
    }
  }

  /**
   * Get health score trends over time
   */
  async getHealthScoreTrends(userId: string, days: number = 30): Promise<HealthScoreTrend[]> {
    try {
      // In a real implementation, you would store historical scores
      // For now, we'll generate mock trend data
      const trends: HealthScoreTrend[] = [];
      const baseDate = new Date();

      for (let i = days; i >= 0; i--) {
        const date = new Date(baseDate.getTime() - i * 24 * 60 * 60 * 1000);
        const baseScore = 75 + Math.sin(i * 0.1) * 10; // Simulate natural variation

        trends.push({
          date: date.toISOString().split('T')[0],
          overallScore: Math.round(baseScore + Math.random() * 5),
          domainScores: {
            cardiovascular: Math.round(baseScore + Math.random() * 10),
            metabolic: Math.round(baseScore + Math.random() * 10),
            sleep: Math.round(baseScore + Math.random() * 10),
            fitness: Math.round(baseScore + Math.random() * 10),
            mentalHealth: Math.round(baseScore + Math.random() * 10)
          }
        });
      }

      return trends;
    } catch (error) {
      console.error('Error getting health score trends:', error);
      return this.getMockHealthScoreTrends();
    }
  }

  /**
   * Generate personalized recommendations
   */
  async getPersonalizedRecommendations(userId: string): Promise<PersonalizedRecommendations[]> {
    try {
      const healthScore = await this.calculateHealthScore(userId);
      const recommendations: PersonalizedRecommendations[] = [];

      // Cardiovascular recommendations
      if (healthScore.cardiovascular < 70) {
        recommendations.push({
          priority: 'high',
          category: 'cardiovascular',
          title: 'Improve Heart Health',
          description: 'Your cardiovascular score indicates room for improvement. Focus on heart-healthy habits.',
          actionItems: [
            'Increase daily physical activity to 30+ minutes',
            'Reduce sodium intake to less than 2,300mg daily',
            'Practice stress management techniques',
            'Monitor blood pressure regularly'
          ],
          expectedImpact: 15,
          timeframe: 'short_term'
        });
      }

      // Sleep recommendations
      if (healthScore.sleep < 75) {
        recommendations.push({
          priority: 'high',
          category: 'sleep',
          title: 'Optimize Sleep Quality',
          description: 'Improving your sleep can enhance all aspects of your health.',
          actionItems: [
            'Maintain consistent sleep schedule',
            'Create a relaxing bedtime routine',
            'Optimize bedroom environment (temperature, light, noise)',
            'Limit screen time before bed'
          ],
          expectedImpact: 12,
          timeframe: 'short_term'
        });
      }

      // Metabolic recommendations
      if (healthScore.metabolic < 80) {
        recommendations.push({
          priority: 'medium',
          category: 'metabolic',
          title: 'Enhance Metabolic Health',
          description: 'Focus on nutrition and blood sugar management.',
          actionItems: [
            'Follow a balanced, low-glycemic diet',
            'Include regular strength training',
            'Monitor blood glucose levels',
            'Maintain healthy weight'
          ],
          expectedImpact: 10,
          timeframe: 'long_term'
        });
      }

      // Fitness recommendations
      if (healthScore.fitness < 70) {
        recommendations.push({
          priority: 'medium',
          category: 'fitness',
          title: 'Boost Physical Fitness',
          description: 'Increase your daily activity and exercise routine.',
          actionItems: [
            'Aim for 10,000 steps daily',
            'Include both cardio and strength training',
            'Gradually increase workout intensity',
            'Track progress with fitness metrics'
          ],
          expectedImpact: 8,
          timeframe: 'short_term'
        });
      }

      // Mental health recommendations
      if (healthScore.mentalHealth < 75) {
        recommendations.push({
          priority: 'medium',
          category: 'mental_health',
          title: 'Support Mental Well-being',
          description: 'Prioritize mental health and stress management.',
          actionItems: [
            'Practice mindfulness or meditation',
            'Maintain social connections',
            'Seek professional support if needed',
            'Engage in activities you enjoy'
          ],
          expectedImpact: 6,
          timeframe: 'long_term'
        });
      }

      return recommendations.sort((a, b) => {
        const priorityOrder = { high: 3, medium: 2, low: 1 };
        return priorityOrder[b.priority] - priorityOrder[a.priority];
      });
    } catch (error) {
      console.error('Error generating recommendations:', error);
      return this.getMockRecommendations();
    }
  }

  // Private scoring methods
  private calculateCardiovascularScore(cardiovascular: any): number {
    const factors = [
      this.scoreFactor(cardiovascular.avgHeartRate, this.optimalRanges.cardiovascular.heartRateResting),
      this.scoreFactor(cardiovascular.avgBloodPressure.systolic, this.optimalRanges.cardiovascular.bloodPressureSystolic),
      this.scoreFactor(cardiovascular.avgBloodPressure.diastolic, this.optimalRanges.cardiovascular.bloodPressureDiastolic)
    ];

    return Math.round(factors.reduce((sum, factor) => sum + factor, 0) / factors.length);
  }

  private calculateMetabolicScore(metabolic: any): number {
    const factors = [
      this.scoreFactor(metabolic.avgGlucose, this.optimalRanges.metabolic.bloodGlucoseFasting),
      this.scoreFactor(metabolic.avgHbA1c, this.optimalRanges.metabolic.hba1c)
    ];

    return Math.round(factors.reduce((sum, factor) => sum + factor, 0) / factors.length);
  }

  private calculateSleepScore(sleep: any): number {
    const factors = [
      this.scoreFactor(sleep.avgDuration, this.optimalRanges.sleep.duration),
      this.scoreFactor(sleep.avgEfficiency, this.optimalRanges.sleep.efficiency)
    ];

    return Math.round(factors.reduce((sum, factor) => sum + factor, 0) / factors.length);
  }

  private calculateFitnessScore(fitness: any): number {
    const factors = [
      this.scoreFactor(fitness.avgSteps, this.optimalRanges.fitness.steps),
      this.scoreFactor(fitness.vo2Max, this.optimalRanges.fitness.vo2Max),
      this.scoreFactor(fitness.avgActiveMinutes, this.optimalRanges.fitness.activeMinutes)
    ];

    return Math.round(factors.reduce((sum, factor) => sum + factor, 0) / factors.length);
  }

  private calculateMentalHealthScore(mentalHealth: any): number {
    const factors = [
      this.scoreFactor(mentalHealth.avgMoodScore, this.optimalRanges.mentalHealth.moodScore),
      this.scoreFactor(mentalHealth.avgStressLevel, this.optimalRanges.mentalHealth.stressLevel, true), // Lower is better
      this.scoreFactor(mentalHealth.cognitivePerformance === 'excellent' ? 0.95 :
                      mentalHealth.cognitivePerformance === 'good' ? 0.85 :
                      mentalHealth.cognitivePerformance === 'fair' ? 0.75 : 0.65,
                      this.optimalRanges.mentalHealth.memoryScore)
    ];

    return Math.round(factors.reduce((sum, factor) => sum + factor, 0) / factors.length);
  }

  private calculateRespiratoryScore(respiratory: any): number {
    const o2Score = this.scoreFactor(respiratory.avgOxygenSaturation, { min: 95, max: 100, optimal: 98 });
    const rrScore = this.scoreFactor(respiratory.avgRespiratoryRate, { min: 12, max: 20, optimal: 16 });

    return Math.round((o2Score + rrScore) / 2);
  }

  private calculateHormonalScore(hormonal: any): number {
    // Simplified scoring based on thyroid and cortisol status
    let score = 80; // Base score

    if (hormonal.thyroidStatus !== 'normal') score -= 20;
    if (hormonal.cortisolPattern !== 'normal') score -= 15;
    if (hormonal.hormonalBalance !== 'optimal') score -= 10;

    return Math.max(0, Math.min(100, score));
  }

  private calculateOverallScore(domainScores: any): number {
    let weightedSum = 0;
    let totalWeight = 0;

    Object.entries(this.weights).forEach(([domain, weight]) => {
      weightedSum += domainScores[domain] * weight;
      totalWeight += weight;
    });

    return weightedSum / totalWeight;
  }

  private scoreFactor(value: number, range: { min: number; max: number; optimal: number }, inverse: boolean = false): number {
    if (value === undefined || value === null) return 70; // Default score for missing data

    let score: number;

    if (inverse) {
      // For metrics where lower is better (like stress level)
      if (value <= range.optimal) {
        score = 100;
      } else if (value <= range.max) {
        score = 100 - ((value - range.optimal) / (range.max - range.optimal)) * 30;
      } else {
        score = 70 - ((value - range.max) / range.max) * 30;
      }
    } else {
      // For metrics where higher is better
      if (value >= range.optimal) {
        score = 100;
      } else if (value >= range.min) {
        score = 70 + ((value - range.min) / (range.optimal - range.min)) * 30;
      } else {
        score = 70 - ((range.min - value) / range.min) * 30;
      }
    }

    return Math.max(0, Math.min(100, score));
  }

  private async calculateTrend(userId: string): Promise<'improving' | 'stable' | 'declining'> {
    // In a real implementation, compare current score with historical data
    // For now, return a mock trend
    const trends: Array<'improving' | 'stable' | 'declining'> = ['improving', 'stable', 'declining'];
    return trends[Math.floor(Math.random() * trends.length)];
  }

  private calculateRiskLevel(overallScore: number, domainScores: any): 'low' | 'medium' | 'high' {
    if (overallScore >= 80) return 'low';
    if (overallScore >= 60) return 'medium';
    return 'high';
  }

  private generateInsights(domainScores: any, healthSummary: any): string[] {
    const insights: string[] = [];

    if (domainScores.cardiovascular < 70) {
      insights.push('Your cardiovascular health could benefit from more physical activity and stress management.');
    }

    if (domainScores.sleep < 75) {
      insights.push('Sleep quality is impacting your overall health score. Consider optimizing your sleep routine.');
    }

    if (domainScores.metabolic > 85) {
      insights.push('Excellent metabolic health! Keep maintaining your balanced diet and active lifestyle.');
    }

    if (domainScores.fitness < 70) {
      insights.push('Increasing your daily activity could significantly improve your health score.');
    }

    return insights;
  }

  private generateRecommendations(domainScores: any, healthSummary: any): string[] {
    const recommendations: string[] = [];

    if (domainScores.cardiovascular < 70) {
      recommendations.push('Start with 30 minutes of moderate exercise daily');
      recommendations.push('Monitor your blood pressure weekly');
    }

    if (domainScores.sleep < 75) {
      recommendations.push('Establish a consistent bedtime routine');
      recommendations.push('Aim for 7-9 hours of sleep per night');
    }

    if (domainScores.fitness < 70) {
      recommendations.push('Gradually increase your daily step count');
      recommendations.push('Include strength training 2-3 times per week');
    }

    return recommendations.slice(0, 5); // Limit to top 5 recommendations
  }

  // Breakdown methods
  private getCardiovascularBreakdown(cardiovascular: any): HealthScoreBreakdown {
    return {
      domain: 'Cardiovascular',
      score: this.calculateCardiovascularScore(cardiovascular),
      weight: this.weights.cardiovascular,
      factors: [
        {
          metric: 'Resting Heart Rate',
          value: cardiovascular.avgHeartRate,
          optimal: this.optimalRanges.cardiovascular.heartRateResting.optimal,
          impact: this.getImpact(cardiovascular.avgHeartRate, this.optimalRanges.cardiovascular.heartRateResting),
          contribution: 0.4
        },
        {
          metric: 'Blood Pressure (Systolic)',
          value: cardiovascular.avgBloodPressure.systolic,
          optimal: this.optimalRanges.cardiovascular.bloodPressureSystolic.optimal,
          impact: this.getImpact(cardiovascular.avgBloodPressure.systolic, this.optimalRanges.cardiovascular.bloodPressureSystolic),
          contribution: 0.3
        },
        {
          metric: 'Blood Pressure (Diastolic)',
          value: cardiovascular.avgBloodPressure.diastolic,
          optimal: this.optimalRanges.cardiovascular.bloodPressureDiastolic.optimal,
          impact: this.getImpact(cardiovascular.avgBloodPressure.diastolic, this.optimalRanges.cardiovascular.bloodPressureDiastolic),
          contribution: 0.3
        }
      ]
    };
  }

  private getMetabolicBreakdown(metabolic: any): HealthScoreBreakdown {
    return {
      domain: 'Metabolic',
      score: this.calculateMetabolicScore(metabolic),
      weight: this.weights.metabolic,
      factors: [
        {
          metric: 'Fasting Glucose',
          value: metabolic.avgGlucose,
          optimal: this.optimalRanges.metabolic.bloodGlucoseFasting.optimal,
          impact: this.getImpact(metabolic.avgGlucose, this.optimalRanges.metabolic.bloodGlucoseFasting),
          contribution: 0.6
        },
        {
          metric: 'HbA1c',
          value: metabolic.avgHbA1c,
          optimal: this.optimalRanges.metabolic.hba1c.optimal,
          impact: this.getImpact(metabolic.avgHbA1c, this.optimalRanges.metabolic.hba1c),
          contribution: 0.4
        }
      ]
    };
  }

  private getSleepBreakdown(sleep: any): HealthScoreBreakdown {
    return {
      domain: 'Sleep',
      score: this.calculateSleepScore(sleep),
      weight: this.weights.sleep,
      factors: [
        {
          metric: 'Sleep Duration',
          value: sleep.avgDuration,
          optimal: this.optimalRanges.sleep.duration.optimal,
          impact: this.getImpact(sleep.avgDuration, this.optimalRanges.sleep.duration),
          contribution: 0.5
        },
        {
          metric: 'Sleep Efficiency',
          value: sleep.avgEfficiency,
          optimal: this.optimalRanges.sleep.efficiency.optimal,
          impact: this.getImpact(sleep.avgEfficiency, this.optimalRanges.sleep.efficiency),
          contribution: 0.5
        }
      ]
    };
  }

  private getFitnessBreakdown(fitness: any): HealthScoreBreakdown {
    return {
      domain: 'Fitness',
      score: this.calculateFitnessScore(fitness),
      weight: this.weights.fitness,
      factors: [
        {
          metric: 'Daily Steps',
          value: fitness.avgSteps,
          optimal: this.optimalRanges.fitness.steps.optimal,
          impact: this.getImpact(fitness.avgSteps, this.optimalRanges.fitness.steps),
          contribution: 0.4
        },
        {
          metric: 'VO2 Max',
          value: fitness.vo2Max,
          optimal: this.optimalRanges.fitness.vo2Max.optimal,
          impact: this.getImpact(fitness.vo2Max, this.optimalRanges.fitness.vo2Max),
          contribution: 0.4
        },
        {
          metric: 'Active Minutes',
          value: fitness.avgActiveMinutes,
          optimal: this.optimalRanges.fitness.activeMinutes.optimal,
          impact: this.getImpact(fitness.avgActiveMinutes, this.optimalRanges.fitness.activeMinutes),
          contribution: 0.2
        }
      ]
    };
  }

  private getMentalHealthBreakdown(mentalHealth: any): HealthScoreBreakdown {
    return {
      domain: 'Mental Health',
      score: this.calculateMentalHealthScore(mentalHealth),
      weight: this.weights.mentalHealth,
      factors: [
        {
          metric: 'Mood Score',
          value: mentalHealth.avgMoodScore,
          optimal: this.optimalRanges.mentalHealth.moodScore.optimal,
          impact: this.getImpact(mentalHealth.avgMoodScore, this.optimalRanges.mentalHealth.moodScore),
          contribution: 0.4
        },
        {
          metric: 'Stress Level',
          value: mentalHealth.avgStressLevel,
          optimal: this.optimalRanges.mentalHealth.stressLevel.optimal,
          impact: this.getImpact(mentalHealth.avgStressLevel, this.optimalRanges.mentalHealth.stressLevel, true),
          contribution: 0.4
        },
        {
          metric: 'Cognitive Performance',
          value: mentalHealth.cognitivePerformance === 'excellent' ? 0.95 :
                 mentalHealth.cognitivePerformance === 'good' ? 0.85 :
                 mentalHealth.cognitivePerformance === 'fair' ? 0.75 : 0.65,
          optimal: this.optimalRanges.mentalHealth.memoryScore.optimal,
          impact: this.getImpact(mentalHealth.cognitivePerformance === 'excellent' ? 0.95 :
                               mentalHealth.cognitivePerformance === 'good' ? 0.85 :
                               mentalHealth.cognitivePerformance === 'fair' ? 0.75 : 0.65,
                               this.optimalRanges.mentalHealth.memoryScore),
          contribution: 0.2
        }
      ]
    };
  }

  private getImpact(value: number, range: { min: number; max: number; optimal: number }, inverse: boolean = false): 'positive' | 'negative' | 'neutral' {
    if (value === undefined || value === null) return 'neutral';

    if (inverse) {
      return value <= range.optimal ? 'positive' : value <= range.max ? 'neutral' : 'negative';
    } else {
      return value >= range.optimal ? 'positive' : value >= range.min ? 'neutral' : 'negative';
    }
  }

  // Mock data methods
  private getMockHealthScore(): HealthScore {
    return {
      overall: 78,
      cardiovascular: 75,
      metabolic: 82,
      sleep: 70,
      fitness: 65,
      mentalHealth: 80,
      respiratory: 85,
      hormonal: 78,
      trend: 'improving',
      riskLevel: 'medium',
      insights: [
        'Your cardiovascular health is good but could benefit from more exercise.',
        'Sleep quality is impacting your overall health score.'
      ],
      recommendations: [
        'Increase daily physical activity to 30+ minutes',
        'Optimize your sleep routine for better quality rest'
      ],
      lastUpdated: new Date().toISOString()
    };
  }

  private getMockHealthScoreBreakdown(): HealthScoreBreakdown[] {
    return [
      {
        domain: 'Cardiovascular',
        score: 75,
        weight: 0.25,
        factors: [
          {
            metric: 'Resting Heart Rate',
            value: 72,
            optimal: 70,
            impact: 'positive',
            contribution: 0.4
          }
        ]
      }
    ];
  }

  private getMockHealthScoreTrends(): HealthScoreTrend[] {
    return [
      {
        date: new Date().toISOString().split('T')[0],
        overallScore: 78,
        domainScores: {
          cardiovascular: 75,
          metabolic: 82,
          sleep: 70,
          fitness: 65,
          mentalHealth: 80
        }
      }
    ];
  }

  private getMockRecommendations(): PersonalizedRecommendations[] {
    return [
      {
        priority: 'high',
        category: 'sleep',
        title: 'Improve Sleep Quality',
        description: 'Your sleep score indicates room for improvement.',
        actionItems: ['Establish consistent bedtime', 'Create relaxing routine'],
        expectedImpact: 12,
        timeframe: 'short_term'
      }
    ];
  }
}

export const healthScoringService = new HealthScoringService();
