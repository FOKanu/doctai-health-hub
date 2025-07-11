/**
 * Personalized Health Scoring Service - Mock Implementation
 * Provides basic health scoring functionality for development
 */

export interface HealthScore {
  overall: number; // 0-100
  cardiovascular: number;
  metabolic: number;
  sleep: number;
  fitness: number;
  mental: number;
  trend?: number;
  riskLevel?: 'low' | 'medium' | 'high';
  insights?: string[];
  lastUpdated?: string;
}

export interface HealthScoreBreakdown {
  domains: {
    cardiovascular: { score: number; details: string[] };
    metabolic: { score: number; details: string[] };
    sleep: { score: number; details: string[] };
    fitness: { score: number; details: string[] };
    mental: { score: number; details: string[] };
  };
}

export interface PersonalizedRecommendations {
  immediate: string[];
  shortTerm: string[];
  longTerm: string[];
}

export interface HealthScoreDetails {
  score: HealthScore;
  insights: string[];
  recommendations: string[];
  trends: {
    direction: 'improving' | 'stable' | 'declining';
    percentage: number;
  };
}

export class HealthScoringService {
  async calculateHealthScore(userId: string): Promise<HealthScoreDetails> {
    // Mock implementation
    return {
      score: {
        overall: 78,
        cardiovascular: 82,
        metabolic: 75,
        sleep: 68,
        fitness: 85,
        mental: 72
      },
      insights: [
        'Your cardiovascular health is excellent',
        'Sleep quality could be improved',
        'Fitness levels are above average'
      ],
      recommendations: [
        'Consider establishing a consistent sleep schedule',
        'Continue your current exercise routine',
        'Monitor stress levels and practice relaxation techniques'
      ],
      trends: {
        direction: 'improving',
        percentage: 12
      }
    };
  }

  async getHealthTrends(userId: string, days: number = 30): Promise<any[]> {
    // Mock trend data
    const trends = [];
    for (let i = days; i >= 0; i--) {
      const date = new Date(Date.now() - i * 24 * 60 * 60 * 1000);
      trends.push({
        date: date.toISOString().split('T')[0],
        overall: Math.floor(Math.random() * 20) + 70,
        cardiovascular: Math.floor(Math.random() * 20) + 75,
        sleep: Math.floor(Math.random() * 25) + 60
      });
    }
    return trends;
  }

  async getHealthScoreBreakdown(userId: string): Promise<HealthScoreBreakdown> {
    return {
      domains: {
        cardiovascular: { 
          score: 82, 
          details: ['Blood pressure within normal range', 'Heart rate variability good'] 
        },
        metabolic: { 
          score: 75, 
          details: ['Blood sugar levels stable', 'BMI in healthy range'] 
        },
        sleep: { 
          score: 68, 
          details: ['Average 6.5 hours sleep', 'Sleep quality could improve'] 
        },
        fitness: { 
          score: 85, 
          details: ['Regular exercise routine', 'Good cardiovascular endurance'] 
        },
        mental: { 
          score: 72, 
          details: ['Moderate stress levels', 'Good overall mood'] 
        }
      }
    };
  }

  async getPersonalizedRecommendations(userId: string): Promise<PersonalizedRecommendations> {
    return {
      immediate: [
        'Ensure 7-8 hours of sleep tonight',
        'Take a 10-minute walk after lunch'
      ],
      shortTerm: [
        'Establish a consistent sleep schedule',
        'Reduce caffeine intake after 2 PM'
      ],
      longTerm: [
        'Implement stress management techniques',
        'Schedule regular health checkups'
      ]
    };
  }
}

export const healthScoringService = new HealthScoringService();