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
  diet: number;
  mental: number;
  trend?: 'improving' | 'stable' | 'declining';
  riskLevel?: 'low' | 'medium' | 'high';
  insights?: string[];
  lastUpdated?: string;
}

export interface HealthScoreBreakdown {
  domain: string;
  score: number;
  weight: number;
  factors: Array<{
    metric: string;
    value: number;
    optimal: number;
    impact: string;
  }>;
}

export interface PersonalizedRecommendations {
  title: string;
  description: string;
  priority: 'high' | 'medium' | 'low';
  actionItems: string[];
  expectedImpact: number;
  timeframe: string;
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
  async calculateHealthScore(userId: string): Promise<HealthScore> {
    // Mock implementation - return just the score object
    return {
      overall: 78,
      cardiovascular: 82,
      metabolic: 75,
      sleep: 68,
      fitness: 85,
      diet: 72,
      mental: 72,
      trend: 'improving',
      riskLevel: 'low',
      insights: [
        'Your cardiovascular health is excellent',
        'Sleep quality could be improved',
        'Fitness levels are above average',
        'Diet adherence needs attention',
        'Consider increasing protein intake'
      ],
      lastUpdated: new Date().toISOString()
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
        sleep: Math.floor(Math.random() * 25) + 60,
        fitness: Math.floor(Math.random() * 20) + 75,
        diet: Math.floor(Math.random() * 25) + 65
      });
    }
    return trends;
  }

  async getHealthScoreBreakdown(userId: string): Promise<HealthScoreBreakdown[]> {
    return [
      {
        domain: 'cardiovascular',
        score: 82,
        weight: 0.25,
        factors: [
          { metric: 'Blood Pressure', value: 120, optimal: 120, impact: 'positive' },
          { metric: 'Heart Rate', value: 70, optimal: 70, impact: 'positive' }
        ]
      },
      {
        domain: 'metabolic',
        score: 75,
        weight: 0.20,
        factors: [
          { metric: 'Blood Sugar', value: 95, optimal: 90, impact: 'neutral' },
          { metric: 'BMI', value: 23.5, optimal: 22, impact: 'positive' }
        ]
      },
      {
        domain: 'sleep',
        score: 68,
        weight: 0.20,
        factors: [
          { metric: 'Sleep Duration', value: 6.5, optimal: 8, impact: 'negative' },
          { metric: 'Sleep Quality', value: 70, optimal: 85, impact: 'neutral' }
        ]
      },
      {
        domain: 'fitness',
        score: 85,
        weight: 0.15,
        factors: [
          { metric: 'Daily Steps', value: 8547, optimal: 10000, impact: 'neutral' },
          { metric: 'Active Minutes', value: 45, optimal: 60, impact: 'neutral' },
          { metric: 'Workout Frequency', value: 4, optimal: 5, impact: 'positive' }
        ]
      },
      {
        domain: 'diet',
        score: 72,
        weight: 0.20,
        factors: [
          { metric: 'Calorie Intake', value: 1650, optimal: 2200, impact: 'negative' },
          { metric: 'Protein Intake', value: 85, optimal: 110, impact: 'negative' },
          { metric: 'Water Intake', value: 1800, optimal: 2500, impact: 'negative' },
          { metric: 'Meal Consistency', value: 3, optimal: 4, impact: 'neutral' }
        ]
      }
    ];
  }

  async getPersonalizedRecommendations(userId: string): Promise<PersonalizedRecommendations[]> {
    return [
      {
        title: 'Improve Sleep Quality',
        description: 'Your sleep metrics show room for improvement. Better sleep will boost your overall health score.',
        priority: 'high',
        actionItems: [
          'Establish a consistent bedtime routine',
          'Limit screen time before bed',
          'Create a comfortable sleep environment'
        ],
        expectedImpact: 8,
        timeframe: 'short_term'
      },
      {
        title: 'Enhance Diet Adherence',
        description: 'Your nutrition tracking shows gaps in calorie and protein intake. Focus on balanced meals.',
        priority: 'high',
        actionItems: [
          'Add protein-rich snacks between meals',
          'Increase water intake by 2-3 glasses daily',
          'Log all meals consistently'
        ],
        expectedImpact: 6,
        timeframe: 'short_term'
      },
      {
        title: 'Maintain Cardiovascular Health',
        description: 'Your cardiovascular metrics are excellent. Keep up the good work!',
        priority: 'medium',
        actionItems: [
          'Continue regular exercise routine',
          'Monitor blood pressure weekly'
        ],
        expectedImpact: 3,
        timeframe: 'long_term'
      },
      {
        title: 'Optimize Fitness Routine',
        description: 'Your fitness levels are good, but you can improve step count and active minutes.',
        priority: 'medium',
        actionItems: [
          'Aim for 10,000 steps daily',
          'Increase active minutes to 60 per day',
          'Add one more workout session per week'
        ],
        expectedImpact: 5,
        timeframe: 'short_term'
      }
    ];
  }
}

export const healthScoringService = new HealthScoringService();
