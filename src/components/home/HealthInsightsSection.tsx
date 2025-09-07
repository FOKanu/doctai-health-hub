import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { TrendingUp, TrendingDown, AlertTriangle, CheckCircle, Brain } from 'lucide-react';
import { useHealthData } from '@/contexts/HealthDataContext';

interface Insight {
  id: string;
  type: 'positive' | 'negative' | 'neutral' | 'warning';
  title: string;
  description: string;
  actionable: boolean;
  priority: 'low' | 'medium' | 'high';
}

export const HealthInsightsSection: React.FC = () => {
  const { metrics, healthScore } = useHealthData();

  // Generate insights based on metrics data
  const generateInsights = (): Insight[] => {
    const insights: Insight[] = [];

    // Check for metrics that are significantly off target
    metrics.forEach(metric => {
      const progress = (metric.value / metric.target) * 100;
      
      if (progress < 60) {
        insights.push({
          id: `low_${metric.id}`,
          type: 'warning',
          title: `${metric.name} Below Target`,
          description: `Your ${metric.name.toLowerCase()} is at only ${Math.round(progress)}% of your goal. Consider increasing your daily activity.`,
          actionable: true,
          priority: 'high'
        });
      } else if (progress >= 100) {
        insights.push({
          id: `good_${metric.id}`,
          type: 'positive',
          title: `${metric.name} Goal Achieved`,
          description: `Excellent work! You've reached your ${metric.name.toLowerCase()} target.`,
          actionable: false,
          priority: 'low'
        });
      }
    });

    // Health score insights
    if (healthScore >= 85) {
      insights.push({
        id: 'health_score_excellent',
        type: 'positive',
        title: 'Excellent Health Score',
        description: 'Your overall health metrics are performing exceptionally well. Keep up the great work!',
        actionable: false,
        priority: 'low'
      });
    } else if (healthScore < 70) {
      insights.push({
        id: 'health_score_needs_attention',
        type: 'warning',
        title: 'Health Score Needs Attention',
        description: 'Consider focusing on the metrics that are below target to improve your overall health score.',
        actionable: true,
        priority: 'high'
      });
    }

    // Add some trend-based insights
    const improvingMetrics = metrics.filter(m => m.trend === 'up');
    const decliningMetrics = metrics.filter(m => m.trend === 'down');

    if (improvingMetrics.length > 0) {
      insights.push({
        id: 'positive_trends',
        type: 'positive',
        title: 'Positive Health Trends',
        description: `${improvingMetrics.length} of your health metrics are trending upward this week.`,
        actionable: false,
        priority: 'medium'
      });
    }

    if (decliningMetrics.length > 1) {
      insights.push({
        id: 'declining_trends',
        type: 'negative',
        title: 'Multiple Declining Metrics',
        description: `${decliningMetrics.length} metrics are trending downward. Consider reviewing your routine.`,
        actionable: true,
        priority: 'high'
      });
    }

    return insights.sort((a, b) => {
      const priorities = { high: 3, medium: 2, low: 1 };
      return priorities[b.priority] - priorities[a.priority];
    }).slice(0, 4); // Show top 4 insights
  };

  const insights = generateInsights();

  const getInsightIcon = (type: string) => {
    switch (type) {
      case 'positive': return <CheckCircle className="w-5 h-5 text-green-500" />;
      case 'negative': return <TrendingDown className="w-5 h-5 text-red-500" />;
      case 'warning': return <AlertTriangle className="w-5 h-5 text-yellow-500" />;
      default: return <Brain className="w-5 h-5 text-blue-500" />;
    }
  };

  const getBadgeVariant = (priority: string) => {
    switch (priority) {
      case 'high': return 'destructive';
      case 'medium': return 'default';
      default: return 'secondary';
    }
  };

  if (insights.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Brain className="w-5 h-5" />
            <span>AI Health Insights</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8 text-muted-foreground">
            <Brain className="w-12 h-12 mx-auto mb-4 opacity-50" />
            <p>No insights available yet. Keep tracking your metrics to get personalized recommendations.</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center space-x-2">
          <Brain className="w-5 h-5" />
          <span>AI Health Insights</span>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {insights.map((insight) => (
            <div 
              key={insight.id}
              className="flex items-start space-x-3 p-4 rounded-lg border bg-muted/50"
            >
              {getInsightIcon(insight.type)}
              <div className="flex-1 space-y-2">
                <div className="flex items-center justify-between">
                  <h4 className="font-medium text-foreground">{insight.title}</h4>
                  <Badge variant={getBadgeVariant(insight.priority)}>
                    {insight.priority}
                  </Badge>
                </div>
                <p className="text-sm text-muted-foreground">{insight.description}</p>
                {insight.actionable && (
                  <div className="flex items-center space-x-2 text-xs text-primary">
                    <TrendingUp className="w-3 h-3" />
                    <span>Action recommended</span>
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
};