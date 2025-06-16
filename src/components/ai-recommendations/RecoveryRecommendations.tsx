
import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Button } from '../ui/button';
import { Badge } from '../ui/badge';
import { Progress } from '../ui/progress';
import { Moon, Heart, Battery, AlertTriangle, CheckCircle, TrendingDown } from 'lucide-react';

export const RecoveryRecommendations: React.FC = () => {
  const recoveryMetrics = [
    {
      metric: 'Sleep Quality',
      value: 6.2,
      target: 8.0,
      unit: 'hours',
      status: 'poor',
      trend: 'declining'
    },
    {
      metric: 'Heart Rate Variability',
      value: 42,
      target: 50,
      unit: 'ms',
      status: 'fair',
      trend: 'stable'
    },
    {
      metric: 'Resting Heart Rate',
      value: 68,
      target: 60,
      unit: 'bpm',
      status: 'elevated',
      trend: 'increasing'
    },
    {
      metric: 'Stress Level',
      value: 7.2,
      target: 4.0,
      unit: '/10',
      status: 'high',
      trend: 'increasing'
    }
  ];

  const recoveryInsights = [
    {
      id: 1,
      type: 'sleep_pattern',
      title: 'Poor Sleep Consistency',
      description: 'Your bedtime varies by 2+ hours nightly. Irregular sleep patterns impact recovery and performance.',
      recommendation: 'Set a consistent bedtime of 10:30 PM. Use sleep reminders starting tonight.',
      priority: 'high',
      icon: Moon,
      color: 'text-purple-600',
      bgColor: 'bg-purple-50',
      impact: 'Improve recovery by 30%'
    },
    {
      id: 2,
      type: 'hrv_trend',
      title: 'Declining Heart Rate Variability',
      description: 'Your HRV has dropped 15% this week, indicating accumulated stress or overtraining.',
      recommendation: 'Take a rest day tomorrow. Focus on light stretching and breathing exercises.',
      priority: 'high',
      icon: Heart,
      color: 'text-red-600',
      bgColor: 'bg-red-50',
      impact: 'Prevent overtraining'
    },
    {
      id: 3,
      type: 'stress_management',
      title: 'Elevated Stress Levels',
      description: 'Chronic stress impairs recovery and can lead to burnout. Your stress has been high for 5 days.',
      recommendation: 'Try 10 minutes of meditation or deep breathing. Consider lighter workouts.',
      priority: 'medium',
      icon: AlertTriangle,
      color: 'text-orange-600',
      bgColor: 'bg-orange-50',
      impact: 'Reduce cortisol levels'
    }
  ];

  const recoveryActions = [
    {
      action: 'Optimize Sleep Environment',
      description: 'Cool room (65-68°F), blackout curtains, no screens 1hr before bed',
      timeCommitment: '5 min setup',
      impact: 'High'
    },
    {
      action: 'Progressive Muscle Relaxation',
      description: '10-minute guided relaxation to reduce muscle tension',
      timeCommitment: '10 min',
      impact: 'Medium'
    },
    {
      action: 'Cold Shower/Ice Bath',
      description: '2-3 minutes cold exposure to improve circulation and recovery',
      timeCommitment: '5 min',
      impact: 'High'
    },
    {
      action: 'Breathing Exercise',
      description: '4-7-8 breathing technique to activate parasympathetic nervous system',
      timeCommitment: '3 min',
      impact: 'Medium'
    }
  ];

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'poor': return 'bg-red-500';
      case 'fair': return 'bg-yellow-500';
      case 'good': return 'bg-green-500';
      case 'excellent': return 'bg-green-600';
      case 'high': return 'bg-red-500';
      case 'elevated': return 'bg-orange-500';
      default: return 'bg-gray-500';
    }
  };

  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case 'declining':
      case 'increasing':
        return <TrendingDown className="w-3 h-3 text-red-500" />;
      case 'improving':
        return <TrendingDown className="w-3 h-3 text-green-500 rotate-180" />;
      default:
        return <div className="w-3 h-3 rounded-full bg-gray-400"></div>;
    }
  };

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'high': return 'border-l-red-500';
      case 'medium': return 'border-l-yellow-500';
      case 'low': return 'border-l-green-500';
      default: return 'border-l-gray-500';
    }
  };

  return (
    <div className="space-y-4">
      {/* Recovery Metrics Overview */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Recovery Metrics</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {recoveryMetrics.map((metric) => (
              <div key={metric.metric}>
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center space-x-2">
                    <span className="font-medium text-sm">{metric.metric}</span>
                    {getTrendIcon(metric.trend)}
                  </div>
                  <span className="text-sm text-gray-600">
                    {metric.value}{metric.unit} / {metric.target}{metric.unit}
                  </span>
                </div>
                <Progress 
                  value={(metric.value / metric.target) * 100} 
                  className={`h-2 [&>div]:${getStatusColor(metric.status)}`}
                />
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Recovery Insights */}
      <div className="space-y-3">
        {recoveryInsights.map((insight) => {
          const IconComponent = insight.icon;
          
          return (
            <Card key={insight.id} className={`border-l-4 ${getPriorityColor(insight.priority)}`}>
              <CardContent className="p-4">
                <div className="flex items-start space-x-3">
                  <div className={`p-2 rounded-lg ${insight.bgColor}`}>
                    <IconComponent className={`w-4 h-4 ${insight.color}`} />
                  </div>
                  <div className="flex-1">
                    <div className="flex items-start justify-between mb-2">
                      <h4 className="font-medium">{insight.title}</h4>
                      <Badge variant={insight.priority === 'high' ? 'destructive' : 'default'} className="text-xs">
                        {insight.priority}
                      </Badge>
                    </div>
                    <p className="text-sm text-gray-600 mb-2">{insight.description}</p>
                    <div className="bg-blue-50 p-2 rounded text-sm mb-2">
                      <strong>Recommendation:</strong> {insight.recommendation}
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-green-600 font-medium">{insight.impact}</span>
                      <Button size="sm">
                        Apply
                      </Button>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          );
        })}
      </div>

      {/* Recovery Actions */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Recommended Recovery Actions</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {recoveryActions.map((action, index) => (
              <div key={index} className="p-3 bg-gray-50 rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <h4 className="font-medium text-sm">{action.action}</h4>
                  <div className="flex items-center space-x-2">
                    <Badge variant="outline" className="text-xs">
                      {action.impact} Impact
                    </Badge>
                    <Badge variant="secondary" className="text-xs">
                      {action.timeCommitment}
                    </Badge>
                  </div>
                </div>
                <p className="text-sm text-gray-600 mb-2">{action.description}</p>
                <div className="flex gap-2">
                  <Button variant="outline" size="sm">
                    Learn More
                  </Button>
                  <Button size="sm">
                    Start Now
                  </Button>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Sleep Optimization */}
      <Card className="bg-gradient-to-r from-purple-50 to-blue-50">
        <CardHeader>
          <CardTitle className="text-base flex items-center">
            <Moon className="w-4 h-4 mr-2" />
            Sleep Optimization Plan
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            <div className="flex items-center justify-between p-2 bg-white rounded">
              <span className="text-sm">Bedtime reminder</span>
              <span className="text-xs text-gray-500">10:00 PM</span>
            </div>
            <div className="flex items-center justify-between p-2 bg-white rounded">
              <span className="text-sm">No screens</span>
              <span className="text-xs text-gray-500">9:30 PM</span>
            </div>
            <div className="flex items-center justify-between p-2 bg-white rounded">
              <span className="text-sm">Reading time</span>
              <span className="text-xs text-gray-500">9:30-10:00 PM</span>
            </div>
            <div className="flex items-center justify-between p-2 bg-white rounded">
              <span className="text-sm">Wake up target</span>
              <span className="text-xs text-gray-500">6:30 AM</span>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
