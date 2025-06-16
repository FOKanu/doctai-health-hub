
import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Button } from '../ui/button';
import { Badge } from '../ui/badge';
import { Progress } from '../ui/progress';
import { Dumbbell, TrendingDown, AlertTriangle, CheckCircle, Calendar } from 'lucide-react';

export const WorkoutRecommendations: React.FC = () => {
  const workoutInsights = [
    {
      id: 1,
      type: 'muscle_group_gap',
      title: 'Leg Training Gap Detected',
      description: 'You\'ve skipped leg workouts 3 times this week. This could lead to muscle imbalances.',
      recommendation: 'Schedule 2 leg sessions this week focusing on squats and deadlifts',
      priority: 'high',
      impact: 'Prevent muscle imbalances',
      icon: AlertTriangle,
      color: 'text-red-600',
      bgColor: 'bg-red-50'
    },
    {
      id: 2,
      type: 'performance_trend',
      title: 'Bench Press Plateau',
      description: 'Your bench press hasn\'t improved in 3 weeks. Time to mix things up.',
      recommendation: 'Try incline bench, dumbbell variations, or drop sets',
      priority: 'medium',
      impact: 'Break through plateau',
      icon: TrendingDown,
      color: 'text-orange-600',
      bgColor: 'bg-orange-50'
    },
    {
      id: 3,
      type: 'recovery_optimization',
      title: 'Optimal Recovery Window',
      description: 'Your heart rate variability suggests good recovery. Perfect time for intense training.',
      recommendation: 'Schedule your hardest workout today or tomorrow',
      priority: 'medium',
      impact: 'Maximize gains',
      icon: CheckCircle,
      color: 'text-green-600',
      bgColor: 'bg-green-50'
    }
  ];

  const weeklyProgress = [
    { muscle: 'Chest', sessions: 2, recommended: 2, percentage: 100 },
    { muscle: 'Back', sessions: 2, recommended: 2, percentage: 100 },
    { muscle: 'Shoulders', sessions: 1, recommended: 2, percentage: 50 },
    { muscle: 'Arms', sessions: 3, recommended: 2, percentage: 100 },
    { muscle: 'Legs', sessions: 0, recommended: 2, percentage: 0 },
    { muscle: 'Core', sessions: 1, recommended: 3, percentage: 33 }
  ];

  const suggestedWorkouts = [
    {
      name: 'Intense Leg Day',
      duration: '45 min',
      difficulty: 'High',
      focus: 'Squats, Deadlifts, Lunges',
      reason: 'Address leg training gap'
    },
    {
      name: 'Upper Body Power',
      duration: '40 min',
      difficulty: 'Medium',
      focus: 'Bench variations, Rows',
      reason: 'Break plateau patterns'
    },
    {
      name: 'Core & Stability',
      duration: '30 min',
      difficulty: 'Medium',
      focus: 'Planks, Russian twists',
      reason: 'Improve core strength'
    }
  ];

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
      {/* Performance Insights */}
      <div className="space-y-3">
        {workoutInsights.map((insight) => {
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
                      <strong>AI Suggestion:</strong> {insight.recommendation}
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-green-600 font-medium">Impact: {insight.impact}</span>
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

      {/* Weekly Muscle Group Progress */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Weekly Training Balance</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {weeklyProgress.map((muscle) => (
              <div key={muscle.muscle}>
                <div className="flex items-center justify-between mb-1">
                  <span className="text-sm font-medium">{muscle.muscle}</span>
                  <span className="text-sm text-gray-600">
                    {muscle.sessions}/{muscle.recommended} sessions
                  </span>
                </div>
                <Progress 
                  value={muscle.percentage} 
                  className={`h-2 ${muscle.percentage < 50 ? '[&>div]:bg-red-500' : muscle.percentage < 100 ? '[&>div]:bg-yellow-500' : '[&>div]:bg-green-500'}`}
                />
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Suggested Workouts */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Recommended Workouts</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {suggestedWorkouts.map((workout, index) => (
              <div key={index} className="p-3 bg-gray-50 rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <h4 className="font-medium">{workout.name}</h4>
                  <Badge variant="outline" className="text-xs">
                    {workout.difficulty}
                  </Badge>
                </div>
                <div className="text-sm text-gray-600 mb-2">
                  <div className="flex items-center space-x-4">
                    <span>🕐 {workout.duration}</span>
                    <span>💪 {workout.focus}</span>
                  </div>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-xs text-blue-600 italic">{workout.reason}</span>
                  <div className="flex gap-2">
                    <Button variant="outline" size="sm">
                      Schedule
                    </Button>
                    <Button size="sm">
                      Start Now
                    </Button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
