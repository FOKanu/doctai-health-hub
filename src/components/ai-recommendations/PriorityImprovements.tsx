
import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Button } from '../ui/button';
import { Badge } from '../ui/badge';
import { Progress } from '../ui/progress';
import { Target, TrendingUp, Star, Calendar, CheckCircle } from 'lucide-react';

export const PriorityImprovements: React.FC = () => {
  const priorityAreas = [
    {
      id: 1,
      area: 'Leg Training Consistency',
      currentScore: 25,
      targetScore: 80,
      priority: 1,
      impact: 'High',
      timeframe: '2 weeks',
      description: 'Missing leg workouts creates muscle imbalances and limits overall strength gains',
      actions: [
        'Schedule 2 leg sessions per week',
        'Start with bodyweight squats if needed',
        'Track leg workout completion'
      ],
      category: 'Fitness'
    },
    {
      id: 2,
      area: 'Sleep Quality',
      currentScore: 55,
      targetScore: 85,
      priority: 2,
      impact: 'High',
      timeframe: '1 week',
      description: 'Poor sleep affects recovery, performance, and overall health',
      actions: [
        'Set consistent bedtime (10:30 PM)',
        'Create sleep-friendly environment',
        'Limit screens 1 hour before bed'
      ],
      category: 'Recovery'
    },
    {
      id: 3,
      area: 'Vegetable Intake',
      currentScore: 45,
      targetScore: 90,
      priority: 3,
      impact: 'Medium',
      timeframe: '1 week',
      description: 'Low vegetable intake means missing essential micronutrients and fiber',
      actions: [
        'Add vegetables to each meal',
        'Try new vegetable preparation methods',
        'Track daily vegetable servings'
      ],
      category: 'Nutrition'
    },
    {
      id: 4,
      area: 'Hydration Consistency',
      currentScore: 60,
      targetScore: 90,
      priority: 4,
      impact: 'Medium',
      timeframe: '3 days',
      description: 'Inconsistent hydration affects energy, focus, and workout performance',
      actions: [
        'Set hourly water reminders',
        'Carry water bottle everywhere',
        'Track daily water intake'
      ],
      category: 'Health'
    }
  ];

  const weeklyGoals = [
    {
      goal: 'Complete 2 leg workouts',
      progress: 0,
      target: 2,
      daysLeft: 5
    },
    {
      goal: 'Sleep 7+ hours nightly',
      progress: 3,
      target: 7,
      daysLeft: 4
    },
    {
      goal: 'Eat 5 servings of vegetables daily',
      progress: 2,
      target: 7,
      daysLeft: 4
    }
  ];

  const getCategoryColor = (category: string) => {
    switch (category) {
      case 'Fitness': return 'bg-blue-100 text-blue-800';
      case 'Nutrition': return 'bg-green-100 text-green-800';
      case 'Recovery': return 'bg-purple-100 text-purple-800';
      case 'Health': return 'bg-orange-100 text-orange-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const getPriorityBadge = (priority: number) => {
    switch (priority) {
      case 1: return <Badge variant="destructive" className="text-xs">Priority 1</Badge>;
      case 2: return <Badge variant="default" className="text-xs">Priority 2</Badge>;
      case 3: return <Badge variant="secondary" className="text-xs">Priority 3</Badge>;
      case 4: return <Badge variant="outline" className="text-xs">Priority 4</Badge>;
      default: return null;
    }
  };

  return (
    <div className="space-y-4">
      {/* What to Improve Section */}
      <Card className="border-l-4 border-l-blue-500">
        <CardHeader>
          <CardTitle className="text-base flex items-center">
            <Target className="w-4 h-4 mr-2" />
            What to Improve (Ranked by Impact)
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {priorityAreas.map((area) => (
              <div key={area.id} className="p-4 bg-gray-50 rounded-lg">
                <div className="flex items-start justify-between mb-3">
                  <div className="flex-1">
                    <div className="flex items-center space-x-2 mb-1">
                      <h4 className="font-medium">{area.area}</h4>
                      {getPriorityBadge(area.priority)}
                      <Badge className={`text-xs ${getCategoryColor(area.category)}`}>
                        {area.category}
                      </Badge>
                    </div>
                    <p className="text-sm text-gray-600 mb-2">{area.description}</p>
                  </div>
                </div>

                <div className="mb-3">
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-sm">Progress</span>
                    <span className="text-sm text-gray-600">
                      {area.currentScore}/100 → Target: {area.targetScore}/100
                    </span>
                  </div>
                  <Progress value={area.currentScore} className="h-2 mb-1" />
                  <div className="flex justify-between text-xs text-gray-500">
                    <span>{area.impact} Impact</span>
                    <span>Timeframe: {area.timeframe}</span>
                  </div>
                </div>

                <div className="mb-3">
                  <h5 className="font-medium text-sm mb-2">Action Steps:</h5>
                  <ul className="text-sm text-gray-600 space-y-1">
                    {area.actions.map((action, index) => (
                      <li key={index} className="flex items-start">
                        <CheckCircle className="w-3 h-3 mt-0.5 mr-2 text-green-500" />
                        {action}
                      </li>
                    ))}
                  </ul>
                </div>

                <div className="flex gap-2">
                  <Button size="sm" className="flex-1">
                    Start Improving
                  </Button>
                  <Button variant="outline" size="sm">
                    Learn More
                  </Button>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Weekly Goals Tracker */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base flex items-center">
            <Calendar className="w-4 h-4 mr-2" />
            This Week's Goals
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {weeklyGoals.map((goal, index) => (
              <div key={index} className="p-3 bg-gray-50 rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <span className="font-medium text-sm">{goal.goal}</span>
                  <span className="text-xs text-gray-500">{goal.daysLeft} days left</span>
                </div>
                <div className="flex items-center justify-between mb-1">
                  <span className="text-sm text-gray-600">
                    {goal.progress}/{goal.target} completed
                  </span>
                  <span className="text-xs text-gray-500">
                    {Math.round((goal.progress / goal.target) * 100)}%
                  </span>
                </div>
                <Progress value={(goal.progress / goal.target) * 100} className="h-2" />
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Success Streak */}
      <Card className="bg-gradient-to-r from-green-50 to-blue-50">
        <CardHeader>
          <CardTitle className="text-base flex items-center">
            <Star className="w-4 h-4 mr-2 text-yellow-500" />
            Success Streaks
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">5</div>
              <div className="text-xs text-gray-600">Days tracking meals</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600">3</div>
              <div className="text-xs text-gray-600">Workouts this week</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-purple-600">2</div>
              <div className="text-xs text-gray-600">Good sleep nights</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-orange-600">4</div>
              <div className="text-xs text-gray-600">Hydration goals met</div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
