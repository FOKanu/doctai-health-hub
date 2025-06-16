
import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Button } from '../ui/button';
import { Badge } from '../ui/badge';
import { Progress } from '../ui/progress';
import { Apple, TrendingUp, AlertCircle, Lightbulb, Target } from 'lucide-react';

export const DietRecommendations: React.FC = () => {
  const nutritionGaps = [
    {
      nutrient: 'Fiber',
      current: 18,
      target: 25,
      unit: 'g',
      percentage: 72,
      status: 'low',
      foods: ['Broccoli', 'Beans', 'Apples with skin', 'Oats']
    },
    {
      nutrient: 'Protein',
      current: 85,
      target: 150,
      unit: 'g',
      percentage: 57,
      status: 'low',
      foods: ['Greek yogurt', 'Chicken breast', 'Lentils', 'Eggs']
    },
    {
      nutrient: 'Omega-3',
      current: 0.8,
      target: 2.0,
      unit: 'g',
      percentage: 40,
      status: 'very_low',
      foods: ['Salmon', 'Walnuts', 'Chia seeds', 'Flax seeds']
    }
  ];

  const dietInsights = [
    {
      id: 1,
      type: 'post_workout',
      title: 'Post-Workout Nutrition Timing',
      description: 'You completed strength training 45 minutes ago. Your muscle protein synthesis window is still open.',
      recommendation: 'Consume 20-30g protein within the next 15 minutes for optimal recovery',
      urgency: 'high',
      icon: Target,
      timeRemaining: '15 minutes'
    },
    {
      id: 2,
      type: 'macro_balance',
      title: 'Carb Loading Opportunity',
      description: 'You have a leg workout scheduled tomorrow. Your glycogen stores could use topping up.',
      recommendation: 'Add 50-75g extra carbs to dinner tonight (rice, pasta, or sweet potato)',
      urgency: 'medium',
      icon: TrendingUp
    },
    {
      id: 3,
      type: 'hydration',
      title: 'Hydration Strategy',
      description: 'Your water intake has been inconsistent. Dehydration can impact performance and recovery.',
      recommendation: 'Set hourly water reminders. Aim for 500ml in the next 2 hours',
      urgency: 'medium',
      icon: AlertCircle
    }
  ];

  const mealSwaps = [
    {
      current: 'White bread toast',
      suggested: 'Whole grain avocado toast',
      benefit: '+5g fiber, +healthy fats',
      calories: '+50 cal'
    },
    {
      current: 'Regular pasta',
      suggested: 'Lentil pasta',
      benefit: '+12g protein, +8g fiber',
      calories: '+20 cal'
    },
    {
      current: 'Soda',
      suggested: 'Sparkling water with lemon',
      benefit: '-35g sugar, +hydration',
      calories: '-140 cal'
    }
  ];

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'very_low': return 'bg-red-500';
      case 'low': return 'bg-yellow-500';
      case 'good': return 'bg-green-500';
      default: return 'bg-gray-500';
    }
  };

  const getUrgencyColor = (urgency: string) => {
    switch (urgency) {
      case 'high': return 'border-l-red-500';
      case 'medium': return 'border-l-yellow-500';
      case 'low': return 'border-l-green-500';
      default: return 'border-l-gray-500';
    }
  };

  return (
    <div className="space-y-4">
      {/* Nutrition Gaps */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Nutrition Gaps Analysis</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {nutritionGaps.map((gap) => (
              <div key={gap.nutrient}>
                <div className="flex items-center justify-between mb-2">
                  <span className="font-medium">{gap.nutrient}</span>
                  <span className="text-sm text-gray-600">
                    {gap.current}{gap.unit} / {gap.target}{gap.unit}
                  </span>
                </div>
                <Progress 
                  value={gap.percentage} 
                  className={`h-2 mb-2 [&>div]:${getStatusColor(gap.status)}`}
                />
                <div className="text-xs text-gray-600">
                  <strong>Add:</strong> {gap.foods.join(', ')}
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* AI Diet Insights */}
      <div className="space-y-3">
        {dietInsights.map((insight) => {
          const IconComponent = insight.icon;
          
          return (
            <Card key={insight.id} className={`border-l-4 ${getUrgencyColor(insight.urgency)}`}>
              <CardContent className="p-4">
                <div className="flex items-start space-x-3">
                  <div className="p-2 bg-blue-50 rounded-lg">
                    <IconComponent className="w-4 h-4 text-blue-600" />
                  </div>
                  <div className="flex-1">
                    <div className="flex items-start justify-between mb-2">
                      <div>
                        <h4 className="font-medium">{insight.title}</h4>
                        {insight.timeRemaining && (
                          <p className="text-xs text-orange-600 font-medium">
                            ⏰ {insight.timeRemaining}
                          </p>
                        )}
                      </div>
                      <Badge variant={insight.urgency === 'high' ? 'destructive' : 'default'} className="text-xs">
                        {insight.urgency}
                      </Badge>
                    </div>
                    <p className="text-sm text-gray-600 mb-2">{insight.description}</p>
                    <div className="bg-green-50 p-2 rounded text-sm mb-2">
                      <strong>Recommendation:</strong> {insight.recommendation}
                    </div>
                    <Button size="sm">
                      Apply Suggestion
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          );
        })}
      </div>

      {/* Smart Meal Swaps */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base flex items-center">
            <Lightbulb className="w-4 h-4 mr-2" />
            Smart Meal Swaps
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {mealSwaps.map((swap, index) => (
              <div key={index} className="p-3 bg-gray-50 rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <div className="text-sm">
                    <span className="line-through text-gray-500">{swap.current}</span>
                    <span className="mx-2">→</span>
                    <span className="font-medium text-green-700">{swap.suggested}</span>
                  </div>
                  <Button size="sm" variant="outline">
                    Swap
                  </Button>
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-green-600">{swap.benefit}</span>
                  <span className="text-gray-500">{swap.calories}</span>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Meal Timing Optimization */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Meal Timing Optimization</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            <div className="flex items-center justify-between p-3 bg-blue-50 rounded-lg">
              <div>
                <div className="font-medium text-sm">Pre-Workout (30 min before)</div>
                <div className="text-xs text-gray-600">Quick carbs + minimal protein</div>
              </div>
              <Badge variant="outline">Banana + coffee</Badge>
            </div>
            <div className="flex items-center justify-between p-3 bg-green-50 rounded-lg">
              <div>
                <div className="font-medium text-sm">Post-Workout (within 30 min)</div>
                <div className="text-xs text-gray-600">Protein + fast carbs</div>
              </div>
              <Badge variant="outline">Protein shake + fruit</Badge>
            </div>
            <div className="flex items-center justify-between p-3 bg-purple-50 rounded-lg">
              <div>
                <div className="font-medium text-sm">Before Bed (2 hours before)</div>
                <div className="text-xs text-gray-600">Slow protein + minimal carbs</div>
              </div>
              <Badge variant="outline">Greek yogurt + nuts</Badge>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
