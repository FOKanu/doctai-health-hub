
import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Button } from '../ui/button';
import { Badge } from '../ui/badge';
import { Lightbulb, TrendingUp, AlertCircle, CheckCircle, Clock, Zap } from 'lucide-react';

export const SmartSuggestions: React.FC = () => {
  const suggestions = [
    {
      id: 1,
      type: 'workout_based',
      title: 'Post-Workout Nutrition',
      description: 'You completed a strength training session. Consider adding protein within 30 minutes.',
      recommendation: 'Greek yogurt with berries or protein shake',
      priority: 'high',
      calories: '+200-300 cal',
      icon: Zap,
      color: 'text-orange-600',
      bgColor: 'bg-orange-50',
      timeWindow: '20 minutes remaining'
    },
    {
      id: 2,
      type: 'nutrition_gap',
      title: 'Low Fiber Intake',
      description: 'You\'ve only consumed 12g of fiber today. Aim for 25g daily.',
      recommendation: 'Add vegetables to your next meal or have an apple with skin',
      priority: 'medium',
      calories: '+80-120 cal',
      icon: TrendingUp,
      color: 'text-green-600',
      bgColor: 'bg-green-50'
    },
    {
      id: 3,
      type: 'hydration',
      title: 'Hydration Reminder',
      description: 'You\'ve had 3 glasses of water today. Consider increasing intake.',
      recommendation: 'Drink 2 more glasses before your next meal',
      priority: 'medium',
      calories: '0 cal',
      icon: AlertCircle,
      color: 'text-blue-600',
      bgColor: 'bg-blue-50'
    },
    {
      id: 4,
      type: 'missed_meal',
      title: 'Missed Afternoon Snack',
      description: 'Your energy might dip. Consider a balanced snack.',
      recommendation: 'Nuts and fruit or vegetable sticks with hummus',
      priority: 'low',
      calories: '+150-200 cal',
      icon: Clock,
      color: 'text-purple-600',
      bgColor: 'bg-purple-50'
    }
  ];

  const getPriorityBadge = (priority: string) => {
    switch (priority) {
      case 'high':
        return <Badge variant="destructive" className="text-xs">High Priority</Badge>;
      case 'medium':
        return <Badge variant="default" className="text-xs">Medium</Badge>;
      case 'low':
        return <Badge variant="secondary" className="text-xs">Low</Badge>;
      default:
        return null;
    }
  };

  const completedSuggestions = [
    {
      title: 'Increased Protein',
      description: 'Added chicken to lunch',
      time: '2 hours ago'
    },
    {
      title: 'Hydration Goal',
      description: 'Reached 6 glasses of water',
      time: '4 hours ago'
    }
  ];

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold">Smart Suggestions</h2>
        <Badge variant="outline">
          {suggestions.length} active
        </Badge>
      </div>

      {/* AI-Based Suggestions */}
      <div className="space-y-3">
        {suggestions.map((suggestion) => {
          const IconComponent = suggestion.icon;
          
          return (
            <Card key={suggestion.id} className={`border-l-4 ${suggestion.priority === 'high' ? 'border-l-red-500' : suggestion.priority === 'medium' ? 'border-l-yellow-500' : 'border-l-blue-500'}`}>
              <CardContent className="p-4">
                <div className="flex items-start space-x-3">
                  <div className={`p-2 rounded-lg ${suggestion.bgColor} mt-1`}>
                    <IconComponent className={`w-4 h-4 ${suggestion.color}`} />
                  </div>
                  <div className="flex-1">
                    <div className="flex items-start justify-between mb-2">
                      <div>
                        <h4 className="font-medium">{suggestion.title}</h4>
                        {suggestion.timeWindow && (
                          <p className="text-xs text-orange-600 font-medium">{suggestion.timeWindow}</p>
                        )}
                      </div>
                      {getPriorityBadge(suggestion.priority)}
                    </div>
                    <p className="text-sm text-gray-600 mb-2">{suggestion.description}</p>
                    <div className="bg-gray-50 p-2 rounded text-sm mb-3">
                      <strong>Suggestion:</strong> {suggestion.recommendation}
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-gray-500">{suggestion.calories}</span>
                      <div className="flex gap-2">
                        <Button size="sm" variant="outline">
                          Not Now
                        </Button>
                        <Button size="sm">
                          Apply
                        </Button>
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          );
        })}
      </div>

      {/* Meal Swaps */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base flex items-center">
            <Lightbulb className="w-4 h-4 mr-2" />
            Smart Meal Swaps
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
              <div>
                <div className="font-medium text-sm">White Rice → Quinoa</div>
                <div className="text-xs text-gray-600">+5g protein, +3g fiber</div>
              </div>
              <Button size="sm" variant="outline">
                Swap
              </Button>
            </div>
            <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
              <div>
                <div className="font-medium text-sm">Regular Pasta → Lentil Pasta</div>
                <div className="text-xs text-gray-600">+8g protein, +7g fiber</div>
              </div>
              <Button size="sm" variant="outline">
                Swap
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Completed Suggestions */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base flex items-center">
            <CheckCircle className="w-4 h-4 mr-2 text-green-600" />
            Completed Today
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            {completedSuggestions.map((item, index) => (
              <div key={index} className="flex items-center justify-between p-2 bg-green-50 rounded-lg">
                <div>
                  <div className="font-medium text-sm text-green-800">{item.title}</div>
                  <div className="text-xs text-green-600">{item.description}</div>
                </div>
                <span className="text-xs text-green-600">{item.time}</span>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
