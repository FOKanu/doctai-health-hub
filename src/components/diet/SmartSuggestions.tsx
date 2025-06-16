
import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Brain, Lightbulb, TrendingUp, AlertCircle, Clock, Zap } from 'lucide-react';

interface NutrientData {
  calories: number;
  protein: number;
  carbs: number;
  fat: number;
  fiber: number;
  water: number;
}

interface SmartSuggestionsProps {
  currentIntake: NutrientData;
  dailyTargets: NutrientData;
  detailed?: boolean;
}

export const SmartSuggestions: React.FC<SmartSuggestionsProps> = ({ 
  currentIntake, 
  dailyTargets,
  detailed = false 
}) => {
  const generateSuggestions = () => {
    const suggestions = [];
    
    // Calorie suggestions
    const calorieDeficit = dailyTargets.calories - currentIntake.calories;
    if (calorieDeficit > 300) {
      suggestions.push({
        type: 'energy',
        priority: 'high',
        title: 'Add a Healthy Snack',
        description: `You need ${calorieDeficit} more calories today. Try a protein smoothie or nuts.`,
        action: 'Add snack',
        icon: Zap,
        color: 'text-orange-600 bg-orange-100'
      });
    }

    // Protein suggestions
    const proteinDeficit = dailyTargets.protein - currentIntake.protein;
    if (proteinDeficit > 15) {
      suggestions.push({
        type: 'protein',
        priority: 'high',
        title: 'Boost Your Protein',
        description: `Add ${proteinDeficit.toFixed(0)}g more protein. Consider Greek yogurt or lean meat.`,
        action: 'Find protein foods',
        icon: TrendingUp,
        color: 'text-red-600 bg-red-100'
      });
    }

    // Hydration suggestions
    const waterDeficit = dailyTargets.water - currentIntake.water;
    if (waterDeficit > 500) {
      suggestions.push({
        type: 'hydration',
        priority: 'medium',
        title: 'Stay Hydrated',
        description: `Drink ${(waterDeficit / 250).toFixed(0)} more glasses of water today.`,
        action: 'Set reminder',
        icon: AlertCircle,
        color: 'text-blue-600 bg-blue-100'
      });
    }

    // Timing suggestions
    const currentHour = new Date().getHours();
    if (currentHour >= 18 && calorieDeficit > 400) {
      suggestions.push({
        type: 'timing',
        priority: 'medium',
        title: 'Plan Your Dinner',
        description: 'It\'s evening and you have calories left. Plan a balanced dinner.',
        action: 'View dinner ideas',
        icon: Clock,
        color: 'text-purple-600 bg-purple-100'
      });
    }

    // Meal swap suggestions
    if (currentIntake.carbs > dailyTargets.carbs * 0.8) {
      suggestions.push({
        type: 'swap',
        priority: 'low',
        title: 'Smart Meal Swap',
        description: 'Consider replacing refined carbs with vegetables for your next meal.',
        action: 'View alternatives',
        icon: Lightbulb,
        color: 'text-green-600 bg-green-100'
      });
    }

    return suggestions;
  };

  const mealRecommendations = [
    {
      name: 'Protein Power Bowl',
      calories: 420,
      protein: 35,
      description: 'Grilled chicken, quinoa, and vegetables',
      reason: 'High protein to meet your daily goal'
    },
    {
      name: 'Green Energy Smoothie',
      calories: 280,
      protein: 15,
      description: 'Spinach, banana, protein powder, almond milk',
      reason: 'Quick calories and nutrients'
    },
    {
      name: 'Mediterranean Snack',
      calories: 200,
      protein: 8,
      description: 'Hummus with vegetables and whole grain pita',
      reason: 'Balanced macros and fiber'
    }
  ];

  const suggestions = generateSuggestions();

  return (
    <div className="space-y-6">
      {/* AI Insights Header */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="w-5 h-5" />
            Smart Nutrition Insights
          </CardTitle>
          <CardDescription>
            Personalized recommendations based on your current intake
          </CardDescription>
        </CardHeader>
      </Card>

      {/* Priority Suggestions */}
      <div className="space-y-4">
        {suggestions.map((suggestion, index) => (
          <Card key={index} className="border-l-4 border-l-blue-500">
            <CardContent className="p-4">
              <div className="flex items-start justify-between">
                <div className="flex items-start space-x-3">
                  <div className={`p-2 rounded-full ${suggestion.color}`}>
                    <suggestion.icon className="w-4 h-4" />
                  </div>
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <h3 className="font-semibold text-gray-900">{suggestion.title}</h3>
                      <Badge variant={suggestion.priority === 'high' ? 'destructive' : 
                                    suggestion.priority === 'medium' ? 'default' : 'secondary'}>
                        {suggestion.priority}
                      </Badge>
                    </div>
                    <p className="text-sm text-gray-600 mb-2">{suggestion.description}</p>
                    <Button size="sm" variant="outline">
                      {suggestion.action}
                    </Button>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Meal Recommendations */}
      {detailed && (
        <Card>
          <CardHeader>
            <CardTitle>Recommended for You</CardTitle>
            <CardDescription>
              Meals that align with your current nutritional needs
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {mealRecommendations.map((meal, index) => (
              <div key={index} className="flex items-center justify-between p-4 border rounded-lg hover:bg-gray-50">
                <div>
                  <h4 className="font-semibold text-gray-900">{meal.name}</h4>
                  <p className="text-sm text-gray-600 mb-1">{meal.description}</p>
                  <p className="text-xs text-blue-600">{meal.reason}</p>
                  <div className="flex gap-3 text-xs text-gray-500 mt-2">
                    <span>{meal.calories} cal</span>
                    <span>{meal.protein}g protein</span>
                  </div>
                </div>
                <Button size="sm">Add to Plan</Button>
              </div>
            ))}
          </CardContent>
        </Card>
      )}

      {/* Weekly Insights */}
      {detailed && (
        <Card>
          <CardHeader>
            <CardTitle>Weekly Patterns</CardTitle>
            <CardDescription>
              Insights from your nutrition trends
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="p-4 bg-green-50 rounded-lg">
                <h4 className="font-semibold text-green-800 mb-2">💪 Strength</h4>
                <p className="text-sm text-green-700">
                  You consistently meet your protein goals on workout days
                </p>
              </div>
              <div className="p-4 bg-yellow-50 rounded-lg">
                <h4 className="font-semibold text-yellow-800 mb-2">🎯 Opportunity</h4>
                <p className="text-sm text-yellow-700">
                  Weekend hydration could be improved by 20%
                </p>
              </div>
              <div className="p-4 bg-blue-50 rounded-lg">
                <h4 className="font-semibold text-blue-800 mb-2">📊 Trend</h4>
                <p className="text-sm text-blue-700">
                  Your fiber intake has improved 15% this week
                </p>
              </div>
              <div className="p-4 bg-purple-50 rounded-lg">
                <h4 className="font-semibold text-purple-800 mb-2">🔮 Prediction</h4>
                <p className="text-sm text-purple-700">
                  You're on track to exceed weekly nutrition goals
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};
