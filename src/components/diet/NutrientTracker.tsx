
import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Progress } from '../ui/progress';
import { Badge } from '../ui/badge';
import { Droplets, Zap, Shield, Wheat, Beef, Banana } from 'lucide-react';

interface NutrientTrackerProps {
  dailyGoals: {
    calories: number;
    protein: number;
    carbs: number;
    fat: number;
    fiber: number;
    water: number;
  };
  currentProgress: {
    calories: number;
    protein: number;
    carbs: number;
    fat: number;
    fiber: number;
    water: number;
  };
}

export const NutrientTracker: React.FC<NutrientTrackerProps> = ({ dailyGoals, currentProgress }) => {
  const calculatePercentage = (current: number, goal: number) => {
    return Math.min((current / goal) * 100, 100);
  };

  const getProgressColor = (percentage: number) => {
    if (percentage >= 100) return 'bg-green-500';
    if (percentage >= 80) return 'bg-yellow-500';
    return 'bg-blue-500';
  };

  const nutrients = [
    {
      name: 'Calories',
      current: currentProgress.calories,
      goal: dailyGoals.calories,
      unit: '',
      icon: Zap,
      color: 'text-orange-600',
      bgColor: 'bg-orange-100'
    },
    {
      name: 'Protein',
      current: currentProgress.protein,
      goal: dailyGoals.protein,
      unit: 'g',
      icon: Beef,
      color: 'text-red-600',
      bgColor: 'bg-red-100'
    },
    {
      name: 'Carbs',
      current: currentProgress.carbs,
      goal: dailyGoals.carbs,
      unit: 'g',
      icon: Banana,
      color: 'text-yellow-600',
      bgColor: 'bg-yellow-100'
    },
    {
      name: 'Fat',
      current: currentProgress.fat,
      goal: dailyGoals.fat,
      unit: 'g',
      icon: Shield,
      color: 'text-purple-600',
      bgColor: 'bg-purple-100'
    },
    {
      name: 'Fiber',
      current: currentProgress.fiber,
      goal: dailyGoals.fiber,
      unit: 'g',
      icon: Wheat,
      color: 'text-green-600',
      bgColor: 'bg-green-100'
    },
    {
      name: 'Water',
      current: currentProgress.water,
      goal: dailyGoals.water,
      unit: ' glasses',
      icon: Droplets,
      color: 'text-blue-600',
      bgColor: 'bg-blue-100'
    }
  ];

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold">Nutrient Progress</h2>
        <Badge variant="outline">
          {new Date().toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' })}
        </Badge>
      </div>

      {/* Macronutrients Overview */}
      <Card className="bg-gradient-to-r from-blue-50 to-purple-50">
        <CardHeader>
          <CardTitle className="text-base">Macronutrients</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-3 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-red-600">{currentProgress.protein}g</div>
              <div className="text-xs text-gray-600">Protein</div>
              <div className="text-xs text-gray-500">{((currentProgress.protein * 4) / currentProgress.calories * 100).toFixed(0)}% of calories</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-yellow-600">{currentProgress.carbs}g</div>
              <div className="text-xs text-gray-600">Carbs</div>
              <div className="text-xs text-gray-500">{((currentProgress.carbs * 4) / currentProgress.calories * 100).toFixed(0)}% of calories</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-purple-600">{currentProgress.fat}g</div>
              <div className="text-xs text-gray-600">Fat</div>
              <div className="text-xs text-gray-500">{((currentProgress.fat * 9) / currentProgress.calories * 100).toFixed(0)}% of calories</div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Detailed Nutrient Tracking */}
      <div className="space-y-3">
        {nutrients.map((nutrient) => {
          const percentage = calculatePercentage(nutrient.current, nutrient.goal);
          const IconComponent = nutrient.icon;
          
          return (
            <Card key={nutrient.name}>
              <CardContent className="p-4">
                <div className="flex items-center space-x-3">
                  <div className={`p-2 rounded-lg ${nutrient.bgColor}`}>
                    <IconComponent className={`w-5 h-5 ${nutrient.color}`} />
                  </div>
                  <div className="flex-1">
                    <div className="flex items-center justify-between mb-2">
                      <span className="font-medium">{nutrient.name}</span>
                      <span className="text-sm text-gray-600">
                        {nutrient.current}{nutrient.unit} / {nutrient.goal}{nutrient.unit}
                      </span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <Progress value={percentage} className="flex-1 h-2" />
                      <span className="text-xs text-gray-500 min-w-[35px]">
                        {Math.round(percentage)}%
                      </span>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          );
        })}
      </div>

      {/* Weekly Trends */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Weekly Trends</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-7 gap-1 text-center">
            {['M', 'T', 'W', 'T', 'F', 'S', 'S'].map((day, index) => (
              <div key={index} className="space-y-1">
                <div className="text-xs text-gray-500">{day}</div>
                <div className={`w-full h-8 rounded ${index <= 4 ? 'bg-green-200' : 'bg-gray-200'}`}></div>
                <div className="text-xs text-gray-600">{index <= 4 ? '✓' : '-'}</div>
              </div>
            ))}
          </div>
          <p className="text-xs text-gray-500 text-center mt-2">
            5/7 days on track this week
          </p>
        </CardContent>
      </Card>
    </div>
  );
};
