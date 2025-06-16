
import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { Target, TrendingUp, TrendingDown, Minus, Droplet } from 'lucide-react';

interface NutrientData {
  calories: number;
  protein: number;
  carbs: number;
  fat: number;
  fiber: number;
  water: number;
}

interface NutrientTrackerProps {
  currentIntake: NutrientData;
  dailyTargets: NutrientData;
}

export const NutrientTracker: React.FC<NutrientTrackerProps> = ({ 
  currentIntake, 
  dailyTargets 
}) => {
  const getNutrientStatus = (current: number, target: number) => {
    const percentage = (current / target) * 100;
    if (percentage >= 100) return 'complete';
    if (percentage >= 75) return 'good';
    if (percentage >= 50) return 'moderate';
    return 'low';
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'complete': return 'text-green-600 bg-green-100';
      case 'good': return 'text-blue-600 bg-blue-100';
      case 'moderate': return 'text-yellow-600 bg-yellow-100';
      case 'low': return 'text-red-600 bg-red-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const getStatusIcon = (current: number, target: number) => {
    if (current >= target) return <TrendingUp className="w-4 h-4" />;
    if (current >= target * 0.75) return <Minus className="w-4 h-4" />;
    return <TrendingDown className="w-4 h-4" />;
  };

  const nutrients = [
    {
      name: 'Calories',
      current: currentIntake.calories,
      target: dailyTargets.calories,
      unit: 'kcal',
      description: 'Energy for daily activities',
      icon: '🔥'
    },
    {
      name: 'Protein',
      current: currentIntake.protein,
      target: dailyTargets.protein,
      unit: 'g',
      description: 'Muscle building and repair',
      icon: '💪'
    },
    {
      name: 'Carbohydrates',
      current: currentIntake.carbs,
      target: dailyTargets.carbs,
      unit: 'g',
      description: 'Primary energy source',
      icon: '🌾'
    },
    {
      name: 'Fat',
      current: currentIntake.fat,
      target: dailyTargets.fat,
      unit: 'g',
      description: 'Essential fatty acids',
      icon: '🥑'
    },
    {
      name: 'Fiber',
      current: currentIntake.fiber,
      target: dailyTargets.fiber,
      unit: 'g',
      description: 'Digestive health',
      icon: '🌿'
    },
    {
      name: 'Water',
      current: currentIntake.water,
      target: dailyTargets.water,
      unit: 'ml',
      description: 'Hydration and metabolism',
      icon: '💧'
    }
  ];

  return (
    <div className="space-y-6">
      {/* Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card>
          <CardContent className="p-6 text-center">
            <Target className="w-8 h-8 mx-auto text-blue-500 mb-2" />
            <p className="text-2xl font-bold text-gray-900">
              {nutrients.filter(n => (n.current / n.target) >= 1).length}
            </p>
            <p className="text-sm text-gray-600">Goals Achieved</p>
          </CardContent>
        </Card>
        
        <Card>
          <CardContent className="p-6 text-center">
            <TrendingUp className="w-8 h-8 mx-auto text-green-500 mb-2" />
            <p className="text-2xl font-bold text-gray-900">
              {Math.round((currentIntake.calories / dailyTargets.calories) * 100)}%
            </p>
            <p className="text-sm text-gray-600">Calorie Progress</p>
          </CardContent>
        </Card>
        
        <Card>
          <CardContent className="p-6 text-center">
            <Droplet className="w-8 h-8 mx-auto text-blue-400 mb-2" />
            <p className="text-2xl font-bold text-gray-900">
              {(currentIntake.water / 1000).toFixed(1)}L
            </p>
            <p className="text-sm text-gray-600">Water Intake</p>
          </CardContent>
        </Card>
      </div>

      {/* Detailed Nutrient Tracking */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {nutrients.map((nutrient, index) => {
          const percentage = Math.min((nutrient.current / nutrient.target) * 100, 100);
          const status = getNutrientStatus(nutrient.current, nutrient.target);
          const remaining = Math.max(nutrient.target - nutrient.current, 0);
          
          return (
            <Card key={index}>
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-lg flex items-center gap-2">
                    <span className="text-xl">{nutrient.icon}</span>
                    {nutrient.name}
                  </CardTitle>
                  <Badge className={getStatusColor(status)}>
                    {getStatusIcon(nutrient.current, nutrient.target)}
                    <span className="ml-1 capitalize">{status}</span>
                  </Badge>
                </div>
                <CardDescription>{nutrient.description}</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {/* Progress Bar */}
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span>{nutrient.current}{nutrient.unit}</span>
                    <span className="text-gray-500">
                      of {nutrient.target}{nutrient.unit}
                    </span>
                  </div>
                  <Progress value={percentage} className="h-3" />
                  <div className="text-xs text-gray-500">
                    {percentage.toFixed(1)}% complete
                  </div>
                </div>

                {/* Remaining Amount */}
                {remaining > 0 && (
                  <div className="p-3 bg-gray-50 rounded-lg">
                    <p className="text-sm">
                      <span className="font-medium">{remaining.toFixed(1)}{nutrient.unit}</span>
                      <span className="text-gray-600"> remaining to reach your goal</span>
                    </p>
                  </div>
                )}

                {/* Exceeded Goal */}
                {nutrient.current > nutrient.target && (
                  <div className="p-3 bg-green-50 rounded-lg">
                    <p className="text-sm text-green-800">
                      🎉 Goal exceeded by{' '}
                      <span className="font-medium">
                        {(nutrient.current - nutrient.target).toFixed(1)}{nutrient.unit}
                      </span>
                    </p>
                  </div>
                )}
              </CardContent>
            </Card>
          );
        })}
      </div>

      {/* Weekly Trends */}
      <Card>
        <CardHeader>
          <CardTitle>Weekly Nutrient Trends</CardTitle>
          <CardDescription>
            Your nutrient intake patterns over the past week
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-7 gap-2 text-center text-xs">
            {['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'].map((day, index) => (
              <div key={day} className="space-y-2">
                <p className="font-medium text-gray-700">{day}</p>
                <div className="space-y-1">
                  <div className="h-12 bg-blue-200 rounded flex items-end justify-center">
                    <div className={`w-full bg-blue-500 rounded ${index === 6 ? 'h-10' : 'h-8'}`}></div>
                  </div>
                  <p className="text-gray-600">{index === 6 ? '75%' : '85%'}</p>
                </div>
              </div>
            ))}
          </div>
          <p className="text-center text-sm text-gray-600 mt-4">
            Calorie goal achievement percentage
          </p>
        </CardContent>
      </Card>
    </div>
  );
};
