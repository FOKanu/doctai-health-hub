
import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { User, Target, AlertCircle, Leaf } from 'lucide-react';

interface UserProfile {
  age: number;
  weight: number;
  height: number;
  activityLevel: string;
  goal: string;
  allergies: string[];
  preferences: string[];
}

interface PersonalizedMealPlanProps {
  userProfile: UserProfile;
}

export const PersonalizedMealPlan: React.FC<PersonalizedMealPlanProps> = ({ userProfile }) => {
  const mealPlanType = () => {
    if (userProfile.preferences.includes('vegetarian')) return 'Vegetarian Balance';
    if (userProfile.goal === 'weight_loss') return 'Lean & Clean';
    if (userProfile.goal === 'muscle_gain') return 'Power Building';
    return 'Balanced Nutrition';
  };

  const recommendedCalories = () => {
    const bmr = userProfile.weight * 22; // Simplified calculation
    const activityMultiplier = userProfile.activityLevel === 'high' ? 1.7 : 
                              userProfile.activityLevel === 'moderate' ? 1.5 : 1.3;
    return Math.round(bmr * activityMultiplier);
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <User className="w-5 h-5" />
          Your Personalized Plan
        </CardTitle>
        <CardDescription>
          Customized for your goals, preferences, and dietary needs
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Plan Overview */}
        <div className="p-4 bg-blue-50 rounded-lg">
          <h3 className="font-semibold text-blue-900 mb-2">{mealPlanType()}</h3>
          <p className="text-sm text-blue-700">
            Designed for {userProfile.activityLevel} activity level with {userProfile.goal} goals
          </p>
        </div>

        {/* Daily Targets */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-600">Daily Calories</span>
            <Badge variant="outline">{recommendedCalories()} kcal</Badge>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-600">Protein Target</span>
            <Badge variant="outline">{Math.round(userProfile.weight * 1.6)}g</Badge>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-600">Meals per Day</span>
            <Badge variant="outline">4-5</Badge>
          </div>
        </div>

        {/* Dietary Preferences & Restrictions */}
        <div className="space-y-2">
          <h4 className="font-medium text-gray-900">Preferences & Restrictions</h4>
          <div className="flex flex-wrap gap-2">
            {userProfile.preferences.map((pref, index) => (
              <Badge key={index} variant="secondary" className="text-green-700 bg-green-100">
                <Leaf className="w-3 h-3 mr-1" />
                {pref}
              </Badge>
            ))}
            {userProfile.allergies.map((allergy, index) => (
              <Badge key={index} variant="destructive" className="bg-red-100 text-red-700">
                <AlertCircle className="w-3 h-3 mr-1" />
                No {allergy}
              </Badge>
            ))}
          </div>
        </div>

        {/* Key Recommendations */}
        <div className="space-y-2">
          <h4 className="font-medium text-gray-900">Key Recommendations</h4>
          <ul className="text-sm text-gray-600 space-y-1">
            <li>• Focus on whole grains and lean proteins</li>
            <li>• Include 5+ servings of fruits & vegetables daily</li>
            <li>• Stay hydrated with 2.5L+ water intake</li>
            <li>• Time protein intake around workouts</li>
          </ul>
        </div>

        <Button className="w-full">
          <Target className="w-4 h-4 mr-2" />
          Customize Plan
        </Button>
      </CardContent>
    </Card>
  );
};
