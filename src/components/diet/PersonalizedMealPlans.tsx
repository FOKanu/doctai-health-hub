
import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Button } from '../ui/button';
import { Badge } from '../ui/badge';
import { Clock, Users, Flame, Award } from 'lucide-react';

interface UserProfile {
  goals: string[];
  allergies: string[];
  activityLevel: string;
  preferences: string[];
}

interface PersonalizedMealPlansProps {
  userProfile: UserProfile;
}

export const PersonalizedMealPlans: React.FC<PersonalizedMealPlansProps> = ({ userProfile }) => {
  const mealPlans = [
    {
      id: 1,
      name: 'High Protein Weight Loss',
      description: 'Optimized for muscle preservation during weight loss',
      calories: '1800-2000',
      duration: '4 weeks',
      difficulty: 'Beginner',
      tags: ['High Protein', 'Weight Loss', 'Vegetarian'],
      compatibility: 95,
      meals: 5,
      prepTime: '30 min/day'
    },
    {
      id: 2,
      name: 'Lean Muscle Builder',
      description: 'Balanced macros for steady muscle growth',
      calories: '2200-2400',
      duration: '6 weeks',
      difficulty: 'Intermediate',
      tags: ['Muscle Gain', 'Balanced', 'Dairy-Free'],
      compatibility: 88,
      meals: 6,
      prepTime: '45 min/day'
    },
    {
      id: 3,
      name: 'Active Lifestyle',
      description: 'Quick prep meals for busy schedules',
      calories: '2000-2200',
      duration: '2 weeks',
      difficulty: 'Beginner',
      tags: ['Quick Prep', 'Balanced', 'On-the-Go'],
      compatibility: 82,
      meals: 4,
      prepTime: '20 min/day'
    }
  ];

  const getCompatibilityColor = (score: number) => {
    if (score >= 90) return 'text-green-600 bg-green-50';
    if (score >= 80) return 'text-yellow-600 bg-yellow-50';
    return 'text-orange-600 bg-orange-50';
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold">Recommended Plans</h2>
        <Button variant="outline" size="sm">
          Customize
        </Button>
      </div>

      {/* User Profile Summary */}
      <Card className="bg-blue-50 border-blue-200">
        <CardContent className="pt-4">
          <div className="flex flex-wrap gap-2">
            <Badge variant="outline">Goals: {userProfile.goals.join(', ')}</Badge>
            <Badge variant="outline">Activity: {userProfile.activityLevel}</Badge>
            <Badge variant="outline">Preferences: {userProfile.preferences.join(', ')}</Badge>
            {userProfile.allergies.length > 0 && (
              <Badge variant="destructive">Avoid: {userProfile.allergies.join(', ')}</Badge>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Meal Plans */}
      {mealPlans.map((plan) => (
        <Card key={plan.id} className="relative">
          <CardHeader className="pb-3">
            <div className="flex items-start justify-between">
              <div>
                <CardTitle className="text-lg">{plan.name}</CardTitle>
                <p className="text-sm text-gray-600 mt-1">{plan.description}</p>
              </div>
              <div className={`px-2 py-1 rounded-full text-xs font-medium ${getCompatibilityColor(plan.compatibility)}`}>
                {plan.compatibility}% match
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-4 mb-4">
              <div className="flex items-center text-sm text-gray-600">
                <Flame className="w-4 h-4 mr-2" />
                {plan.calories} cal/day
              </div>
              <div className="flex items-center text-sm text-gray-600">
                <Clock className="w-4 h-4 mr-2" />
                {plan.prepTime}
              </div>
              <div className="flex items-center text-sm text-gray-600">
                <Users className="w-4 h-4 mr-2" />
                {plan.meals} meals/day
              </div>
              <div className="flex items-center text-sm text-gray-600">
                <Award className="w-4 h-4 mr-2" />
                {plan.difficulty}
              </div>
            </div>

            <div className="flex flex-wrap gap-1 mb-4">
              {plan.tags.map((tag, index) => (
                <Badge key={index} variant="secondary" className="text-xs">
                  {tag}
                </Badge>
              ))}
            </div>

            <div className="flex gap-2">
              <Button className="flex-1" size="sm">
                Start Plan
              </Button>
              <Button variant="outline" size="sm">
                Preview
              </Button>
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  );
};
