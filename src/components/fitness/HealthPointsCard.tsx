
import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { Trophy, Award, Star, Target, TrendingUp } from 'lucide-react';

interface HealthPointsCardProps {
  points: number;
}

export const HealthPointsCard: React.FC<HealthPointsCardProps> = ({ points }) => {
  const levels = [
    { name: 'Beginner', min: 0, max: 499, color: 'bg-gray-500', icon: Star },
    { name: 'Active', min: 500, max: 999, color: 'bg-green-500', icon: Target },
    { name: 'Athlete', min: 1000, max: 1999, color: 'bg-blue-500', icon: Award },
    { name: 'Champion', min: 2000, max: 4999, color: 'bg-purple-500', icon: Trophy },
    { name: 'Legend', min: 5000, max: 999999, color: 'bg-yellow-500', icon: TrendingUp }
  ];

  const currentLevel = levels.find(level => points >= level.min && points <= level.max) || levels[0];
  const nextLevel = levels[levels.indexOf(currentLevel) + 1];
  
  const progressToNextLevel = nextLevel 
    ? ((points - currentLevel.min) / (nextLevel.min - currentLevel.min)) * 100 
    : 100;

  const pointsToNext = nextLevel ? nextLevel.min - points : 0;

  const recentAchievements = [
    { title: 'Consistency King', description: '7 days in a row', points: 100, date: '2 days ago' },
    { title: 'Cardio Crusher', description: 'Completed HIIT workout', points: 150, date: 'Yesterday' },
    { title: 'Early Bird', description: 'Morning workout', points: 50, date: 'Today' }
  ];

  const IconComponent = currentLevel.icon;

  return (
    <Card className="bg-gradient-to-br from-blue-50 to-indigo-100 border-blue-200">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-blue-900">
          <IconComponent className="w-5 h-5" />
          Health Points System
        </CardTitle>
        <CardDescription className="text-blue-700">
          Earn points by completing workouts and achieving fitness goals
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Current Level */}
        <div className="text-center">
          <div className="flex items-center justify-center space-x-2 mb-2">
            <Badge className={`${currentLevel.color} text-white px-3 py-1`}>
              {currentLevel.name}
            </Badge>
          </div>
          <div className="text-3xl font-bold text-blue-900">{points.toLocaleString()}</div>
          <div className="text-sm text-blue-700">Total Health Points</div>
        </div>

        {/* Progress to Next Level */}
        {nextLevel && (
          <div className="space-y-2">
            <div className="flex justify-between text-sm text-blue-700">
              <span>Progress to {nextLevel.name}</span>
              <span>{pointsToNext} points to go</span>
            </div>
            <Progress value={progressToNextLevel} className="h-2" />
          </div>
        )}

        {/* Points Breakdown */}
        <div className="grid grid-cols-2 gap-4 text-center">
          <div className="bg-white rounded-lg p-3 border border-blue-200">
            <div className="text-lg font-bold text-blue-900">+185</div>
            <div className="text-xs text-blue-600">This Week</div>
          </div>
          <div className="bg-white rounded-lg p-3 border border-blue-200">
            <div className="text-lg font-bold text-blue-900">+45</div>
            <div className="text-xs text-blue-600">Today</div>
          </div>
        </div>

        {/* Recent Achievements */}
        <div>
          <h4 className="font-semibold text-blue-900 mb-3">Recent Achievements</h4>
          <div className="space-y-2">
            {recentAchievements.map((achievement, index) => (
              <div key={index} className="flex items-center justify-between bg-white rounded-lg p-2 border border-blue-200">
                <div className="flex-1">
                  <div className="font-medium text-sm text-gray-900">{achievement.title}</div>
                  <div className="text-xs text-gray-600">{achievement.description}</div>
                </div>
                <div className="text-right">
                  <div className="text-sm font-bold text-green-600">+{achievement.points}</div>
                  <div className="text-xs text-gray-500">{achievement.date}</div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Point Values Guide */}
        <div className="bg-white rounded-lg p-3 border border-blue-200">
          <h5 className="font-medium text-blue-900 mb-2 text-sm">How to Earn Points:</h5>
          <div className="space-y-1 text-xs text-blue-700">
            <div className="flex justify-between">
              <span>Complete workout</span>
              <span>50-200 pts</span>
            </div>
            <div className="flex justify-between">
              <span>Daily checklist item</span>
              <span>25-100 pts</span>
            </div>
            <div className="flex justify-between">
              <span>Weekly goal achieved</span>
              <span>150 pts</span>
            </div>
            <div className="flex justify-between">
              <span>Perfect day bonus</span>
              <span>50 pts</span>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};
