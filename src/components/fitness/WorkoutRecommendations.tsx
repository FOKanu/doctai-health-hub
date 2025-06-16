
import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Clock, Target, Flame, Dumbbell, Heart, Zap } from 'lucide-react';

interface WorkoutRecommendationsProps {
  onUpdateWorkout: (workout: any) => void;
}

export const WorkoutRecommendations: React.FC<WorkoutRecommendationsProps> = ({ onUpdateWorkout }) => {
  const recommendations = [
    {
      id: 1,
      title: 'Upper Body Strength',
      description: 'Focus on chest, shoulders, and arms',
      duration: 45,
      difficulty: 'Intermediate',
      calories: 320,
      exercises: ['Push-ups', 'Bench Press', 'Shoulder Press', 'Bicep Curls'],
      muscleGroups: ['Chest', 'Shoulders', 'Arms'],
      type: 'strength',
      icon: Dumbbell,
      color: 'bg-blue-600'
    },
    {
      id: 2,
      title: 'HIIT Cardio Blast',
      description: 'High-intensity interval training',
      duration: 30,
      difficulty: 'Advanced',
      calories: 450,
      exercises: ['Burpees', 'Mountain Climbers', 'Jump Squats', 'High Knees'],
      muscleGroups: ['Full Body'],
      type: 'cardio',
      icon: Zap,
      color: 'bg-orange-600'
    },
    {
      id: 3,
      title: 'Lower Body Power',
      description: 'Legs and glutes focused workout',
      duration: 50,
      difficulty: 'Intermediate',
      calories: 380,
      exercises: ['Squats', 'Deadlifts', 'Lunges', 'Calf Raises'],
      muscleGroups: ['Legs', 'Glutes'],
      type: 'strength',
      icon: Target,
      color: 'bg-green-600'
    },
    {
      id: 4,
      title: 'Active Recovery',
      description: 'Light cardio and stretching',
      duration: 25,
      difficulty: 'Beginner',
      calories: 150,
      exercises: ['Walking', 'Yoga Poses', 'Light Stretching', 'Breathing'],
      muscleGroups: ['Full Body'],
      type: 'recovery',
      icon: Heart,
      color: 'bg-purple-600'
    }
  ];

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case 'Beginner': return 'bg-green-100 text-green-800';
      case 'Intermediate': return 'bg-yellow-100 text-yellow-800';
      case 'Advanced': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-gray-900">Recommended Workouts</h2>
          <p className="text-gray-600">AI-powered recommendations based on your goals and history</p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {recommendations.map((workout) => {
          const IconComponent = workout.icon;
          return (
            <Card key={workout.id} className="hover:shadow-lg transition-shadow">
              <CardHeader>
                <div className="flex items-start justify-between">
                  <div className="flex items-center space-x-3">
                    <div className={`p-2 rounded-lg ${workout.color} text-white`}>
                      <IconComponent className="w-5 h-5" />
                    </div>
                    <div>
                      <CardTitle className="text-lg">{workout.title}</CardTitle>
                      <CardDescription>{workout.description}</CardDescription>
                    </div>
                  </div>
                  <Badge className={getDifficultyColor(workout.difficulty)}>
                    {workout.difficulty}
                  </Badge>
                </div>
              </CardHeader>
              <CardContent className="space-y-4">
                {/* Workout Stats */}
                <div className="flex items-center justify-between text-sm">
                  <div className="flex items-center space-x-1">
                    <Clock className="w-4 h-4 text-gray-500" />
                    <span>{workout.duration} min</span>
                  </div>
                  <div className="flex items-center space-x-1">
                    <Flame className="w-4 h-4 text-orange-500" />
                    <span>{workout.calories} cal</span>
                  </div>
                </div>

                {/* Muscle Groups */}
                <div>
                  <h4 className="text-sm font-medium text-gray-700 mb-2">Target Muscles:</h4>
                  <div className="flex flex-wrap gap-1">
                    {workout.muscleGroups.map((muscle, index) => (
                      <Badge key={index} variant="secondary" className="text-xs">
                        {muscle}
                      </Badge>
                    ))}
                  </div>
                </div>

                {/* Exercises Preview */}
                <div>
                  <h4 className="text-sm font-medium text-gray-700 mb-2">Exercises:</h4>
                  <div className="text-sm text-gray-600">
                    {workout.exercises.slice(0, 3).join(', ')}
                    {workout.exercises.length > 3 && ` +${workout.exercises.length - 3} more`}
                  </div>
                </div>

                {/* Actions */}
                <div className="flex space-x-2 pt-2">
                  <Button 
                    onClick={() => onUpdateWorkout(workout)}
                    className="flex-1"
                    variant="default"
                  >
                    Start Workout
                  </Button>
                  <Button variant="outline" size="sm">
                    View Details
                  </Button>
                </div>
              </CardContent>
            </Card>
          );
        })}
      </div>

      {/* AI Insights */}
      <Card className="bg-gradient-to-r from-blue-50 to-indigo-50 border-blue-200">
        <CardHeader>
          <CardTitle className="text-lg text-blue-900">🤖 AI Fitness Insights</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-2 text-sm text-blue-800">
            <p>• Based on your recent activity, upper body strength training is recommended</p>
            <p>• You've been consistent with cardio - consider adding more strength training</p>
            <p>• Your recovery time suggests you're ready for higher intensity workouts</p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
