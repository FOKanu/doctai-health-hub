
import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Checkbox } from '@/components/ui/checkbox';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { CheckCircle2, Clock, Target, Award } from 'lucide-react';

export const DailyWorkoutChecklist: React.FC = () => {
  const [checkedItems, setCheckedItems] = useState<Record<string, boolean>>({});

  const dailyTasks = [
    {
      id: 'warmup',
      title: '5-Minute Warm-up',
      description: 'Light cardio and dynamic stretching',
      points: 25,
      category: 'Essential',
      estimated: 5
    },
    {
      id: 'strength',
      title: 'Strength Training',
      description: 'Complete your scheduled strength workout',
      points: 100,
      category: 'Main Workout',
      estimated: 45
    },
    {
      id: 'cardio',
      title: 'Cardio Session',
      description: '20 minutes of cardiovascular exercise',
      points: 75,
      category: 'Main Workout',
      estimated: 20
    },
    {
      id: 'core',
      title: 'Core Strengthening',
      description: 'Targeted abdominal and core exercises',
      points: 50,
      category: 'Supplementary',
      estimated: 15
    },
    {
      id: 'flexibility',
      title: 'Flexibility & Mobility',
      description: 'Stretching and mobility exercises',
      points: 40,
      category: 'Recovery',
      estimated: 10
    },
    {
      id: 'hydration',
      title: 'Hydration Check',
      description: 'Drink at least 8 glasses of water',
      points: 30,
      category: 'Wellness',
      estimated: 0
    },
    {
      id: 'sleep',
      title: 'Sleep Tracking',
      description: 'Log your sleep quality and duration',
      points: 35,
      category: 'Recovery',
      estimated: 2
    },
    {
      id: 'nutrition',
      title: 'Nutrition Log',
      description: 'Track your meals and macros',
      points: 45,
      category: 'Wellness',
      estimated: 5
    }
  ];

  const handleCheckboxChange = (taskId: string, checked: boolean) => {
    setCheckedItems(prev => ({
      ...prev,
      [taskId]: checked
    }));
  };

  const completedTasks = Object.values(checkedItems).filter(Boolean).length;
  const totalPoints = dailyTasks
    .filter(task => checkedItems[task.id])
    .reduce((sum, task) => sum + task.points, 0);
  const maxPoints = dailyTasks.reduce((sum, task) => sum + task.points, 0);
  const completionPercentage = (completedTasks / dailyTasks.length) * 100;

  const getCategoryColor = (category: string) => {
    switch (category) {
      case 'Essential': return 'bg-red-100 text-red-800';
      case 'Main Workout': return 'bg-blue-100 text-blue-800';
      case 'Supplementary': return 'bg-green-100 text-green-800';
      case 'Recovery': return 'bg-purple-100 text-purple-800';
      case 'Wellness': return 'bg-orange-100 text-orange-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <div className="space-y-6">
      {/* Progress Overview */}
      <Card className="bg-gradient-to-r from-green-50 to-emerald-50 border-green-200">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-green-900">
            <Target className="w-5 h-5" />
            Daily Progress
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-green-700">{completedTasks}/{dailyTasks.length}</div>
              <div className="text-sm text-green-600">Tasks Completed</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-green-700">{totalPoints}</div>
              <div className="text-sm text-green-600">Health Points Earned</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-green-700">{Math.round(completionPercentage)}%</div>
              <div className="text-sm text-green-600">Daily Goal</div>
            </div>
          </div>
          <Progress value={completionPercentage} className="mt-4 h-2" />
        </CardContent>
      </Card>

      {/* Checklist */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <CheckCircle2 className="w-5 h-5" />
            Today's Fitness Checklist
          </CardTitle>
          <CardDescription>
            Complete tasks to earn health points and improve your fitness score
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {dailyTasks.map((task) => (
              <div key={task.id} className={`p-4 rounded-lg border-2 transition-all ${
                checkedItems[task.id] 
                  ? 'border-green-200 bg-green-50' 
                  : 'border-gray-200 hover:border-gray-300'
              }`}>
                <div className="flex items-start space-x-3">
                  <Checkbox
                    id={task.id}
                    checked={checkedItems[task.id] || false}
                    onCheckedChange={(checked) => handleCheckboxChange(task.id, checked as boolean)}
                    className="mt-1"
                  />
                  <div className="flex-1">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-2">
                        <h3 className={`font-semibold ${
                          checkedItems[task.id] ? 'text-green-800 line-through' : 'text-gray-900'
                        }`}>
                          {task.title}
                        </h3>
                        <Badge className={getCategoryColor(task.category)}>
                          {task.category}
                        </Badge>
                      </div>
                      <div className="flex items-center space-x-2">
                        {task.estimated > 0 && (
                          <div className="flex items-center space-x-1 text-xs text-gray-500">
                            <Clock className="w-3 h-3" />
                            <span>{task.estimated}min</span>
                          </div>
                        )}
                        <div className="flex items-center space-x-1">
                          <Award className="w-4 h-4 text-orange-500" />
                          <span className="text-sm font-medium text-orange-600">+{task.points}pt</span>
                        </div>
                      </div>
                    </div>
                    <p className={`text-sm mt-1 ${
                      checkedItems[task.id] ? 'text-green-600' : 'text-gray-600'
                    }`}>
                      {task.description}
                    </p>
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Bonus Achievement */}
          {completionPercentage === 100 && (
            <div className="mt-6 p-4 bg-gradient-to-r from-yellow-100 to-orange-100 border-2 border-yellow-300 rounded-lg">
              <div className="flex items-center space-x-2">
                <Award className="w-6 h-6 text-yellow-600" />
                <div>
                  <h3 className="font-bold text-yellow-800">Perfect Day Bonus!</h3>
                  <p className="text-sm text-yellow-700">You've completed all tasks! +50 bonus points earned!</p>
                </div>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
};
