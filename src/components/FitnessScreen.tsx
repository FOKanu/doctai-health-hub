
import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Activity,
  Target,
  Trophy,
  Clock,
  Flame,
  Heart,
  Footprints,
  Plus,
  CheckCircle2,
  Circle,
  Dumbbell,
  Calendar,
  TrendingUp,
  Award,
  Zap,
  BarChart3
} from 'lucide-react';
import { WorkoutRecommendations } from './fitness/WorkoutRecommendations';
import { DailyWorkoutChecklist } from './fitness/DailyWorkoutChecklist';
import { UpdateWorkoutModal } from './fitness/UpdateWorkoutModal';
import { HealthPointsCard } from './fitness/HealthPointsCard';
import { FitnessDataSync } from './fitness/FitnessDataSync';
import { MotivationalTips } from './fitness/MotivationalTips';

const FitnessScreen = () => {
  const [isUpdateModalOpen, setIsUpdateModalOpen] = useState(false);
  const [selectedWorkout, setSelectedWorkout] = useState(null);

  // Sample fitness data
  const todaysStats = {
    steps: 8547,
    stepsGoal: 10000,
    calories: 420,
    caloriesGoal: 600,
    activeMinutes: 45,
    activeGoal: 60,
    heartRate: 75,
    healthPoints: 1250,
  };

  const weeklyProgress = [
    { day: 'Mon', completed: true, points: 150 },
    { day: 'Tue', completed: true, points: 200 },
    { day: 'Wed', completed: false, points: 0 },
    { day: 'Thu', completed: true, points: 180 },
    { day: 'Fri', completed: false, points: 0 },
    { day: 'Sat', completed: false, points: 0 },
    { day: 'Sun', completed: false, points: 0 },
  ];

  const achievements = [
    { title: 'Week Warrior', description: 'Complete 5 workouts this week', progress: 3, total: 5, unlocked: false },
    { title: 'Cardio Champion', description: '30 min cardio sessions', progress: 2, total: 3, unlocked: false },
    { title: 'Strength Builder', description: 'Complete 10 strength workouts', progress: 7, total: 10, unlocked: false },
    { title: 'Early Bird', description: 'Complete morning workout', progress: 1, total: 1, unlocked: true },
  ];

  const handleUpdateWorkout = (workout) => {
    setSelectedWorkout(workout);
    setIsUpdateModalOpen(true);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Fitness Dashboard</h1>
          <p className="text-gray-600">Track your workouts, sync data, and achieve your fitness goals</p>
        </div>
        <Button onClick={() => setIsUpdateModalOpen(true)} className="bg-orange-600 hover:bg-orange-700">
          <Plus className="w-4 h-4 mr-2" />
          Log Workout
        </Button>
      </div>

      {/* Today's Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Steps</p>
                <p className="text-2xl font-bold text-gray-900">{todaysStats.steps.toLocaleString()}</p>
                <Progress value={(todaysStats.steps / todaysStats.stepsGoal) * 100} className="mt-2" />
              </div>
              <Footprints className="w-8 h-8 text-blue-600" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Calories</p>
                <p className="text-2xl font-bold text-gray-900">{todaysStats.calories}</p>
                <Progress value={(todaysStats.calories / todaysStats.caloriesGoal) * 100} className="mt-2" />
              </div>
              <Flame className="w-8 h-8 text-orange-600" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Active Minutes</p>
                <p className="text-2xl font-bold text-gray-900">{todaysStats.activeMinutes}</p>
                <Progress value={(todaysStats.activeMinutes / todaysStats.activeGoal) * 100} className="mt-2" />
              </div>
              <Clock className="w-8 h-8 text-green-600" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Heart Rate</p>
                <p className="text-2xl font-bold text-gray-900">{todaysStats.heartRate} BPM</p>
                <p className="text-xs text-green-600 mt-2">Resting rate</p>
              </div>
              <Heart className="w-8 h-8 text-red-600" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Main Content Tabs */}
      <Tabs defaultValue="overview" className="space-y-6">
        <div className="relative group">
          {/* Left Scroll Arrow */}
          <button
            onClick={() => {
              const container = document.querySelector('.fitness-tabs-scroll-container');
              if (container) {
                container.scrollBy({ left: -200, behavior: 'smooth' });
              }
            }}
            className="absolute left-0 top-1/2 -translate-y-1/2 z-10 bg-white/80 backdrop-blur-sm border border-gray-200 rounded-full p-1 shadow-md opacity-0 group-hover:opacity-100 transition-opacity duration-200 hover:bg-white"
          >
            <svg className="w-4 h-4 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
            </svg>
          </button>

          {/* Right Scroll Arrow */}
          <button
            onClick={() => {
              const container = document.querySelector('.fitness-tabs-scroll-container');
              if (container) {
                container.scrollBy({ left: 200, behavior: 'smooth' });
              }
            }}
            className="absolute right-0 top-1/2 -translate-y-1/2 z-10 bg-white/80 backdrop-blur-sm border border-gray-200 rounded-full p-1 shadow-md opacity-0 group-hover:opacity-100 transition-opacity duration-200 hover:bg-white"
          >
            <svg className="w-4 h-4 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
            </svg>
          </button>

          <div className="overflow-x-auto scrollbar-hide fitness-tabs-scroll-container">
            <TabsList className="flex w-max min-w-full space-x-1 px-4">
              <TabsTrigger value="overview" className="whitespace-nowrap">Overview</TabsTrigger>
              <TabsTrigger value="workouts" className="whitespace-nowrap">Workouts</TabsTrigger>
              <TabsTrigger value="checklist" className="whitespace-nowrap">Daily Tasks</TabsTrigger>
              <TabsTrigger value="progress" className="whitespace-nowrap">Progress</TabsTrigger>
              <TabsTrigger value="sync" className="whitespace-nowrap">Data Sync</TabsTrigger>
              <TabsTrigger value="motivation" className="whitespace-nowrap">Motivation</TabsTrigger>
            </TabsList>
          </div>
        </div>

        <TabsContent value="overview" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <HealthPointsCard points={todaysStats.healthPoints} />

            {/* Weekly Progress */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Calendar className="w-5 h-5" />
                  Weekly Progress
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex justify-between items-center mb-4">
                  {weeklyProgress.map((day, index) => (
                    <div key={index} className="flex flex-col items-center space-y-2">
                      <span className="text-xs font-medium text-gray-600">{day.day}</span>
                      <div className={`w-8 h-8 rounded-full flex items-center justify-center ${
                        day.completed ? 'bg-green-100 text-green-600' : 'bg-gray-100 text-gray-400'
                      }`}>
                        {day.completed ? <CheckCircle2 className="w-4 h-4" /> : <Circle className="w-4 h-4" />}
                      </div>
                      <span className="text-xs text-gray-500">{day.points}pt</span>
                    </div>
                  ))}
                </div>
                <div className="text-center">
                  <p className="text-sm text-gray-600">3 of 7 days completed</p>
                  <Progress value={42.8} className="mt-2" />
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Achievements */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Trophy className="w-5 h-5" />
                Fitness Achievements
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {achievements.map((achievement, index) => (
                  <div key={index} className={`p-4 rounded-lg border ${
                    achievement.unlocked ? 'border-yellow-300 bg-yellow-50' : 'border-gray-200 bg-gray-50'
                  }`}>
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <h3 className={`font-semibold ${
                          achievement.unlocked ? 'text-yellow-800' : 'text-gray-700'
                        }`}>
                          {achievement.title}
                        </h3>
                        <p className="text-sm text-gray-600 mt-1">{achievement.description}</p>
                        {!achievement.unlocked && (
                          <div className="mt-2">
                            <Progress value={(achievement.progress / achievement.total) * 100} className="h-2" />
                            <p className="text-xs text-gray-500 mt-1">
                              {achievement.progress}/{achievement.total}
                            </p>
                          </div>
                        )}
                      </div>
                      <Award className={`w-6 h-6 ${
                        achievement.unlocked ? 'text-yellow-600' : 'text-gray-400'
                      }`} />
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="workouts" className="space-y-6">
          <WorkoutRecommendations onUpdateWorkout={handleUpdateWorkout} />
        </TabsContent>

        <TabsContent value="checklist" className="space-y-6">
          <DailyWorkoutChecklist />
        </TabsContent>

        <TabsContent value="progress" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <BarChart3 className="w-5 h-5" />
                Fitness Analytics
              </CardTitle>
              <CardDescription>
                Your detailed fitness performance and trends
              </CardDescription>
            </CardHeader>
            <CardContent>
              <p className="text-center text-gray-500 py-8">
                Analytics charts will be displayed here, connected to the main Analytics page
              </p>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="sync" className="space-y-6">
          <FitnessDataSync />
        </TabsContent>

        <TabsContent value="motivation" className="space-y-6">
          <MotivationalTips />
        </TabsContent>
      </Tabs>

      {/* Update Workout Modal */}
      <UpdateWorkoutModal
        isOpen={isUpdateModalOpen}
        onClose={() => setIsUpdateModalOpen(false)}
        workout={selectedWorkout}
      />
    </div>
  );
};

export default FitnessScreen;
