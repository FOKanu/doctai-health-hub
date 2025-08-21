
import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Apple,
  Calendar,
  Target,
  BarChart3,
  Plus,
  Camera,
  Search,
  Brain,
  Clock,
  Zap,
  Heart
} from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { PersonalizedMealPlan } from './diet/PersonalizedMealPlan';
import { WeeklyMenu } from './diet/WeeklyMenu';
import { MealLogging } from './diet/MealLogging';
import { NutrientTracker } from './diet/NutrientTracker';
import { SmartSuggestions } from './diet/SmartSuggestions';

const DietPlanScreen = () => {
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState('overview');

  // Sample user profile and nutrition data
  const userProfile = {
    age: 28,
    weight: 70,
    height: 175,
    activityLevel: 'moderate',
    goal: 'maintenance',
    allergies: ['nuts', 'dairy'],
    preferences: ['vegetarian']
  };

  const dailyTargets = {
    calories: 2200,
    protein: 110, // grams
    carbs: 275,   // grams
    fat: 73,      // grams
    fiber: 25,    // grams
    water: 2500   // ml
  };

  const currentIntake = {
    calories: 1650,
    protein: 85,
    carbs: 180,
    fat: 55,
    fiber: 18,
    water: 1800
  };

  const todaysMeals = [
    {
      id: 1,
      type: 'breakfast',
      name: 'Oatmeal with Berries',
      calories: 320,
      protein: 12,
      carbs: 58,
      fat: 6,
      time: '08:00',
      logged: true
    },
    {
      id: 2,
      type: 'lunch',
      name: 'Quinoa Buddha Bowl',
      calories: 480,
      protein: 18,
      carbs: 65,
      fat: 16,
      time: '13:00',
      logged: true
    },
    {
      id: 3,
      type: 'snack',
      name: 'Greek Yogurt & Almonds',
      calories: 250,
      protein: 15,
      carbs: 12,
      fat: 14,
      time: '16:00',
      logged: true
    },
    {
      id: 4,
      type: 'dinner',
      name: 'Grilled Salmon & Vegetables',
      calories: 600,
      protein: 40,
      carbs: 45,
      fat: 19,
      time: '19:00',
      logged: false
    }
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Diet Plan</h1>
          <p className="text-gray-600">Personalized nutrition guidance for your health goals</p>
        </div>
        <div className="flex items-center gap-3">
          <Badge variant="outline" className="text-green-600 border-green-200">
            <Target className="w-3 h-3 mr-1" />
            75% Daily Goal
          </Badge>
          <Button variant="outline" size="sm" onClick={() => navigate('/patient/analytics')}>
            <BarChart3 className="w-4 h-4 mr-2" />
            View Analytics
          </Button>
        </div>
      </div>

      {/* Quick Stats Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Calories</p>
                <p className="text-2xl font-bold text-gray-900">{currentIntake.calories}</p>
                <p className="text-xs text-gray-500">of {dailyTargets.calories}</p>
              </div>
              <Zap className="w-8 h-8 text-orange-500" />
            </div>
            <Progress value={(currentIntake.calories / dailyTargets.calories) * 100} className="mt-2" />
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Protein</p>
                <p className="text-2xl font-bold text-gray-900">{currentIntake.protein}g</p>
                <p className="text-xs text-gray-500">of {dailyTargets.protein}g</p>
              </div>
              <Heart className="w-8 h-8 text-red-500" />
            </div>
            <Progress value={(currentIntake.protein / dailyTargets.protein) * 100} className="mt-2" />
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Carbs</p>
                <p className="text-2xl font-bold text-gray-900">{currentIntake.carbs}g</p>
                <p className="text-xs text-gray-500">of {dailyTargets.carbs}g</p>
              </div>
              <Apple className="w-8 h-8 text-green-500" />
            </div>
            <Progress value={(currentIntake.carbs / dailyTargets.carbs) * 100} className="mt-2" />
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Water</p>
                <p className="text-2xl font-bold text-gray-900">{(currentIntake.water / 1000).toFixed(1)}L</p>
                <p className="text-xs text-gray-500">of {(dailyTargets.water / 1000).toFixed(1)}L</p>
              </div>
              <div className="w-8 h-8 bg-blue-500 rounded-full flex items-center justify-center">
                <div className="w-4 h-4 bg-white rounded-full"></div>
              </div>
            </div>
            <Progress value={(currentIntake.water / dailyTargets.water) * 100} className="mt-2" />
          </CardContent>
        </Card>
      </div>

      {/* Main Content Tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <div className="relative group">
          {/* Left Scroll Arrow */}
          <button
            onClick={() => {
              const container = document.querySelector('.diet-tabs-scroll-container');
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
              const container = document.querySelector('.diet-tabs-scroll-container');
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

          <div className="overflow-x-auto scrollbar-hide diet-tabs-scroll-container">
            <TabsList className="flex w-max min-w-full space-x-1 px-4">
              <TabsTrigger value="overview" className="whitespace-nowrap">Overview</TabsTrigger>
              <TabsTrigger value="meal-plan" className="whitespace-nowrap">Meal Plan</TabsTrigger>
              <TabsTrigger value="logging" className="whitespace-nowrap">Log Meals</TabsTrigger>
              <TabsTrigger value="tracker" className="whitespace-nowrap">Nutrients</TabsTrigger>
              <TabsTrigger value="suggestions" className="whitespace-nowrap">Smart Tips</TabsTrigger>
            </TabsList>
          </div>
        </div>

        <TabsContent value="overview">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <PersonalizedMealPlan userProfile={userProfile} />
            <SmartSuggestions currentIntake={currentIntake} dailyTargets={dailyTargets} />
          </div>
        </TabsContent>

        <TabsContent value="meal-plan">
          <WeeklyMenu meals={todaysMeals} />
        </TabsContent>

        <TabsContent value="logging">
          <MealLogging onMealLogged={(meal) => console.log('Meal logged:', meal)} />
        </TabsContent>

        <TabsContent value="tracker">
          <NutrientTracker currentIntake={currentIntake} dailyTargets={dailyTargets} />
        </TabsContent>

        <TabsContent value="suggestions">
          <SmartSuggestions currentIntake={currentIntake} dailyTargets={dailyTargets} detailed={true} />
        </TabsContent>
      </Tabs>

      {/* Quick Actions */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Plus className="w-5 h-5" />
            Quick Actions
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <Button variant="outline" className="h-auto p-4 flex flex-col items-center space-y-2">
              <Camera className="w-6 h-6" />
              <span className="text-sm">Scan Barcode</span>
            </Button>
            <Button variant="outline" className="h-auto p-4 flex flex-col items-center space-y-2">
              <Search className="w-6 h-6" />
              <span className="text-sm">Search Food</span>
            </Button>
            <Button variant="outline" className="h-auto p-4 flex flex-col items-center space-y-2">
              <Brain className="w-6 h-6" />
              <span className="text-sm">AI Suggestions</span>
            </Button>
            <Button variant="outline" className="h-auto p-4 flex flex-col items-center space-y-2">
              <Calendar className="w-6 h-6" />
              <span className="text-sm">Meal Planner</span>
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default DietPlanScreen;
