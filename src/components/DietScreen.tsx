
import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Progress } from './ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Apple, Utensils, BarChart3, Camera, Search, Plus, Target, Zap } from 'lucide-react';
import { PersonalizedMealPlans } from './diet/PersonalizedMealPlans';
import { WeeklyMenu } from './diet/WeeklyMenu';
import { MealLogging } from './diet/MealLogging';
import { NutrientTracker } from './diet/NutrientTracker';
import { SmartSuggestions } from './diet/SmartSuggestions';

const DietScreen = () => {
  const [activeTab, setActiveTab] = useState('meal-plans');

  // Mock user profile data
  const userProfile = {
    goals: ['weight_loss', 'muscle_gain'],
    allergies: ['nuts', 'dairy'],
    activityLevel: 'moderate',
    preferences: ['vegetarian']
  };

  // Mock daily nutrition goals
  const dailyGoals = {
    calories: 2000,
    protein: 150,
    carbs: 200,
    fat: 67,
    fiber: 25,
    water: 8 // glasses
  };

  // Mock current progress
  const currentProgress = {
    calories: 1420,
    protein: 95,
    carbs: 140,
    fat: 45,
    fiber: 18,
    water: 5
  };

  return (
    <div className="min-h-screen bg-gray-50 pb-20">
      {/* Header */}
      <div className="bg-white shadow-sm">
        <div className="flex items-center justify-between p-4">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-green-100 rounded-lg">
              <Apple className="w-6 h-6 text-green-600" />
            </div>
            <div>
              <h1 className="text-xl font-semibold text-gray-800">Diet Plan</h1>
              <p className="text-sm text-gray-600">Personalized nutrition tracking</p>
            </div>
          </div>
          <Button variant="outline" size="sm">
            <Target className="w-4 h-4 mr-2" />
            Goals
          </Button>
        </div>
      </div>

      <div className="p-4">
        {/* Quick Stats */}
        <Card className="mb-6">
          <CardHeader className="pb-3">
            <CardTitle className="text-lg">Today's Progress</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span>Calories</span>
                  <span>{currentProgress.calories}/{dailyGoals.calories}</span>
                </div>
                <Progress value={(currentProgress.calories / dailyGoals.calories) * 100} className="h-2" />
              </div>
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span>Protein</span>
                  <span>{currentProgress.protein}g/{dailyGoals.protein}g</span>
                </div>
                <Progress value={(currentProgress.protein / dailyGoals.protein) * 100} className="h-2" />
              </div>
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span>Carbs</span>
                  <span>{currentProgress.carbs}g/{dailyGoals.carbs}g</span>
                </div>
                <Progress value={(currentProgress.carbs / dailyGoals.carbs) * 100} className="h-2" />
              </div>
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span>Fat</span>
                  <span>{currentProgress.fat}g/{dailyGoals.fat}g</span>
                </div>
                <Progress value={(currentProgress.fat / dailyGoals.fat) * 100} className="h-2" />
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Main Content Tabs */}
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="grid w-full grid-cols-5">
            <TabsTrigger value="meal-plans" className="text-xs">Plans</TabsTrigger>
            <TabsTrigger value="weekly-menu" className="text-xs">Menu</TabsTrigger>
            <TabsTrigger value="log-meal" className="text-xs">Log</TabsTrigger>
            <TabsTrigger value="tracker" className="text-xs">Track</TabsTrigger>
            <TabsTrigger value="suggestions" className="text-xs">Tips</TabsTrigger>
          </TabsList>

          <TabsContent value="meal-plans" className="mt-4">
            <PersonalizedMealPlans userProfile={userProfile} />
          </TabsContent>

          <TabsContent value="weekly-menu" className="mt-4">
            <WeeklyMenu />
          </TabsContent>

          <TabsContent value="log-meal" className="mt-4">
            <MealLogging />
          </TabsContent>

          <TabsContent value="tracker" className="mt-4">
            <NutrientTracker 
              dailyGoals={dailyGoals} 
              currentProgress={currentProgress} 
            />
          </TabsContent>

          <TabsContent value="suggestions" className="mt-4">
            <SmartSuggestions />
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
};

export default DietScreen;
