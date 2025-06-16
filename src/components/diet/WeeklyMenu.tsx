
import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Calendar, Clock, ChefHat, MoreHorizontal } from 'lucide-react';

interface Meal {
  id: number;
  type: string;
  name: string;
  calories: number;
  protein: number;
  carbs: number;
  fat: number;
  time: string;
  logged: boolean;
}

interface WeeklyMenuProps {
  meals: Meal[];
  userProfile: any;
}

export const WeeklyMenu: React.FC<WeeklyMenuProps> = ({ meals, userProfile }) => {
  const [selectedDay, setSelectedDay] = useState('today');

  const weekDays = [
    { key: 'today', label: 'Today', date: 'Dec 16' },
    { key: 'tomorrow', label: 'Tomorrow', date: 'Dec 17' },
    { key: 'wed', label: 'Wednesday', date: 'Dec 18' },
    { key: 'thu', label: 'Thursday', date: 'Dec 19' },
    { key: 'fri', label: 'Friday', date: 'Dec 20' },
    { key: 'sat', label: 'Saturday', date: 'Dec 21' },
    { key: 'sun', label: 'Sunday', date: 'Dec 22' },
  ];

  const getMealIcon = (type: string) => {
    switch (type) {
      case 'breakfast': return '🌅';
      case 'lunch': return '☀️';
      case 'snack': return '🍎';
      case 'dinner': return '🌙';
      default: return '🍽️';
    }
  };

  const totalCalories = meals.reduce((sum, meal) => sum + meal.calories, 0);
  const totalProtein = meals.reduce((sum, meal) => sum + meal.protein, 0);

  return (
    <div className="space-y-6">
      {/* Day Selector */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Calendar className="w-5 h-5" />
            Weekly Menu
          </CardTitle>
          <CardDescription>
            Your personalized meal plan for the week
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex space-x-2 overflow-x-auto pb-2">
            {weekDays.map((day) => (
              <Button
                key={day.key}
                variant={selectedDay === day.key ? "default" : "outline"}
                onClick={() => setSelectedDay(day.key)}
                className="min-w-fit"
              >
                <div className="text-center">
                  <div className="text-sm font-medium">{day.label}</div>
                  <div className="text-xs opacity-70">{day.date}</div>
                </div>
              </Button>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Daily Summary */}
      <Card>
        <CardContent className="p-6">
          <div className="grid grid-cols-3 gap-4 text-center">
            <div>
              <p className="text-2xl font-bold text-gray-900">{totalCalories}</p>
              <p className="text-sm text-gray-600">Total Calories</p>
            </div>
            <div>
              <p className="text-2xl font-bold text-gray-900">{totalProtein}g</p>
              <p className="text-sm text-gray-600">Protein</p>
            </div>
            <div>
              <p className="text-2xl font-bold text-gray-900">{meals.length}</p>
              <p className="text-sm text-gray-600">Meals</p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Meals List */}
      <div className="space-y-4">
        {meals.map((meal) => (
          <Card key={meal.id} className={meal.logged ? 'bg-green-50 border-green-200' : ''}>
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-4">
                  <div className="text-2xl">{getMealIcon(meal.type)}</div>
                  <div>
                    <h3 className="font-semibold text-gray-900">{meal.name}</h3>
                    <div className="flex items-center space-x-4 text-sm text-gray-600">
                      <span className="flex items-center">
                        <Clock className="w-3 h-3 mr-1" />
                        {meal.time}
                      </span>
                      <span>{meal.calories} cal</span>
                      <span>{meal.protein}g protein</span>
                    </div>
                  </div>
                </div>
                <div className="flex items-center space-x-2">
                  {meal.logged && (
                    <Badge variant="secondary" className="text-green-700 bg-green-100">
                      Logged
                    </Badge>
                  )}
                  <Button variant="ghost" size="sm">
                    <MoreHorizontal className="w-4 h-4" />
                  </Button>
                </div>
              </div>
              
              {/* Macros breakdown */}
              <div className="mt-3 grid grid-cols-3 gap-2 text-xs">
                <div className="bg-orange-100 p-2 rounded text-center">
                  <p className="font-medium text-orange-800">{meal.carbs}g</p>
                  <p className="text-orange-600">Carbs</p>
                </div>
                <div className="bg-red-100 p-2 rounded text-center">
                  <p className="font-medium text-red-800">{meal.protein}g</p>
                  <p className="text-red-600">Protein</p>
                </div>
                <div className="bg-yellow-100 p-2 rounded text-center">
                  <p className="font-medium text-yellow-800">{meal.fat}g</p>
                  <p className="text-yellow-600">Fat</p>
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Generate New Menu */}
      <Card>
        <CardContent className="p-6 text-center">
          <ChefHat className="w-12 h-12 mx-auto text-gray-400 mb-4" />
          <h3 className="font-semibold text-gray-900 mb-2">Want a fresh menu?</h3>
          <p className="text-sm text-gray-600 mb-4">
            Generate a new weekly meal plan based on your preferences
          </p>
          <Button>Generate New Menu</Button>
        </CardContent>
      </Card>
    </div>
  );
};
