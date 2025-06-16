import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '../ui/input';
import { Badge } from '../ui/badge';
import { Camera, Search, Plus, Scan, Clock, Trash2 } from 'lucide-react';

export const MealLogging: React.FC = () => {
  const [searchTerm, setSearchTerm] = useState('');
  const [loggedMeals, setLoggedMeals] = useState([
    {
      id: 1,
      name: 'Greek Yogurt',
      time: '8:30 AM',
      calories: 120,
      protein: 15,
      carbs: 8,
      fat: 5,
      quantity: '1 cup'
    },
    {
      id: 2,
      name: 'Banana',
      time: '8:35 AM',
      calories: 105,
      protein: 1,
      carbs: 27,
      fat: 0,
      quantity: '1 medium'
    }
  ]);

  const quickAddFoods = [
    { name: 'Apple', calories: 95, protein: 0.5, carbs: 25, fat: 0.3 },
    { name: 'Almonds (1oz)', calories: 164, protein: 6, carbs: 6, fat: 14 },
    { name: 'Chicken Breast (100g)', calories: 165, protein: 31, carbs: 0, fat: 3.6 },
    { name: 'Brown Rice (1 cup)', calories: 216, protein: 5, carbs: 45, fat: 1.8 },
    { name: 'Avocado (half)', calories: 160, protein: 2, carbs: 9, fat: 15 },
    { name: 'Protein Shake', calories: 120, protein: 25, carbs: 3, fat: 1 }
  ];

  const removeMeal = (id: number) => {
    setLoggedMeals(loggedMeals.filter(meal => meal.id !== id));
  };

  const totalCalories = loggedMeals.reduce((sum, meal) => sum + meal.calories, 0);

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold">Log Your Meals</h2>
        <Badge variant="secondary">
          {totalCalories} calories today
        </Badge>
      </div>

      {/* Quick Add Methods */}
      <div className="grid grid-cols-3 gap-2">
        <Button variant="outline" className="flex flex-col items-center py-4">
          <Camera className="w-5 h-5 mb-1" />
          <span className="text-xs">Photo</span>
        </Button>
        <Button variant="outline" className="flex flex-col items-center py-4">
          <Scan className="w-5 h-5 mb-1" />
          <span className="text-xs">Barcode</span>
        </Button>
        <Button variant="outline" className="flex flex-col items-center py-4">
          <Search className="w-5 h-5 mb-1" />
          <span className="text-xs">Search</span>
        </Button>
      </div>

      {/* Search Bar */}
      <div className="relative">
        <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
        <Input
          placeholder="Search foods..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="pl-10"
        />
      </div>

      {/* Quick Add Foods */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Quick Add</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 gap-2">
            {quickAddFoods.map((food, index) => (
              <div key={index} className="flex items-center justify-between p-2 bg-gray-50 rounded-lg">
                <div>
                  <div className="font-medium text-sm">{food.name}</div>
                  <div className="text-xs text-gray-600">
                    {food.calories} cal • P: {food.protein}g • C: {food.carbs}g • F: {food.fat}g
                  </div>
                </div>
                <Button size="sm" variant="ghost">
                  <Plus className="w-4 h-4" />
                </Button>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Today's Logged Meals */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Today's Meals</CardTitle>
        </CardHeader>
        <CardContent>
          {loggedMeals.length === 0 ? (
            <div className="text-center py-8 text-gray-500">
              <Utensils className="w-8 h-8 mx-auto mb-2 opacity-50" />
              <p>No meals logged yet</p>
            </div>
          ) : (
            <div className="space-y-3">
              {loggedMeals.map((meal) => (
                <div key={meal.id} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                  <div className="flex-1">
                    <div className="flex items-center justify-between">
                      <h4 className="font-medium">{meal.name}</h4>
                      <div className="flex items-center text-xs text-gray-500">
                        <Clock className="w-3 h-3 mr-1" />
                        {meal.time}
                      </div>
                    </div>
                    <div className="text-sm text-gray-600 mt-1">
                      {meal.quantity} • {meal.calories} cal
                    </div>
                    <div className="text-xs text-gray-500">
                      P: {meal.protein}g • C: {meal.carbs}g • F: {meal.fat}g
                    </div>
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => removeMeal(meal.id)}
                    className="text-red-500 hover:text-red-700 hover:bg-red-50"
                  >
                    <Trash2 className="w-4 h-4" />
                  </Button>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Manual Entry */}
      <Button variant="outline" className="w-full">
        <Plus className="w-4 h-4 mr-2" />
        Add Custom Food
      </Button>
    </div>
  );
};
