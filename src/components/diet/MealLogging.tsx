
import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { Camera, Search, Plus, Scan, Clock, Utensils } from 'lucide-react';

interface MealLoggingProps {
  onMealLogged: (meal: any) => void;
}

export const MealLogging: React.FC<MealLoggingProps> = ({ onMealLogged }) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedMeal, setSelectedMeal] = useState<any>(null);

  const popularFoods = [
    { name: 'Banana', calories: 105, protein: 1.3, carbs: 27, fat: 0.3, serving: '1 medium' },
    { name: 'Greek Yogurt', calories: 130, protein: 23, carbs: 9, fat: 0, serving: '1 cup' },
    { name: 'Oatmeal', calories: 158, protein: 6, carbs: 28, fat: 3, serving: '1 cup cooked' },
    { name: 'Chicken Breast', calories: 165, protein: 31, carbs: 0, fat: 3.6, serving: '100g' },
    { name: 'Brown Rice', calories: 216, protein: 5, carbs: 45, fat: 1.8, serving: '1 cup cooked' },
    { name: 'Almonds', calories: 164, protein: 6, carbs: 6, fat: 14, serving: '28g (24 nuts)' },
  ];

  const recentFoods = [
    { name: 'Quinoa Buddha Bowl', calories: 480, protein: 18, carbs: 65, fat: 16 },
    { name: 'Green Smoothie', calories: 220, protein: 8, carbs: 45, fat: 3 },
    { name: 'Avocado Toast', calories: 350, protein: 12, carbs: 30, fat: 22 },
  ];

  const mealTypes = ['Breakfast', 'Lunch', 'Dinner', 'Snack'];

  const filteredFoods = popularFoods.filter(food =>
    food.name.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div className="space-y-6">
      {/* Quick Add Methods */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card className="cursor-pointer hover:shadow-md transition-shadow">
          <CardContent className="p-6 text-center">
            <Camera className="w-12 h-12 mx-auto text-blue-500 mb-4" />
            <h3 className="font-semibold text-gray-900 mb-2">Scan Barcode</h3>
            <p className="text-sm text-gray-600">Quick scan packaged foods</p>
          </CardContent>
        </Card>

        <Card className="cursor-pointer hover:shadow-md transition-shadow">
          <CardContent className="p-6 text-center">
            <Utensils className="w-12 h-12 mx-auto text-green-500 mb-4" />
            <h3 className="font-semibold text-gray-900 mb-2">Photo Recognition</h3>
            <p className="text-sm text-gray-600">Snap a photo of your meal</p>
          </CardContent>
        </Card>

        <Card className="cursor-pointer hover:shadow-md transition-shadow">
          <CardContent className="p-6 text-center">
            <Search className="w-12 h-12 mx-auto text-purple-500 mb-4" />
            <h3 className="font-semibold text-gray-900 mb-2">Search Database</h3>
            <p className="text-sm text-gray-600">Find from millions of foods</p>
          </CardContent>
        </Card>
      </div>

      {/* Search Foods */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Search className="w-5 h-5" />
            Search Foods
          </CardTitle>
          <CardDescription>
            Search our database or add custom foods
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex gap-2">
            <Input
              placeholder="Search foods (e.g., chicken breast, apple)..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="flex-1"
            />
            <Button>
              <Search className="w-4 h-4" />
            </Button>
          </div>

          {/* Search Results */}
          {searchQuery && (
            <div className="space-y-2">
              <h4 className="font-medium text-gray-900">Search Results</h4>
              {filteredFoods.map((food, index) => (
                <div key={index} className="flex items-center justify-between p-3 border rounded-lg hover:bg-gray-50">
                  <div>
                    <h5 className="font-medium">{food.name}</h5>
                    <p className="text-sm text-gray-600">{food.serving} - {food.calories} cal</p>
                  </div>
                  <Button size="sm" onClick={() => setSelectedMeal(food)}>
                    <Plus className="w-4 h-4" />
                  </Button>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Recent & Popular Foods */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Recent Foods */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Clock className="w-5 h-5" />
              Recent Foods
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            {recentFoods.map((food, index) => (
              <div key={index} className="flex items-center justify-between p-3 border rounded-lg hover:bg-gray-50">
                <div>
                  <h5 className="font-medium">{food.name}</h5>
                  <p className="text-sm text-gray-600">{food.calories} cal</p>
                </div>
                <Button size="sm" variant="outline">
                  <Plus className="w-4 h-4" />
                </Button>
              </div>
            ))}
          </CardContent>
        </Card>

        {/* Popular Foods */}
        <Card>
          <CardHeader>
            <CardTitle>Popular Foods</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            {popularFoods.slice(0, 3).map((food, index) => (
              <div key={index} className="flex items-center justify-between p-3 border rounded-lg hover:bg-gray-50">
                <div>
                  <h5 className="font-medium">{food.name}</h5>
                  <p className="text-sm text-gray-600">{food.serving} - {food.calories} cal</p>
                </div>
                <Button size="sm" variant="outline" onClick={() => setSelectedMeal(food)}>
                  <Plus className="w-4 h-4" />
                </Button>
              </div>
            ))}
          </CardContent>
        </Card>
      </div>

      {/* Quick Add by Meal Type */}
      <Card>
        <CardHeader>
          <CardTitle>Quick Add to Meal</CardTitle>
          <CardDescription>
            Add directly to a specific meal
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {mealTypes.map((type) => (
              <Button key={type} variant="outline" className="h-auto p-4 flex flex-col">
                <span className="text-2xl mb-2">
                  {type === 'Breakfast' ? '🌅' : 
                   type === 'Lunch' ? '☀️' : 
                   type === 'Dinner' ? '🌙' : '🍎'}
                </span>
                <span className="text-sm">{type}</span>
              </Button>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Manual Entry */}
      <Card>
        <CardHeader>
          <CardTitle>Manual Entry</CardTitle>
          <CardDescription>
            Add custom foods or recipes
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <Label htmlFor="food-name">Food Name</Label>
              <Input id="food-name" placeholder="e.g., Homemade Pasta" />
            </div>
            <div>
              <Label htmlFor="serving">Serving Size</Label>
              <Input id="serving" placeholder="e.g., 1 cup" />
            </div>
          </div>
          <div className="grid grid-cols-4 gap-4">
            <div>
              <Label htmlFor="calories">Calories</Label>
              <Input id="calories" type="number" placeholder="300" />
            </div>
            <div>
              <Label htmlFor="protein">Protein (g)</Label>
              <Input id="protein" type="number" placeholder="15" />
            </div>
            <div>
              <Label htmlFor="carbs">Carbs (g)</Label>
              <Input id="carbs" type="number" placeholder="45" />
            </div>
            <div>
              <Label htmlFor="fat">Fat (g)</Label>
              <Input id="fat" type="number" placeholder="10" />
            </div>
          </div>
          <Button className="w-full">
            <Plus className="w-4 h-4 mr-2" />
            Add Custom Food
          </Button>
        </CardContent>
      </Card>
    </div>
  );
};
