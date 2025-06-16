
import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Button } from '../ui/button';
import { Badge } from '../ui/badge';
import { ChevronLeft, ChevronRight, Clock, Flame, Utensils } from 'lucide-react';

export const WeeklyMenu: React.FC = () => {
  const [selectedDay, setSelectedDay] = useState(0);

  const days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
  const fullDays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];

  const weeklyMenu = [
    {
      day: 'Monday',
      meals: [
        {
          type: 'Breakfast',
          name: 'Greek Yogurt Parfait',
          calories: 320,
          protein: 25,
          carbs: 35,
          fat: 8,
          prepTime: 5,
          image: 'https://images.unsplash.com/photo-1488477181946-6428a0291777?w=300&h=200&fit=crop'
        },
        {
          type: 'Lunch',
          name: 'Quinoa Buddha Bowl',
          calories: 480,
          protein: 18,
          carbs: 65,
          fat: 16,
          prepTime: 15,
          image: 'https://images.unsplash.com/photo-1512621776951-a57141f2eefd?w=300&h=200&fit=crop'
        },
        {
          type: 'Dinner',
          name: 'Grilled Chicken & Vegetables',
          calories: 420,
          protein: 35,
          carbs: 25,
          fat: 18,
          prepTime: 25,
          image: 'https://images.unsplash.com/photo-1604503468506-a8da13d82791?w=300&h=200&fit=crop'
        },
        {
          type: 'Snack',
          name: 'Almond & Apple',
          calories: 180,
          protein: 6,
          carbs: 18,
          fat: 12,
          prepTime: 2,
          image: 'https://images.unsplash.com/photo-1619566636858-adf3ef46400b?w=300&h=200&fit=crop'
        }
      ]
    }
    // For brevity, showing only Monday. In real app, would have all 7 days
  ];

  const currentDayMenu = weeklyMenu[0]; // Using Monday as example
  const totalCalories = currentDayMenu.meals.reduce((sum, meal) => sum + meal.calories, 0);
  const totalProtein = currentDayMenu.meals.reduce((sum, meal) => sum + meal.protein, 0);
  const totalCarbs = currentDayMenu.meals.reduce((sum, meal) => sum + meal.carbs, 0);
  const totalFat = currentDayMenu.meals.reduce((sum, meal) => sum + meal.fat, 0);

  return (
    <div className="space-y-4">
      {/* Week Navigation */}
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold">Weekly Menu</h2>
        <div className="flex items-center space-x-2">
          <Button variant="outline" size="sm">
            <ChevronLeft className="w-4 h-4" />
          </Button>
          <span className="text-sm font-medium">Week of Dec 16</span>
          <Button variant="outline" size="sm">
            <ChevronRight className="w-4 h-4" />
          </Button>
        </div>
      </div>

      {/* Day Selector */}
      <div className="flex space-x-2 overflow-x-auto pb-2">
        {days.map((day, index) => (
          <Button
            key={day}
            variant={selectedDay === index ? "default" : "outline"}
            size="sm"
            className="min-w-[60px]"
            onClick={() => setSelectedDay(index)}
          >
            {day}
          </Button>
        ))}
      </div>

      {/* Daily Summary */}
      <Card className="bg-gradient-to-r from-green-50 to-blue-50">
        <CardContent className="pt-4">
          <h3 className="font-semibold mb-2">{fullDays[selectedDay]} Summary</h3>
          <div className="grid grid-cols-4 gap-4 text-center">
            <div>
              <div className="text-lg font-bold text-green-600">{totalCalories}</div>
              <div className="text-xs text-gray-600">Calories</div>
            </div>
            <div>
              <div className="text-lg font-bold text-blue-600">{totalProtein}g</div>
              <div className="text-xs text-gray-600">Protein</div>
            </div>
            <div>
              <div className="text-lg font-bold text-orange-600">{totalCarbs}g</div>
              <div className="text-xs text-gray-600">Carbs</div>
            </div>
            <div>
              <div className="text-lg font-bold text-purple-600">{totalFat}g</div>
              <div className="text-xs text-gray-600">Fat</div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Meals List */}
      <div className="space-y-3">
        {currentDayMenu.meals.map((meal, index) => (
          <Card key={index}>
            <CardContent className="p-4">
              <div className="flex items-center space-x-3">
                <img
                  src={meal.image}
                  alt={meal.name}
                  className="w-16 h-16 rounded-lg object-cover"
                />
                <div className="flex-1">
                  <div className="flex items-center justify-between mb-1">
                    <Badge variant="outline" className="text-xs">
                      {meal.type}
                    </Badge>
                    <div className="flex items-center text-xs text-gray-500">
                      <Clock className="w-3 h-3 mr-1" />
                      {meal.prepTime}m
                    </div>
                  </div>
                  <h4 className="font-medium">{meal.name}</h4>
                  <div className="flex items-center space-x-3 text-xs text-gray-600 mt-1">
                    <span className="flex items-center">
                      <Flame className="w-3 h-3 mr-1" />
                      {meal.calories}
                    </span>
                    <span>P: {meal.protein}g</span>
                    <span>C: {meal.carbs}g</span>
                    <span>F: {meal.fat}g</span>
                  </div>
                </div>
                <Button variant="ghost" size="sm">
                  <Utensils className="w-4 h-4" />
                </Button>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Action Buttons */}
      <div className="flex gap-2">
        <Button variant="outline" className="flex-1">
          Generate New Menu
        </Button>
        <Button className="flex-1">
          Save to Favorites
        </Button>
      </div>
    </div>
  );
};
