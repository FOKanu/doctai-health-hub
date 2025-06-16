
import React from 'react';
import { Plus, Activity, Droplets, Moon, Heart, Weight, Thermometer } from 'lucide-react';

interface WeeklyGoal {
  title: string;
  current: number;
  target: number;
  icon: React.ComponentType<any>;
  unit: string;
}

export const WeeklyGoalsSection: React.FC = () => {
  const weeklyGoals: WeeklyGoal[] = [
    { title: 'Daily Steps', current: 8500, target: 10000, icon: Activity, unit: 'steps' },
    { title: 'Water Intake', current: 6, target: 8, icon: Droplets, unit: 'glasses' },
    { title: 'Sleep Hours', current: 7.2, target: 8, icon: Moon, unit: 'hours' },
    { title: 'Heart Rate', current: 72, target: 65, icon: Heart, unit: 'bpm' },
    { title: 'Weight Goal', current: 75.2, target: 73.0, icon: Weight, unit: 'kg' },
    { title: 'Body Temp', current: 98.6, target: 98.6, icon: Thermometer, unit: '°F' },
  ];

  return (
    <div className="bg-white rounded-lg p-6 shadow-sm border border-gray-200">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold text-gray-900">Weekly Health Goals</h2>
        <button className="flex items-center space-x-2 text-blue-600 hover:text-blue-700 text-sm font-medium">
          <Plus className="w-4 h-4" />
          <span>Add Metric</span>
        </button>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {weeklyGoals.map((goal, index) => {
          const progress = (goal.current / goal.target) * 100;
          const isOnTrack = progress >= 80;
          return (
            <div key={index} className="bg-gray-50 rounded-lg p-4 space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <div className={`p-2 rounded-lg ${isOnTrack ? 'bg-green-100' : 'bg-orange-100'}`}>
                    <goal.icon className={`w-4 h-4 ${isOnTrack ? 'text-green-600' : 'text-orange-600'}`} />
                  </div>
                  <span className="text-sm font-medium text-gray-900">{goal.title}</span>
                </div>
                <span className="text-xs text-gray-500 bg-white px-2 py-1 rounded">
                  {goal.current} / {goal.target} {goal.unit}
                </span>
              </div>
              <div className="space-y-2">
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div 
                    className={`h-2 rounded-full transition-all duration-300 ${
                      isOnTrack ? 'bg-green-500' : 'bg-orange-500'
                    }`}
                    style={{ width: `${Math.min(progress, 100)}%` }}
                  ></div>
                </div>
                <div className="flex justify-between text-xs">
                  <span className={`font-medium ${isOnTrack ? 'text-green-600' : 'text-orange-600'}`}>
                    {Math.round(progress)}% complete
                  </span>
                  <span className="text-gray-500">
                    {isOnTrack ? 'On track' : 'Needs attention'}
                  </span>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
