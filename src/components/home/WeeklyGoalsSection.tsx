
import React from 'react';
import { Plus, Activity, Droplets, Moon, Heart, Weight, Thermometer } from 'lucide-react';

interface WeeklyGoal {
  title: string;
  current: number;
  target: number;
  icon: React.ComponentType<{ className?: string }>;
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
    <div className="bg-card rounded-lg p-6 shadow-sm border border-border">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold text-foreground">Weekly Health Goals</h2>
        <button className="flex items-center space-x-2 text-primary hover:text-primary/80 text-sm font-medium">
          <Plus className="w-4 h-4" />
          <span>Add Metric</span>
        </button>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {weeklyGoals.map((goal, index) => {
          const progress = (goal.current / goal.target) * 100;
          const isOnTrack = progress >= 80;
          return (
            <div key={index} className="bg-muted rounded-lg p-4 space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <div className={`p-2 rounded-lg ${isOnTrack ? 'bg-primary/20' : 'bg-destructive/20'}`}>
                    <goal.icon className={`w-4 h-4 ${isOnTrack ? 'text-primary' : 'text-destructive'}`} />
                  </div>
                  <span className="text-sm font-medium text-foreground">{goal.title}</span>
                </div>
                <span className="text-xs text-muted-foreground bg-background px-2 py-1 rounded">
                  {goal.current} / {goal.target} {goal.unit}
                </span>
              </div>
                <div className="space-y-2">
                  <div className="w-full bg-border rounded-full h-2">
                    <div
                      className={`h-2 rounded-full transition-all duration-300 ${
                        isOnTrack ? 'bg-primary' : 'bg-destructive'
                      }`}
                      style={{ width: `${Math.min(progress, 100)}%` }}
                    ></div>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span className={`font-medium ${isOnTrack ? 'text-primary' : 'text-destructive'}`}>
                      {Math.round(progress)}% complete
                    </span>
                    <span className="text-muted-foreground">
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
