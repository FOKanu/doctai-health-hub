
import React from 'react';
import { ChartContainer, ChartTooltip, ChartTooltipContent } from '@/components/ui/chart';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, ResponsiveContainer, BarChart, Bar } from 'recharts';

interface WeeklyData {
  day: string;
  steps: number;
  calories: number;
  workouts: number;
}

interface WeeklyTrendsChartProps {
  data: WeeklyData[];
}

export const WeeklyTrendsChart: React.FC<WeeklyTrendsChartProps> = ({ data }) => {
  const chartConfig = {
    steps: {
      label: "Steps",
      color: "#3b82f6",
    },
    calories: {
      label: "Calories",
      color: "#f59e0b",
    },
    workouts: {
      label: "Workouts",
      color: "#10b981",
    },
  };

  return (
    <div className="space-y-6">
      {/* Steps and Calories Line Chart */}
      <div>
        <h3 className="text-sm font-medium text-gray-700 mb-3">Daily Activity Trends</h3>
        <ChartContainer config={chartConfig} className="h-64">
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="day" />
            <YAxis />
            <ChartTooltip content={<ChartTooltipContent />} />
            <Line 
              type="monotone" 
              dataKey="steps" 
              stroke={chartConfig.steps.color}
              strokeWidth={2}
              dot={{ r: 4 }}
            />
            <Line 
              type="monotone" 
              dataKey="calories" 
              stroke={chartConfig.calories.color}
              strokeWidth={2}
              dot={{ r: 4 }}
            />
          </LineChart>
        </ChartContainer>
      </div>

      {/* Workouts Bar Chart */}
      <div>
        <h3 className="text-sm font-medium text-gray-700 mb-3">Weekly Workout Sessions</h3>
        <ChartContainer config={chartConfig} className="h-48">
          <BarChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="day" />
            <YAxis />
            <ChartTooltip content={<ChartTooltipContent />} />
            <Bar 
              dataKey="workouts" 
              fill={chartConfig.workouts.color}
              radius={[4, 4, 0, 0]}
            />
          </BarChart>
        </ChartContainer>
      </div>
    </div>
  );
};
