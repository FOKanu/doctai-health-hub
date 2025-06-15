
import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import { TrendingUp, TrendingDown, Target, Activity } from 'lucide-react';

interface HealthOverviewProps {
  dateRange: string;
}

export const HealthOverview: React.FC<HealthOverviewProps> = ({ dateRange }) => {
  // Sample data for health score trend
  const healthScoreData = [
    { date: '2024-01-01', score: 75 },
    { date: '2024-01-08', score: 78 },
    { date: '2024-01-15', score: 82 },
    { date: '2024-01-22', score: 79 },
    { date: '2024-01-29', score: 85 },
    { date: '2024-02-05', score: 87 },
    { date: '2024-02-12', score: 84 },
  ];

  // Sample data for goal completion
  const goalCompletionData = [
    { name: 'Completed', value: 75, color: '#10b981' },
    { name: 'In Progress', value: 15, color: '#f59e0b' },
    { name: 'Missed', value: 10, color: '#ef4444' },
  ];

  const currentScore = 84;
  const previousScore = 79;
  const scoreChange = currentScore - previousScore;

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* Health Score Trend */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Activity className="w-5 h-5 text-blue-600" />
            Health Score Trend
          </CardTitle>
          <CardDescription>
            AI-generated overall health assessment over time
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <span className="text-3xl font-bold text-gray-900">{currentScore}</span>
              <div className="flex items-center gap-1">
                {scoreChange > 0 ? (
                  <TrendingUp className="w-4 h-4 text-green-600" />
                ) : scoreChange < 0 ? (
                  <TrendingDown className="w-4 h-4 text-red-600" />
                ) : null}
                <span className={`text-sm font-medium ${
                  scoreChange > 0 ? 'text-green-600' : scoreChange < 0 ? 'text-red-600' : 'text-gray-600'
                }`}>
                  {scoreChange > 0 ? '+' : ''}{scoreChange}
                </span>
              </div>
            </div>
            <div className="text-sm text-gray-500">
              vs. previous week
            </div>
          </div>
          
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={healthScoreData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="date" 
                tickFormatter={(value) => new Date(value).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
              />
              <YAxis domain={[60, 100]} />
              <Tooltip 
                labelFormatter={(value) => new Date(value).toLocaleDateString()}
                formatter={(value) => [`${value}`, 'Health Score']}
              />
              <Line 
                type="monotone" 
                dataKey="score" 
                stroke="#3b82f6" 
                strokeWidth={3}
                dot={{ fill: '#3b82f6', strokeWidth: 2, r: 4 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {/* Goal Completion Rate */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Target className="w-5 h-5 text-green-600" />
            Goal Completion Rate
          </CardTitle>
          <CardDescription>
            Progress on your health goals this {dateRange === '7d' ? 'week' : 'month'}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center">
            <ResponsiveContainer width="100%" height={200}>
              <PieChart>
                <Pie
                  data={goalCompletionData}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={80}
                  paddingAngle={5}
                  dataKey="value"
                >
                  {goalCompletionData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip formatter={(value) => `${value}%`} />
              </PieChart>
            </ResponsiveContainer>
          </div>
          
          <div className="grid grid-cols-3 gap-4 mt-4">
            {goalCompletionData.map((item, index) => (
              <div key={index} className="text-center">
                <div 
                  className="w-3 h-3 rounded-full mx-auto mb-1"
                  style={{ backgroundColor: item.color }}
                />
                <div className="text-sm font-medium">{item.value}%</div>
                <div className="text-xs text-gray-500">{item.name}</div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
