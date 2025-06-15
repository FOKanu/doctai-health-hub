
import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { TrendingUp } from 'lucide-react';

interface RiskProgressionData {
  date: string;
  low: number;
  medium: number;
  high: number;
}

interface RiskProgressionChartProps {
  data: RiskProgressionData[];
}

export const RiskProgressionChart: React.FC<RiskProgressionChartProps> = ({ data }) => {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <TrendingUp className="w-5 h-5 text-blue-600" />
          Risk Level Progression
        </CardTitle>
        <CardDescription>
          How your risk levels have changed over time
        </CardDescription>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              dataKey="date" 
              tickFormatter={(value) => new Date(value).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
            />
            <YAxis domain={[0, 100]} />
            <Tooltip 
              labelFormatter={(value) => new Date(value).toLocaleDateString()}
              formatter={(value, name) => {
                const nameStr = String(name);
                return [`${value}%`, nameStr.charAt(0).toUpperCase() + nameStr.slice(1) + ' Risk'];
              }}
            />
            <Line 
              type="monotone" 
              dataKey="low" 
              stroke="#10b981" 
              strokeWidth={2}
              name="low"
            />
            <Line 
              type="monotone" 
              dataKey="medium" 
              stroke="#f59e0b" 
              strokeWidth={2}
              name="medium"
            />
            <Line 
              type="monotone" 
              dataKey="high" 
              stroke="#ef4444" 
              strokeWidth={2}
              name="high"
            />
          </LineChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
};
