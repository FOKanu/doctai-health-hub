
import React, { useEffect, useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { TrendingUp, Loader2 } from 'lucide-react';
import { timeSeriesService, type RiskProgression } from '@/services/timeseriesService';

interface RiskProgressionChartProps {
  userId?: string;
  dateRange?: string;
}

export const RiskProgressionChart: React.FC<RiskProgressionChartProps> = ({
  userId = 'mock_user',
  dateRange = '90d'
}) => {
  const [riskData, setRiskData] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchRiskData = async () => {
      try {
        setLoading(true);
        setError(null);

        const startDate = new Date();
        const days = dateRange === '30d' ? 30 : dateRange === '90d' ? 90 : 180;
        startDate.setDate(startDate.getDate() - days);

        const data = await timeSeriesService.getRiskProgression({
          userId,
          startDate: startDate.toISOString(),
          endDate: new Date().toISOString()
        });

        // Process risk data into chart format
        const processedData = processRiskData(data);
        setRiskData(processedData);
      } catch (err) {
        console.error('Error fetching risk progression:', err);
        setError('Failed to load risk progression data');
      } finally {
        setLoading(false);
      }
    };

    fetchRiskData();
  }, [userId, dateRange]);

  const processRiskData = (data: RiskProgression[]): any[] => {
    // Group by date and calculate percentages
    const groupedData: { [key: string]: { low: number; medium: number; high: number; total: number } } = {};

    data.forEach(record => {
      const date = new Date(record.recorded_at).toISOString().split('T')[0];
      if (!groupedData[date]) {
        groupedData[date] = { low: 0, medium: 0, high: 0, total: 0 };
      }

      groupedData[date][record.risk_level as keyof typeof groupedData[typeof date]]++;
      groupedData[date].total++;
    });

    // Convert to percentages and sort by date
    return Object.entries(groupedData)
      .map(([date, counts]) => ({
        date,
        low: counts.total > 0 ? (counts.low / counts.total) * 100 : 0,
        medium: counts.total > 0 ? (counts.medium / counts.total) * 100 : 0,
        high: counts.total > 0 ? (counts.high / counts.total) * 100 : 0
      }))
      .sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime());
  };

  if (loading) {
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
          <div className="flex items-center justify-center h-48">
            <Loader2 className="w-6 h-6 animate-spin text-blue-600" />
            <span className="ml-2 text-gray-600">Loading risk data...</span>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (error) {
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
          <div className="flex items-center justify-center h-48">
            <div className="text-center">
              <div className="text-red-600 mb-2">⚠️</div>
              <div className="text-gray-600">{error}</div>
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }
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
          <LineChart data={riskData}>
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
