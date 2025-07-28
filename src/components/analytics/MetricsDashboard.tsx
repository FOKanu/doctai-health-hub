
import React, { useEffect, useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Footprints, Heart, Moon, Thermometer, Droplets, Weight, TrendingUp, TrendingDown, Loader2 } from 'lucide-react';
import { timeSeriesService, type HealthMetricsData } from '@/services/timeseriesService';

interface MetricsDashboardProps {
  dateRange: string;
  selectedMetric: string;
  userId?: string;
}

export const MetricsDashboard: React.FC<MetricsDashboardProps> = ({
  dateRange,
  selectedMetric,
  userId = 'mock_user'
}) => {
  const [metricsData, setMetricsData] = useState<HealthMetricsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        setLoading(true);
        setError(null);

        const startDate = new Date();
        const days = dateRange === '7d' ? 7 : dateRange === '30d' ? 30 : 90;
        startDate.setDate(startDate.getDate() - days);

        const data = await timeSeriesService.getHealthMetrics({
          userId,
          startDate: startDate.toISOString(),
          endDate: new Date().toISOString()
        });

        setMetricsData(data);
      } catch (err) {
        console.error('Error fetching metrics:', err);
        setError('Failed to load health metrics');
      } finally {
        setLoading(false);
      }
    };

    fetchMetrics();
  }, [dateRange, userId]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="w-8 h-8 animate-spin text-blue-600" />
        <span className="ml-2 text-gray-600">Loading health metrics...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <div className="text-red-600 mb-2">⚠️</div>
          <div className="text-gray-600">{error}</div>
        </div>
      </div>
    );
  }

  if (!metricsData) {
    return null;
  }

  // Transform data for charts
  const stepsData = metricsData.steps.map(item => ({
    date: new Date(item.timestamp).toISOString().split('T')[0],
    steps: item.value
  }));

  const heartRateData = metricsData.heartRate.map(item => ({
    date: new Date(item.timestamp).toISOString().split('T')[0],
    resting: item.value,
    active: item.value + Math.random() * 20 + 50 // Mock active HR
  }));

  const sleepData = metricsData.sleepHours.map(item => ({
    date: new Date(item.timestamp).toISOString().split('T')[0],
    hours: item.value
  }));

  const temperatureData = metricsData.temperature.map(item => ({
    date: new Date(item.timestamp).toISOString().split('T')[0],
    temp: item.value
  }));

  const waterData = metricsData.waterIntake.map(item => ({
    date: new Date(item.timestamp).toISOString().split('T')[0],
    glasses: item.value
  }));

  const weightData = Array.from({ length: 30 }, (_, i) => ({
    date: new Date(Date.now() - (29 - i) * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
    weight: 165 + Math.sin(i / 5) * 3 + Math.random() * 1,
    goal: 160,
  }));

  const MetricCard = ({
    title,
    icon: Icon,
    value,
    change,
    unit,
    status,
    children
  }: {
    title: string;
    icon: React.ComponentType<{ className?: string }>;
    value: string | number;
    change: number;
    unit: string;
    status: 'normal' | 'elevated' | 'risk';
    children: React.ReactNode;
  }) => {
    const statusColors = {
      normal: 'text-green-600 bg-green-50',
      elevated: 'text-orange-600 bg-orange-50',
      risk: 'text-red-600 bg-red-50',
    };

    return (
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Icon className="w-5 h-5" />
              {title}
            </div>
            <div className={`px-2 py-1 rounded-full text-xs font-medium ${statusColors[status]}`}>
              {status.charAt(0).toUpperCase() + status.slice(1)}
            </div>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <span className="text-2xl font-bold">{value}</span>
              <span className="text-sm text-gray-500">{unit}</span>
            </div>
            <div className="flex items-center gap-1">
              {change > 0 ? (
                <TrendingUp className="w-4 h-4 text-green-600" />
              ) : change < 0 ? (
                <TrendingDown className="w-4 h-4 text-red-600" />
              ) : null}
              <span className={`text-sm font-medium ${
                change > 0 ? 'text-green-600' : change < 0 ? 'text-red-600' : 'text-gray-600'
              }`}>
                {change > 0 ? '+' : ''}{change.toFixed(1)}%
              </span>
            </div>
          </div>
          {children}
        </CardContent>
      </Card>
    );
  };

  if (selectedMetric !== 'all' && selectedMetric !== 'steps') {
    // Filter logic for specific metrics would go here
  }

  return (
    <div className="space-y-6">
      {/* Steps */}
      {(selectedMetric === 'all' || selectedMetric === 'steps') && (
        <MetricCard
          title="Daily Steps"
          icon={Footprints}
          value="9,847"
          change={12.5}
          unit="steps"
          status="normal"
        >
          <ResponsiveContainer width="100%" height={150}>
            <BarChart data={stepsData.slice(-7)}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                dataKey="date"
                tickFormatter={(value) => new Date(value).toLocaleDateString('en-US', { weekday: 'short' })}
              />
              <YAxis />
              <Tooltip
                labelFormatter={(value) => new Date(value).toLocaleDateString()}
                formatter={(value) => [`${value}`, 'Steps']}
              />
              <Bar dataKey="steps" fill="#3b82f6" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
          <div className="mt-2 text-sm text-gray-600">
            Weekly average: 9,234 steps • +15% vs last month
          </div>
        </MetricCard>
      )}

      {/* Heart Rate */}
      {(selectedMetric === 'all' || selectedMetric === 'heart-rate') && (
        <MetricCard
          title="Heart Rate"
          icon={Heart}
          value="72"
          change={-2.1}
          unit="bpm"
          status="normal"
        >
          <ResponsiveContainer width="100%" height={150}>
            <LineChart data={heartRateData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                dataKey="date"
                tickFormatter={(value) => new Date(value).toLocaleDateString('en-US', { weekday: 'short' })}
              />
              <YAxis domain={[50, 150]} />
              <Tooltip
                labelFormatter={(value) => new Date(value).toLocaleDateString()}
              />
              <Line
                type="monotone"
                dataKey="resting"
                stroke="#ef4444"
                strokeWidth={2}
                name="Resting HR"
              />
              <Line
                type="monotone"
                dataKey="active"
                stroke="#f59e0b"
                strokeWidth={2}
                name="Active HR"
              />
            </LineChart>
          </ResponsiveContainer>
          <div className="mt-2 text-sm text-gray-600">
            Min: 68 bpm • Max: 145 bpm • Avg: 89 bpm
          </div>
        </MetricCard>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Sleep */}
        {(selectedMetric === 'all' || selectedMetric === 'sleep') && (
          <MetricCard
            title="Sleep Hours"
            icon={Moon}
            value="7.5"
            change={8.2}
            unit="hours"
            status="normal"
          >
            <ResponsiveContainer width="100%" height={120}>
              <LineChart data={sleepData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis
                  dataKey="date"
                  tickFormatter={(value) => new Date(value).toLocaleDateString('en-US', { weekday: 'short' })}
                />
                <YAxis domain={[6, 10]} />
                <Tooltip
                  formatter={(value) => [`${Number(value).toFixed(1)}h`, 'Sleep']}
                />
                <Line
                  type="monotone"
                  dataKey="hours"
                  stroke="#8b5cf6"
                  strokeWidth={2}
                />
              </LineChart>
            </ResponsiveContainer>
            <div className="mt-2 text-sm text-gray-600">
              Sleep efficiency: 92%
            </div>
          </MetricCard>
        )}

        {/* Temperature */}
        {(selectedMetric === 'all' || selectedMetric === 'temperature') && (
          <MetricCard
            title="Body Temperature"
            icon={Thermometer}
            value="98.6"
            change={0.5}
            unit="°F"
            status="normal"
          >
            <ResponsiveContainer width="100%" height={120}>
              <LineChart data={temperatureData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis
                  dataKey="date"
                  tickFormatter={(value) => new Date(value).toLocaleDateString('en-US', { weekday: 'short' })}
                />
                <YAxis domain={[97, 100]} />
                <Tooltip
                  formatter={(value) => [`${Number(value).toFixed(1)}°F`, 'Temperature']}
                />
                <Line
                  type="monotone"
                  dataKey="temp"
                  stroke="#f59e0b"
                  strokeWidth={2}
                />
              </LineChart>
            </ResponsiveContainer>
            <div className="mt-2 text-sm text-gray-600">
              Normal range: 97.8-99.1°F
            </div>
          </MetricCard>
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Water Intake */}
        <MetricCard
          title="Water Intake"
          icon={Droplets}
          value="8"
          change={12.5}
          unit="glasses"
          status="normal"
        >
          <ResponsiveContainer width="100%" height={120}>
            <BarChart data={waterData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                dataKey="date"
                tickFormatter={(value) => new Date(value).toLocaleDateString('en-US', { weekday: 'short' })}
              />
              <YAxis domain={[0, 12]} />
              <Tooltip
                formatter={(value) => [`${value}`, 'Glasses']}
              />
              <Bar dataKey="glasses" fill="#06b6d4" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
          <div className="mt-2 text-sm text-gray-600">
            Goal: 8 glasses/day ✓
          </div>
        </MetricCard>

        {/* Weight Progress */}
        {(selectedMetric === 'all' || selectedMetric === 'weight') && (
          <MetricCard
            title="Weight Progress"
            icon={Weight}
            value="165.2"
            change={-1.8}
            unit="lbs"
            status="normal"
          >
            <ResponsiveContainer width="100%" height={120}>
              <LineChart data={weightData.slice(-7)}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis
                  dataKey="date"
                  tickFormatter={(value) => new Date(value).toLocaleDateString('en-US', { weekday: 'short' })}
                />
                <YAxis domain={[160, 170]} />
                <Tooltip
                  formatter={(value, name) => [
                    `${Number(value).toFixed(1)} lbs`,
                    name === 'weight' ? 'Current' : 'Goal'
                  ]}
                />
                <Line
                  type="monotone"
                  dataKey="weight"
                  stroke="#10b981"
                  strokeWidth={2}
                  name="weight"
                />
                <Line
                  type="monotone"
                  dataKey="goal"
                  stroke="#6b7280"
                  strokeWidth={1}
                  strokeDasharray="5 5"
                  name="goal"
                />
              </LineChart>
            </ResponsiveContainer>
            <div className="mt-2 text-sm text-gray-600">
              BMI: 24.1 • Goal: 160 lbs
            </div>
          </MetricCard>
        )}
      </div>
    </div>
  );
};
