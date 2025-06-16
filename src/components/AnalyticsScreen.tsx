
import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Button } from '@/components/ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Download, Filter, TrendingUp, TrendingDown, Minus } from 'lucide-react';
import { HealthOverview } from './analytics/HealthOverview';
import { MetricsDashboard } from './analytics/MetricsDashboard';
import { RiskPredictions } from './analytics/RiskPredictions';
import { ImagingStats } from './analytics/ImagingStats';
import { AppointmentsMedications } from './analytics/AppointmentsMedications';
import { RescanCompliance } from './analytics/RescanCompliance';

const AnalyticsScreen = () => {
  const [dateRange, setDateRange] = useState('30d');
  const [selectedMetric, setSelectedMetric] = useState('all');

  const handleExport = (format: 'pdf' | 'csv') => {
    console.log(`Exporting analytics data as ${format}`);
    // Implementation for export functionality
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Analytics Dashboard</h1>
          <p className="text-gray-600">AI-powered health insights and visualizations</p>
        </div>
        
        {/* Filters and Export */}
        <div className="flex flex-wrap items-center gap-3">
          <Select value={dateRange} onValueChange={setDateRange}>
            <SelectTrigger className="w-32">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="7d">Last 7 days</SelectItem>
              <SelectItem value="30d">Last 30 days</SelectItem>
              <SelectItem value="90d">Last 3 months</SelectItem>
              <SelectItem value="custom">Custom range</SelectItem>
            </SelectContent>
          </Select>

          <Select value={selectedMetric} onValueChange={setSelectedMetric}>
            <SelectTrigger className="w-40">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Metrics</SelectItem>
              <SelectItem value="steps">Steps</SelectItem>
              <SelectItem value="heart-rate">Heart Rate</SelectItem>
              <SelectItem value="sleep">Sleep</SelectItem>
              <SelectItem value="weight">Weight</SelectItem>
              <SelectItem value="temperature">Temperature</SelectItem>
            </SelectContent>
          </Select>

          <Button variant="outline" size="sm" onClick={() => handleExport('pdf')}>
            <Download className="w-4 h-4 mr-2" />
            Export PDF
          </Button>
          
          <Button variant="outline" size="sm" onClick={() => handleExport('csv')}>
            <Download className="w-4 h-4 mr-2" />
            Export CSV
          </Button>
        </div>
      </div>

      {/* Main Analytics Content */}
      <Tabs defaultValue="overview" className="space-y-6">
        <TabsList className="grid w-full grid-cols-6">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="metrics">Metrics</TabsTrigger>
          <TabsTrigger value="risks">Risk Analysis</TabsTrigger>
          <TabsTrigger value="imaging">Imaging</TabsTrigger>
          <TabsTrigger value="appointments">Care</TabsTrigger>
          <TabsTrigger value="compliance">Compliance</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-6">
          <HealthOverview dateRange={dateRange} />
        </TabsContent>

        <TabsContent value="metrics" className="space-y-6">
          <MetricsDashboard dateRange={dateRange} selectedMetric={selectedMetric} />
        </TabsContent>

        <TabsContent value="risks" className="space-y-6">
          <RiskPredictions dateRange={dateRange} />
        </TabsContent>

        <TabsContent value="imaging" className="space-y-6">
          <ImagingStats dateRange={dateRange} />
        </TabsContent>

        <TabsContent value="appointments" className="space-y-6">
          <AppointmentsMedications dateRange={dateRange} />
        </TabsContent>

        <TabsContent value="compliance" className="space-y-6">
          <RescanCompliance dateRange={dateRange} />
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default AnalyticsScreen;
