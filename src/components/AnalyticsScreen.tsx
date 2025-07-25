
import React, { useState, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
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
import { SequenceAnalyzer } from './analytics/SequenceAnalyzer';
import { HealthScoreCard } from './analytics/HealthScoreCard';
import { TelemedicineConsultation } from './telemedicine/TelemedicineConsultation';

const AnalyticsScreen = () => {
  const [searchParams, setSearchParams] = useSearchParams();
  const [dateRange, setDateRange] = useState('30d');
  const [selectedMetric, setSelectedMetric] = useState('all');
  const [activeTab, setActiveTab] = useState('overview');

  // Handle URL parameters for tab navigation
  useEffect(() => {
    const tabParam = searchParams.get('tab');
    if (tabParam && ['overview', 'metrics', 'risks', 'imaging', 'progression', 'appointments', 'compliance'].includes(tabParam)) {
      setActiveTab(tabParam);
    }
  }, [searchParams]);

  const handleTabChange = (value: string) => {
    setActiveTab(value);
    setSearchParams({ tab: value });
  };

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
      <Tabs value={activeTab} onValueChange={handleTabChange} className="space-y-6">
        <div className="relative group">
          {/* Left Scroll Arrow */}
          <button
            onClick={() => {
              const container = document.querySelector('.tabs-scroll-container');
              if (container) {
                container.scrollBy({ left: -200, behavior: 'smooth' });
              }
            }}
            className="absolute left-0 top-1/2 -translate-y-1/2 z-10 bg-white/80 backdrop-blur-sm border border-gray-200 rounded-full p-1 shadow-md opacity-0 group-hover:opacity-100 transition-opacity duration-200 hover:bg-white"
          >
            <svg className="w-4 h-4 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
            </svg>
          </button>

          {/* Right Scroll Arrow */}
          <button
            onClick={() => {
              const container = document.querySelector('.tabs-scroll-container');
              if (container) {
                container.scrollBy({ left: 200, behavior: 'smooth' });
              }
            }}
            className="absolute right-0 top-1/2 -translate-y-1/2 z-10 bg-white/80 backdrop-blur-sm border border-gray-200 rounded-full p-1 shadow-md opacity-0 group-hover:opacity-100 transition-opacity duration-200 hover:bg-white"
          >
            <svg className="w-4 h-4 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
            </svg>
          </button>

          <div className="overflow-x-auto scrollbar-hide tabs-scroll-container">
            <TabsList className="flex w-max min-w-full space-x-1 px-4">
              <TabsTrigger value="overview" className="whitespace-nowrap">Overview</TabsTrigger>
              <TabsTrigger value="metrics" className="whitespace-nowrap">Metrics</TabsTrigger>
              <TabsTrigger value="risks" className="whitespace-nowrap">Risk Analysis</TabsTrigger>
              <TabsTrigger value="imaging" className="whitespace-nowrap">Imaging</TabsTrigger>
              <TabsTrigger value="progression" className="whitespace-nowrap">Progression</TabsTrigger>
              <TabsTrigger value="health-score" className="whitespace-nowrap">Health Score</TabsTrigger>
              <TabsTrigger value="telemedicine" className="whitespace-nowrap">Telemedicine</TabsTrigger>
              <TabsTrigger value="appointments" className="whitespace-nowrap">Care</TabsTrigger>
              <TabsTrigger value="compliance" className="whitespace-nowrap">Compliance</TabsTrigger>
            </TabsList>
          </div>
        </div>

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

        <TabsContent value="progression" className="space-y-6">
          <SequenceAnalyzer userId="mock_user" />
        </TabsContent>

        <TabsContent value="health-score" className="space-y-6">
          <div className="grid gap-6">
            <HealthScoreCard userId="mock_user" />
          </div>
        </TabsContent>

        <TabsContent value="telemedicine" className="space-y-6">
          <div className="grid gap-6">
            <TelemedicineConsultation userId="mock_user" />
          </div>
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
