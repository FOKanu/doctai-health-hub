
import React from 'react';
import { Routes, Route } from 'react-router-dom';
import { SidebarProvider } from '@/components/ui/sidebar';
import { AppSidebar } from '../components/AppSidebar';
import { AppHeader } from '../components/AppHeader';
import { MobileNavigation } from '../components/MobileNavigation';
import HomeScreen from '../components/HomeScreen';
import ScanScreen from '../components/ScanScreen';
import UploadScreen from '../components/UploadScreen';
import SpecialistScreen from '../components/SpecialistScreen';
import HistoryScreen from '../components/HistoryScreen';
import MedicationsScreen from '../components/MedicationsScreen';
import ProfileScreen from '../components/ProfileScreen';
import PostboxScreen from '../components/PostboxScreen';
import MedicalRecordsScreen from '../components/MedicalRecordsScreen';
import SettingsScreen from '../components/SettingsScreen';
import AppointmentsScreen from '../components/AppointmentsScreen';
import FitnessScreen from '../components/FitnessScreen';
import DietScreen from '../components/DietScreen';
import AIRecommendationsScreen from '../components/AIRecommendationsScreen';
import { useAnalytics } from '../hooks/useAnalytics';

const Index = () => {
  return (
    <SidebarProvider>
      <div className="min-h-screen flex w-full bg-gray-50">
        <AppSidebar />
        <div className="flex-1 flex flex-col">
          <AppHeader />
          <main className="flex-1 p-4 md:p-6 lg:p-8 pb-20 md:pb-6">
            <div className="max-w-7xl mx-auto">
              <Routes>
                <Route path="/" element={<HomeScreen />} />
                <Route path="/analytics" element={<AnalyticsWrapper />} />
                <Route path="/postbox" element={<PostboxScreen />} />
                <Route path="/medical-records" element={<MedicalRecordsScreen />} />
                <Route path="/scan" element={<ScanScreen />} />
                <Route path="/upload" element={<UploadScreen />} />
                <Route path="/specialists" element={<SpecialistScreen />} />
                <Route path="/history" element={<HistoryScreen />} />
                <Route path="/medications" element={<MedicationsScreen />} />
                <Route path="/appointments" element={<AppointmentsScreen />} />
                <Route path="/fitness" element={<FitnessScreen />} />
                <Route path="/diet" element={<DietScreen />} />
                <Route path="/recommendations" element={<AIRecommendationsScreen />} />
                <Route path="/profile" element={<ProfileScreen />} />
                <Route path="/settings" element={<SettingsScreen />} />
              </Routes>
            </div>
          </main>
        </div>
        <MobileNavigation />
      </div>
    </SidebarProvider>
  );
};

// Wrapper component for analytics to handle data loading
const AnalyticsWrapper = () => {
  const { analyticsData, loading, error } = useAnalytics();

  if (loading) {
    return <div className="flex items-center justify-center h-64">Loading analytics...</div>;
  }

  if (error) {
    return <div className="text-red-500 p-4">Error: {error}</div>;
  }

  if (!analyticsData) {
    return <div className="p-4">No analytics data available</div>;
  }

  // Create a simple analytics component since the original requires specific props
  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">Analytics Dashboard</h1>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <div className="bg-white p-6 rounded-lg shadow">
          <h3 className="text-lg font-semibold mb-2">Total Scans</h3>
          <p className="text-3xl font-bold text-blue-600">
            {analyticsData.comparisons.previousScans.length}
          </p>
        </div>
        <div className="bg-white p-6 rounded-lg shadow">
          <h3 className="text-lg font-semibold mb-2">Progress Status</h3>
          <p className="text-lg text-gray-600">
            {analyticsData.insights.progress.currentStatus}
          </p>
        </div>
        <div className="bg-white p-6 rounded-lg shadow">
          <h3 className="text-lg font-semibold mb-2">Improvement</h3>
          <p className="text-2xl font-bold text-green-600">
            {analyticsData.insights.progress.improvement.toFixed(1)}%
          </p>
        </div>
      </div>
    </div>
  );
};

export default Index;
