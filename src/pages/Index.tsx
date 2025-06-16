
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
import AnalyticsScreen from '../components/AnalyticsScreen';
import PostboxScreen from '../components/PostboxScreen';
import MedicalRecordsScreen from '../components/MedicalRecordsScreen';
import SettingsScreen from '../components/SettingsScreen';
import AppointmentsScreen from '../components/AppointmentsScreen';
import FitnessScreen from '../components/FitnessScreen';

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
                <Route path="/analytics" element={<AnalyticsScreen />} />
                <Route path="/postbox" element={<PostboxScreen />} />
                <Route path="/medical-records" element={<MedicalRecordsScreen />} />
                <Route path="/scan" element={<ScanScreen />} />
                <Route path="/upload" element={<UploadScreen />} />
                <Route path="/specialists" element={<SpecialistScreen />} />
                <Route path="/history" element={<HistoryScreen />} />
                <Route path="/medications" element={<MedicationsScreen />} />
                <Route path="/appointments" element={<AppointmentsScreen />} />
                <Route path="/fitness" element={<FitnessScreen />} />
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

export default Index;
