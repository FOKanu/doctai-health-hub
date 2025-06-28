
import React from 'react';
import { Routes, Route } from 'react-router-dom';
import { SidebarProvider } from '@/components/ui/sidebar';
import { AppSidebar } from '../components/AppSidebar';
import { AppHeader } from '../components/AppHeader';
import { MobileNavigation } from '../components/MobileNavigation';
import ErrorBoundary from '../components/ErrorBoundary';
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
import ResultsScreen from '../components/ResultsScreen';
import DietPlanScreen from '../components/DietPlanScreen';
import TotalScansScreen from '../components/TotalScansScreen';
import RiskAssessmentsScreen from '../components/RiskAssessmentsScreen';
import LoginScreen from '../components/LoginScreen';
import WelcomeScreen from '../components/WelcomeScreen';

const Index = () => {
  return (
    <Routes>
      {/* Authentication Routes */}
      <Route path="/login" element={<LoginScreen />} />
      <Route path="/welcome" element={<WelcomeScreen />} />
      
      {/* Main App Routes */}
      <Route path="/*" element={
        <SidebarProvider>
          <div className="min-h-screen flex w-full bg-gray-50">
            <AppSidebar />
            <div className="flex-1 flex flex-col">
              <AppHeader />
              <main className="flex-1 p-4 md:p-6 lg:p-8 pb-20 md:pb-6">
                <div className="max-w-7xl mx-auto">
                  <Routes>
                    <Route path="/" element={<HomeScreen />} />
                    <Route path="/results" element={<ResultsScreen />} />
                    <Route path="/analytics" element={<AnalyticsScreen />} />
                    <Route path="/postbox" element={<PostboxScreen />} />
                    <Route path="/medical-records" element={<MedicalRecordsScreen />} />
                    <Route path="/total-scans" element={<TotalScansScreen />} />
                    <Route path="/risk-assessments" element={<RiskAssessmentsScreen />} />
                    <Route
                      path="/scan"
                      element={
                        <ErrorBoundary
                          onError={(error, errorInfo) => {
                            console.error('ScanScreen Error:', error, errorInfo);
                            // You could send this to an error reporting service
                          }}
                        >
                          <ScanScreen />
                        </ErrorBoundary>
                      }
                    />
                    <Route path="/upload" element={<UploadScreen />} />
                    <Route path="/specialists" element={<SpecialistScreen />} />
                    <Route path="/history" element={<HistoryScreen />} />
                    <Route path="/medications" element={<MedicationsScreen />} />
                    <Route path="/appointments" element={<AppointmentsScreen />} />
                    <Route path="/fitness" element={<FitnessScreen />} />
                    <Route path="/diet" element={<DietPlanScreen />} />
                    <Route path="/profile" element={<ProfileScreen />} />
                    <Route path="/settings" element={<SettingsScreen />} />
                  </Routes>
                </div>
              </main>
            </div>
            <MobileNavigation />
          </div>
        </SidebarProvider>
      } />
    </Routes>
  );
};

export default Index;
