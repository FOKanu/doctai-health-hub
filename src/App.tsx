import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { useEffect, Suspense, lazy } from "react";
import { apiServiceManager } from "./services/api/apiServiceManager";
import { Shield } from 'lucide-react';
import { AuthProvider, useAuth } from './contexts/AuthContext';
import { ProviderRoute, EngineerRoute, PatientRoute, AdminRoute } from './components/auth/RoleBasedRoute';
import LoginScreen from './components/LoginScreen';
import WelcomeScreen from './components/WelcomeScreen';

// Lazy load heavy components for better code splitting
const Index = lazy(() => import("./pages/Index"));
const NotFound = lazy(() => import("./pages/NotFound"));
const ComplianceDashboard = lazy(() => import('@/components/compliance/ComplianceDashboard'));

// Layout components
const ProviderLayout = lazy(() => import('./components/layout/ProviderLayout'));
const EngineerLayout = lazy(() => import('./components/layout/EngineerLayout'));
const AdminLayout = lazy(() => import('./components/layout/AdminLayout'));

// Provider components
const ProviderDashboard = lazy(() => import('./components/provider/ProviderDashboard'));
const PatientManagement = lazy(() => import('./components/provider/pages/PatientManagement'));
const PatientDetail = lazy(() => import('./components/provider/pages/PatientDetail'));
const Schedule = lazy(() => import('./components/provider/pages/Schedule'));
const Messages = lazy(() => import('./components/provider/pages/Messages'));
const ClinicalWorkflow = lazy(() => import('./components/provider/pages/ClinicalWorkflow'));
const AIDiagnosticSupport = lazy(() => import('./components/provider/pages/AIDiagnosticSupport'));
const ComplianceCenter = lazy(() => import('./components/provider/pages/ComplianceCenter'));
const ProviderSettings = lazy(() => import('./components/provider/pages/ProviderSettings'));

// Specialty components
const Cardiology = lazy(() => import('./components/provider/pages/specialties/Cardiology'));
const Neurology = lazy(() => import('./components/provider/pages/specialties/Neurology'));
const Ophthalmology = lazy(() => import('./components/provider/pages/specialties/Ophthalmology'));
const Orthopedics = lazy(() => import('./components/provider/pages/specialties/Orthopedics'));

// Admin and Engineer components
const AdminDashboard = lazy(() => import('./components/admin/AdminDashboard'));
const EngineerDashboard = lazy(() => import('./components/engineer/EngineerDashboard'));

const queryClient = new QueryClient();

// Initialize API services
const initializeApiServices = () => {
  const config = {
    openai: {
      apiKey: import.meta.env.VITE_OPENAI_API_KEY,
      model: import.meta.env.VITE_OPENAI_MODEL || 'gpt-4',
      maxTokens: 1000,
      temperature: 0.7
    },
    notifications: {
      twilio: {
        accountSid: import.meta.env.VITE_TWILIO_ACCOUNT_SID,
        authToken: import.meta.env.VITE_TWILIO_AUTH_TOKEN,
        phoneNumber: import.meta.env.VITE_TWILIO_PHONE_NUMBER
      },
      sendGrid: {
        apiKey: import.meta.env.VITE_SENDGRID_API_KEY,
        fromEmail: import.meta.env.VITE_SENDGRID_FROM_EMAIL
      }
    }
  };

  apiServiceManager.initialize(config);
  console.log('🚀 API Services initialized');
};

// Loading component for Suspense fallback
const LoadingSpinner = () => (
  <div className="min-h-screen flex items-center justify-center bg-gray-50">
    <div className="text-center">
      <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
      <p className="mt-4 text-gray-600">Loading...</p>
    </div>
  </div>
);

// Root route component that handles authentication redirects
const RootRoute = () => {
  const { user, isAuthenticated, isLoading } = useAuth();

  // Show loading state while checking authentication
  if (isLoading) {
    return <LoadingSpinner />;
  }

  // If not authenticated, redirect to login
  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }

  // If authenticated, redirect to appropriate dashboard based on role
  if (user) {
    switch (user.role) {
      case 'patient':
        return <Navigate to="/patient/" replace />;
      case 'provider':
        return <Navigate to="/provider/dashboard" replace />;
      case 'engineer':
        return <Navigate to="/engineer/dashboard" replace />;
      case 'admin':
        return <Navigate to="/admin/dashboard" replace />;
      default:
        return <Navigate to="/patient/" replace />;
    }
  }

  // Fallback to login
  return <Navigate to="/login" replace />;
};

const App = () => {
  useEffect(() => {
    initializeApiServices();
    console.log('🔍 App component mounted');
  }, []);

  return (
    <QueryClientProvider client={queryClient}>
      <AuthProvider>
        <TooltipProvider>
          <Toaster />
          <Sonner />
          <BrowserRouter>
            <Routes>
              {/* Public Routes - No authentication required */}
              <Route path="/login" element={<LoginScreen />} />
              <Route path="/welcome" element={<WelcomeScreen />} />

              {/* Root route - handles authentication redirects */}
              <Route path="/" element={<RootRoute />} />

              {/* Patient/Client Routes - Protected */}
              <Route path="/patient/*" element={
                <PatientRoute>
                  <Suspense fallback={<LoadingSpinner />}>
                    <Index />
                  </Suspense>
                </PatientRoute>
              } />

              {/* Provider Routes - Protected */}
              <Route path="/provider/*" element={
                <ProviderRoute>
                  <Suspense fallback={<LoadingSpinner />}>
                    <ProviderLayout>
                      <Routes>
                        <Route path="/dashboard" element={<ProviderDashboard />} />
                        <Route path="/patients" element={<PatientManagement />} />
                        <Route path="/patients/:id" element={<PatientDetail />} />
                        <Route path="/schedule" element={<Schedule />} />
                        <Route path="/messages" element={<Messages />} />
                        <Route path="/clinical" element={<ClinicalWorkflow />} />
                        <Route path="/ai-support" element={<AIDiagnosticSupport />} />
                        <Route path="/compliance" element={<ComplianceCenter />} />
                        <Route path="/specialty/cardiology" element={<Cardiology />} />
                        <Route path="/specialty/neurology" element={<Neurology />} />
                        <Route path="/specialty/ophthalmology" element={<Ophthalmology />} />
                        <Route path="/specialty/orthopedics" element={<Orthopedics />} />
                        <Route path="/settings" element={<ProviderSettings />} />
                        <Route path="*" element={<ProviderDashboard />} />
                      </Routes>
                    </ProviderLayout>
                  </Suspense>
                </ProviderRoute>
              } />

              {/* Engineer Routes - Protected */}
              <Route path="/engineer/*" element={
                <EngineerRoute>
                  <Suspense fallback={<LoadingSpinner />}>
                    <EngineerLayout>
                      <Routes>
                        <Route path="/dashboard" element={<EngineerDashboard />} />
                        <Route path="/dev-tools" element={<div>Development Tools</div>} />
                        <Route path="/data" element={<div>Data Management</div>} />
                        <Route path="/security" element={<div>Security & Compliance</div>} />
                        <Route path="/logs" element={<div>Logs & Alerts</div>} />
                        <Route path="/settings" element={<div>Engineer Settings</div>} />
                        <Route path="/infrastructure" element={<div>Infrastructure</div>} />
                        <Route path="/performance" element={<div>Performance</div>} />
                        <Route path="/storage" element={<div>Storage</div>} />
                        <Route path="/network" element={<div>Network</div>} />
                        <Route path="*" element={<EngineerDashboard />} />
                      </Routes>
                    </EngineerLayout>
                  </Suspense>
                </EngineerRoute>
              } />

              {/* Admin Routes - Protected */}
              <Route path="/admin/*" element={
                <AdminRoute>
                  <Suspense fallback={<LoadingSpinner />}>
                    <AdminLayout>
                      <Routes>
                        <Route path="/dashboard" element={<AdminDashboard />} />
                        <Route path="/users" element={<div>User Management</div>} />
                        <Route path="/security" element={<div>Security Settings</div>} />
                        <Route path="/system" element={<div>System Health</div>} />
                        <Route path="/data" element={<div>Data Management</div>} />
                        <Route path="/analytics" element={<div>Analytics</div>} />
                        <Route path="/settings" element={<div>Admin Settings</div>} />
                        <Route path="*" element={<AdminDashboard />} />
                      </Routes>
                    </AdminLayout>
                  </Suspense>
                </AdminRoute>
              } />

              {/* Compliance Route */}
              <Route path="/compliance" element={
                <Suspense fallback={<LoadingSpinner />}>
                  <ComplianceDashboard />
                </Suspense>
              } />

              {/* Fallback */}
              <Route path="*" element={
                <Suspense fallback={<LoadingSpinner />}>
                  <NotFound />
                </Suspense>
              } />
            </Routes>
          </BrowserRouter>
        </TooltipProvider>
      </AuthProvider>
    </QueryClientProvider>
  );
};

export default App;
