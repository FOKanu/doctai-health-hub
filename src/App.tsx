import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { useEffect } from "react";
import Index from "./pages/Index";
import NotFound from "./pages/NotFound";
import { apiServiceManager } from "./services/api/apiServiceManager";
import ComplianceDashboard from '@/components/compliance/ComplianceDashboard';
import { Shield } from 'lucide-react';
import { AuthProvider, useAuth } from './contexts/AuthContext';
import { ProviderLayout } from './components/layout/ProviderLayout';
import { EngineerLayout } from './components/layout/EngineerLayout';
import { ProviderRoute, EngineerRoute, PatientRoute } from './components/auth/RoleBasedRoute';
import { ProviderDashboard } from './components/provider/ProviderDashboard';
import { EngineerDashboard } from './components/engineer/EngineerDashboard';
import LoginScreen from './components/LoginScreen';
import WelcomeScreen from './components/WelcomeScreen';

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

// Root route component that handles authentication redirects
const RootRoute = () => {
  const { user, isAuthenticated, isLoading } = useAuth();

  // Show loading state while checking authentication
  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading...</p>
        </div>
      </div>
    );
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
                  <Index />
                </PatientRoute>
              } />

              {/* Provider Routes - Protected */}
              <Route path="/provider/*" element={
                <ProviderRoute>
                  <ProviderLayout>
                    <Routes>
                      <Route path="/dashboard" element={<ProviderDashboard />} />
                      <Route path="/patients" element={<div>Patient Management</div>} />
                      <Route path="/clinical" element={<div>Clinical Workflow</div>} />
                      <Route path="/ai-support" element={<div>AI Diagnostic Support</div>} />
                      <Route path="/compliance" element={<div>Compliance Center</div>} />
                      <Route path="/messages" element={<div>Messages/Chat</div>} />
                      <Route path="/settings" element={<div>Provider Settings</div>} />
                      <Route path="/cardiology" element={<div>Cardiology Tools</div>} />
                      <Route path="/neurology" element={<div>Neurology Tools</div>} />
                      <Route path="/ophthalmology" element={<div>Ophthalmology Tools</div>} />
                      <Route path="/orthopedics" element={<div>Orthopedics Tools</div>} />
                      <Route path="*" element={<ProviderDashboard />} />
                    </Routes>
                  </ProviderLayout>
                </ProviderRoute>
              } />

              {/* Engineer Routes - Protected */}
              <Route path="/engineer/*" element={
                <EngineerRoute>
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
                </EngineerRoute>
              } />

              {/* Admin Routes - Protected */}
              <Route path="/admin/*" element={
                <ProviderRoute>
                  <div>Admin Dashboard</div>
                </ProviderRoute>
              } />

              {/* Compliance Route */}
              <Route path="/compliance" element={<ComplianceDashboard />} />

              {/* Fallback */}
              <Route path="*" element={<NotFound />} />
            </Routes>
          </BrowserRouter>
        </TooltipProvider>
      </AuthProvider>
    </QueryClientProvider>
  );
};

export default App;
