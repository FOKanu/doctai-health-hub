import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { useEffect } from "react";
import Index from "./pages/Index";
import NotFound from "./pages/NotFound";
import { apiServiceManager } from "./services/api/apiServiceManager";

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

const App = () => {
  useEffect(() => {
    initializeApiServices();
  }, []);

  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <Toaster />
        <Sonner />
        <BrowserRouter>
          <Routes>
            <Route path="/*" element={<Index />} />
            <Route path="*" element={<NotFound />} />
          </Routes>
        </BrowserRouter>
      </TooltipProvider>
    </QueryClientProvider>
  );
};

export default App;
