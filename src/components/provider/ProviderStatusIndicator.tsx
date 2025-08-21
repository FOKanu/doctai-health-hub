import React, { useState, useEffect } from 'react';
import { Badge } from '@/components/ui/badge';
import { Globe } from 'lucide-react';

interface ProviderStatusIndicatorProps {
  className?: string;
}

// Mock provider status management
const getProviderStatus = () => {
  const stored = localStorage.getItem('providerSettings');
  if (stored) {
    const settings = JSON.parse(stored);
    return settings.isOnline || false;
  }
  return true; // Default to online
};

export function ProviderStatusIndicator({ className = '' }: ProviderStatusIndicatorProps) {
  const [isOnline, setIsOnline] = useState(getProviderStatus());

  useEffect(() => {
    // Listen for provider status changes from settings
    const handleStatusChange = (event: CustomEvent) => {
      setIsOnline(event.detail.isOnline);
    };

    window.addEventListener('providerStatusChange', handleStatusChange as EventListener);

    // Also check localStorage periodically in case it's updated elsewhere
    const interval = setInterval(() => {
      const currentStatus = getProviderStatus();
      setIsOnline(currentStatus);
    }, 1000);

    return () => {
      window.removeEventListener('providerStatusChange', handleStatusChange as EventListener);
      clearInterval(interval);
    };
  }, []);

  return (
    <div className={`flex items-center space-x-2 px-3 py-1 rounded-full border transition-all duration-200 ${
      isOnline 
        ? 'bg-green-50 border-green-200' 
        : 'bg-gray-50 border-gray-200'
    } ${className}`}>
      <div className={`w-2 h-2 rounded-full ${
        isOnline ? 'bg-green-500 animate-pulse' : 'bg-gray-400'
      }`}></div>
      <Globe className={`w-3 h-3 ${
        isOnline ? 'text-green-600' : 'text-gray-400'
      }`} />
      <span className={`text-sm font-medium ${
        isOnline ? 'text-green-700' : 'text-gray-600'
      }`}>
        {isOnline ? 'Online' : 'Offline'}
      </span>
    </div>
  );
}