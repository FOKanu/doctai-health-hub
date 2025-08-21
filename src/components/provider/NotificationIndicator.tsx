import React, { useState, useEffect } from 'react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Bell, Mail } from 'lucide-react';

interface NotificationIndicatorProps {
  className?: string;
}

// Mock notification management based on settings
const getNotificationSettings = () => {
  const stored = localStorage.getItem('providerSettings');
  if (stored) {
    const settings = JSON.parse(stored);
    return {
      emailNotifications: settings.emailNotifications || true,
      pushNotifications: settings.pushNotifications || true,
      patientMessages: settings.patientMessages || true,
      appointmentReminders: settings.appointmentReminders || true
    };
  }
  return {
    emailNotifications: true,
    pushNotifications: true,
    patientMessages: true,
    appointmentReminders: true
  };
};

export function NotificationIndicator({ className = '' }: NotificationIndicatorProps) {
  const [unreadCount, setUnreadCount] = useState(3);
  const [settings, setSettings] = useState(getNotificationSettings());

  useEffect(() => {
    // Listen for settings changes
    const handleSettingsChange = () => {
      const newSettings = getNotificationSettings();
      setSettings(newSettings);
      
      // Simulate unread count based on notification preferences
      let mockCount = 0;
      if (newSettings.emailNotifications) mockCount += 2;
      if (newSettings.patientMessages) mockCount += 1;
      if (newSettings.appointmentReminders) mockCount += 1;
      
      setUnreadCount(mockCount);
    };

    window.addEventListener('providerStatusChange', handleSettingsChange);
    
    const interval = setInterval(handleSettingsChange, 2000);

    return () => {
      window.removeEventListener('providerStatusChange', handleSettingsChange);
      clearInterval(interval);
    };
  }, []);

  // Don't show notifications if user has disabled all notification types
  const shouldShowNotifications = settings.emailNotifications || 
                                 settings.pushNotifications || 
                                 settings.patientMessages;

  if (!shouldShowNotifications) {
    return (
      <Button
        variant="ghost"
        size="sm"
        className={`p-2 hover:bg-blue-50 rounded-xl relative ${className}`}
        aria-label="Notifications disabled"
      >
        <Bell className="w-5 h-5 text-gray-400" />
      </Button>
    );
  }

  return (
    <Button
      variant="ghost"
      size="sm"
      className={`p-2 hover:bg-blue-50 rounded-xl relative ${className}`}
      aria-label={`${unreadCount} unread notifications`}
    >
      <Mail className="w-5 h-5 text-gray-600" />
      {unreadCount > 0 && (
        <Badge className="absolute -top-1 -right-1 h-5 w-5 text-xs bg-red-500 hover:bg-red-500 rounded-full p-0 flex items-center justify-center">
          {unreadCount > 9 ? '9+' : unreadCount}
        </Badge>
      )}
    </Button>
  );
}