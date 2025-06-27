import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowLeft } from 'lucide-react';
import { SettingsSection } from './SettingsSection';
import { createSettingsSections } from './settingsData';
import { NotificationSettings, PrivacySettings } from './types';
import { GoogleCloudConfigValidator } from '../GoogleCloudConfigValidator';

const SettingsScreen = () => {
  const navigate = useNavigate();
  const [notifications, setNotifications] = useState<NotificationSettings>({
    appointments: true,
    medications: true,
    labResults: true,
    reminders: false
  });
  const [privacy, setPrivacy] = useState<PrivacySettings>({
    dataSharing: false,
    analytics: true,
    biometrics: true
  });

  const settingSections = createSettingsSections(notifications, setNotifications, privacy, setPrivacy);

  return (
    <div className="min-h-screen bg-gray-50 pb-20">
      {/* Header */}
      <div className="bg-white shadow-sm">
        <div className="flex items-center p-4">
          <button
            onClick={() => navigate('/')}
            className="p-2 -ml-2 rounded-full hover:bg-gray-100"
          >
            <ArrowLeft className="w-6 h-6" />
          </button>
          <h1 className="text-xl font-semibold ml-2">Settings</h1>
        </div>
      </div>

      <div className="p-4 space-y-6">
        {settingSections.map((section, sectionIndex) => (
          <SettingsSection key={sectionIndex} section={section} />
        ))}

        {/* Google Cloud Configuration Validator */}
        <div className="bg-white rounded-lg shadow-sm p-4">
          <h2 className="text-lg font-semibold mb-4">Google Cloud Configuration</h2>
          <GoogleCloudConfigValidator />
        </div>

        {/* App Info */}
        <div className="text-center text-gray-500 text-sm mt-8">
          <p>DoctAI Health Assistant</p>
          <p>Made with ❤️ for your health</p>
        </div>
      </div>
    </div>
  );
};

export default SettingsScreen;
