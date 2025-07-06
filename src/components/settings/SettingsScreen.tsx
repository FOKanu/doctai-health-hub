import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowLeft } from 'lucide-react';
import { SettingsSection } from './SettingsSection';
import { BackgroundSelector } from './BackgroundSelector';
import { createSettingsSections } from './settingsData';
import { NotificationSettings, PrivacySettings } from './types';
import { GoogleCloudConfigValidator } from '../GoogleCloudConfigValidator';
import { useSettingsModals } from './hooks/useSettingsModals';
import { ChangePasswordModal } from './modals/ChangePasswordModal';
import { TwoFactorModal } from './modals/TwoFactorModal';
import { PreferencesModal } from './modals/PreferencesModal';
import { DataStorageModal } from './modals/DataStorageModal';
import { SupportModal } from './modals/SupportModal';

const SettingsScreen = () => {
  const navigate = useNavigate();
  const { activeModal, openModal, closeModal } = useSettingsModals();
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

  const settingSections = createSettingsSections(notifications, setNotifications, privacy, setPrivacy, openModal);

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

        {/* Background Images Section */}
        <BackgroundSelector />

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

      {/* Modals */}
      <ChangePasswordModal 
        open={activeModal === 'changePassword'} 
        onOpenChange={(open) => !open && closeModal()} 
      />
      
      <TwoFactorModal 
        open={activeModal === 'twoFactor'} 
        onOpenChange={(open) => !open && closeModal()} 
      />
      
      <PreferencesModal 
        open={activeModal === 'language'} 
        onOpenChange={(open) => !open && closeModal()}
        type="language"
      />
      
      <PreferencesModal 
        open={activeModal === 'dateFormat'} 
        onOpenChange={(open) => !open && closeModal()}
        type="dateFormat"
      />
      
      <PreferencesModal 
        open={activeModal === 'timeZone'} 
        onOpenChange={(open) => !open && closeModal()}
        type="timeZone"
      />
      
      <PreferencesModal 
        open={activeModal === 'units'} 
        onOpenChange={(open) => !open && closeModal()}
        type="units"
      />
      
      <DataStorageModal 
        open={activeModal === 'syncSettings'} 
        onOpenChange={(open) => !open && closeModal()}
        type="sync"
      />
      
      <DataStorageModal 
        open={activeModal === 'exportData'} 
        onOpenChange={(open) => !open && closeModal()}
        type="export"
      />
      
      <DataStorageModal 
        open={activeModal === 'storageUsage'} 
        onOpenChange={(open) => !open && closeModal()}
        type="storage"
      />
      
      <DataStorageModal 
        open={activeModal === 'clearCache'} 
        onOpenChange={(open) => !open && closeModal()}
        type="cache"
      />
      
      <SupportModal 
        open={activeModal === 'helpCenter'} 
        onOpenChange={(open) => !open && closeModal()}
        type="help"
      />
      
      <SupportModal 
        open={activeModal === 'contactSupport'} 
        onOpenChange={(open) => !open && closeModal()}
        type="contact"
      />
      
      <SupportModal 
        open={activeModal === 'reportBug'} 
        onOpenChange={(open) => !open && closeModal()}
        type="bug"
      />
      
      <SupportModal 
        open={activeModal === 'privacyPolicy'} 
        onOpenChange={(open) => !open && closeModal()}
        type="privacy"
      />
      
      <SupportModal 
        open={activeModal === 'termsOfService'} 
        onOpenChange={(open) => !open && closeModal()}
        type="terms"
      />
    </div>
  );
};

export default SettingsScreen;
