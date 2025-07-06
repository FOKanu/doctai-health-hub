import { Shield } from 'lucide-react';
import { SettingSection, PrivacySettings } from '../types';
import { ModalType } from '../hooks/useSettingsModals';

export const createPrivacySection = (
  privacy: PrivacySettings,
  setPrivacy: React.Dispatch<React.SetStateAction<PrivacySettings>>,
  openModal: (modal: ModalType) => void
): SettingSection => ({
  title: 'Privacy & Security',
  icon: Shield,
  items: [
    {
      label: 'Data Sharing',
      description: 'Share anonymized data for research',
      type: 'toggle',
      value: privacy.dataSharing,
      onChange: (value: boolean) => setPrivacy({...privacy, dataSharing: value})
    },
    {
      label: 'Usage Analytics',
      description: 'Help improve the app with usage data',
      type: 'toggle',
      value: privacy.analytics,
      onChange: (value: boolean) => setPrivacy({...privacy, analytics: value})
    },
    {
      label: 'Biometric Authentication',
      description: 'Use fingerprint or face recognition',
      type: 'toggle',
      value: privacy.biometrics,
      onChange: (value: boolean) => setPrivacy({...privacy, biometrics: value})
    },
    {
      label: 'Change Password',
      description: 'Update your account password',
      type: 'action',
      action: () => openModal('changePassword')
    },
    {
      label: 'Two-Factor Authentication',
      description: 'Add an extra layer of security',
      type: 'action',
      action: () => openModal('twoFactor')
    }
  ]
});