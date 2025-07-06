import { HelpCircle } from 'lucide-react';
import { SettingSection } from '../types';
import { ModalType } from '../hooks/useSettingsModals';

export const createSupportSection = (
  openModal: (modal: ModalType) => void
): SettingSection => ({
  title: 'Support',
  icon: HelpCircle,
  items: [
    {
      label: 'Help Center',
      description: 'Get answers to common questions',
      type: 'action',
      action: () => openModal('helpCenter')
    },
    {
      label: 'Contact Support',
      description: 'Get help from our support team',
      type: 'action',
      action: () => openModal('contactSupport')
    },
    {
      label: 'Report a Bug',
      description: 'Help us improve the app',
      type: 'action',
      action: () => openModal('reportBug')
    },
    {
      label: 'Privacy Policy',
      description: 'Review our privacy policy',
      type: 'action',
      action: () => openModal('privacyPolicy')
    },
    {
      label: 'Terms of Service',
      description: 'Review terms and conditions',
      type: 'action',
      action: () => openModal('termsOfService')
    }
  ]
});