import { Globe } from 'lucide-react';
import { SettingSection } from '../types';
import { ModalType } from '../hooks/useSettingsModals';

export const createPreferencesSection = (
  openModal: (modal: ModalType) => void
): SettingSection => ({
  title: 'Preferences',
  icon: Globe,
  items: [
    {
      label: 'Language',
      description: 'English (US)',
      type: 'action',
      action: () => openModal('language')
    },
    {
      label: 'Date Format',
      description: 'MM/DD/YYYY',
      type: 'action',
      action: () => openModal('dateFormat')
    },
    {
      label: 'Time Zone',
      description: 'Eastern Time (ET)',
      type: 'action',
      action: () => openModal('timeZone')
    },
    {
      label: 'Units',
      description: 'Imperial (lbs, ft, °F)',
      type: 'action',
      action: () => openModal('units')
    }
  ]
});