import { Database } from 'lucide-react';
import { SettingSection } from '../types';
import { ModalType } from '../hooks/useSettingsModals';

export const createDataStorageSection = (
  openModal: (modal: ModalType) => void
): SettingSection => ({
  title: 'Data & Storage',
  icon: Database,
  items: [
    {
      label: 'Sync Settings',
      description: 'Manage cloud synchronization',
      type: 'action',
      action: () => openModal('syncSettings')
    },
    {
      label: 'Export Data',
      description: 'Download your health data',
      type: 'action',
      action: () => openModal('exportData')
    },
    {
      label: 'Storage Usage',
      description: '2.3 GB of 5 GB used',
      type: 'action',
      action: () => openModal('storageUsage')
    },
    {
      label: 'Clear Cache',
      description: 'Free up space by clearing cached data',
      type: 'action',
      action: () => openModal('clearCache')
    }
  ]
});