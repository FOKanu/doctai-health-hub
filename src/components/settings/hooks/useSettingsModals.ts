import { useState } from 'react';

export type ModalType = 
  | 'changePassword'
  | 'twoFactor'
  | 'language'
  | 'dateFormat' 
  | 'timeZone'
  | 'units'
  | 'syncSettings'
  | 'exportData'
  | 'storageUsage'
  | 'clearCache'
  | 'helpCenter'
  | 'contactSupport'
  | 'reportBug'
  | 'privacyPolicy'
  | 'termsOfService';

export const useSettingsModals = () => {
  const [activeModal, setActiveModal] = useState<ModalType | null>(null);

  const openModal = (modal: ModalType) => {
    setActiveModal(modal);
  };

  const closeModal = () => {
    setActiveModal(null);
  };

  return {
    activeModal,
    openModal,
    closeModal
  };
};