import { SettingSection, NotificationSettings, PrivacySettings } from '../types';
import { ModalType } from '../hooks/useSettingsModals';
import { createNotificationSection } from './notificationSettings';
import { createPrivacySection } from './privacySettings';
import { createPreferencesSection } from './preferencesSettings';
import { createDataStorageSection } from './dataStorageSettings';
import { createSupportSection } from './supportSettings';
import { createAboutSection } from './aboutSettings';

export const createSettingsSections = (
  notifications: NotificationSettings,
  setNotifications: React.Dispatch<React.SetStateAction<NotificationSettings>>,
  privacy: PrivacySettings,
  setPrivacy: React.Dispatch<React.SetStateAction<PrivacySettings>>,
  openModal: (modal: ModalType) => void
): SettingSection[] => [
  createNotificationSection(notifications, setNotifications),
  createPrivacySection(privacy, setPrivacy, openModal),
  createPreferencesSection(openModal),
  createDataStorageSection(openModal),
  createSupportSection(openModal),
  createAboutSection()
];