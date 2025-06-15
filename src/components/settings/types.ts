
export interface SettingItem {
  label: string;
  description: string;
  type: 'toggle' | 'action' | 'info';
  value?: boolean;
  onChange?: (value: boolean) => void;
  action?: () => void;
}

export interface SettingSection {
  title: string;
  icon: any;
  items: SettingItem[];
}

export interface NotificationSettings {
  appointments: boolean;
  medications: boolean;
  labResults: boolean;
  reminders: boolean;
}

export interface PrivacySettings {
  dataSharing: boolean;
  analytics: boolean;
  biometrics: boolean;
}
