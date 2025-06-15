
import { Bell, Shield, Globe, Database, HelpCircle, Info } from 'lucide-react';
import { SettingSection, NotificationSettings, PrivacySettings } from './types';

export const createSettingsSections = (
  notifications: NotificationSettings,
  setNotifications: React.Dispatch<React.SetStateAction<NotificationSettings>>,
  privacy: PrivacySettings,
  setPrivacy: React.Dispatch<React.SetStateAction<PrivacySettings>>
): SettingSection[] => [
  {
    title: 'Notifications',
    icon: Bell,
    items: [
      {
        label: 'Appointment Reminders',
        description: 'Get notified about upcoming appointments',
        type: 'toggle',
        value: notifications.appointments,
        onChange: (value: boolean) => setNotifications({...notifications, appointments: value})
      },
      {
        label: 'Medication Reminders',
        description: 'Reminders to take your medications',
        type: 'toggle',
        value: notifications.medications,
        onChange: (value: boolean) => setNotifications({...notifications, medications: value})
      },
      {
        label: 'Lab Results',
        description: 'Notifications when new results are available',
        type: 'toggle',
        value: notifications.labResults,
        onChange: (value: boolean) => setNotifications({...notifications, labResults: value})
      },
      {
        label: 'Health Reminders',
        description: 'General health and wellness reminders',
        type: 'toggle',
        value: notifications.reminders,
        onChange: (value: boolean) => setNotifications({...notifications, reminders: value})
      }
    ]
  },
  {
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
        action: () => console.log('Change password')
      },
      {
        label: 'Two-Factor Authentication',
        description: 'Add an extra layer of security',
        type: 'action',
        action: () => console.log('Setup 2FA')
      }
    ]
  },
  {
    title: 'Preferences',
    icon: Globe,
    items: [
      {
        label: 'Language',
        description: 'English (US)',
        type: 'action',
        action: () => console.log('Change language')
      },
      {
        label: 'Date Format',
        description: 'MM/DD/YYYY',
        type: 'action',
        action: () => console.log('Change date format')
      },
      {
        label: 'Time Zone',
        description: 'Eastern Time (ET)',
        type: 'action',
        action: () => console.log('Change timezone')
      },
      {
        label: 'Units',
        description: 'Imperial (lbs, ft, °F)',
        type: 'action',
        action: () => console.log('Change units')
      }
    ]
  },
  {
    title: 'Data & Storage',
    icon: Database,
    items: [
      {
        label: 'Sync Settings',
        description: 'Manage cloud synchronization',
        type: 'action',
        action: () => console.log('Sync settings')
      },
      {
        label: 'Export Data',
        description: 'Download your health data',
        type: 'action',
        action: () => console.log('Export data')
      },
      {
        label: 'Storage Usage',
        description: '2.3 GB of 5 GB used',
        type: 'action',
        action: () => console.log('Storage usage')
      },
      {
        label: 'Clear Cache',
        description: 'Free up space by clearing cached data',
        type: 'action',
        action: () => console.log('Clear cache')
      }
    ]
  },
  {
    title: 'Support',
    icon: HelpCircle,
    items: [
      {
        label: 'Help Center',
        description: 'Get answers to common questions',
        type: 'action',
        action: () => console.log('Help center')
      },
      {
        label: 'Contact Support',
        description: 'Get help from our support team',
        type: 'action',
        action: () => console.log('Contact support')
      },
      {
        label: 'Report a Bug',
        description: 'Help us improve the app',
        type: 'action',
        action: () => console.log('Report bug')
      },
      {
        label: 'Privacy Policy',
        description: 'Review our privacy policy',
        type: 'action',
        action: () => console.log('Privacy policy')
      },
      {
        label: 'Terms of Service',
        description: 'Review terms and conditions',
        type: 'action',
        action: () => console.log('Terms of service')
      }
    ]
  },
  {
    title: 'About',
    icon: Info,
    items: [
      {
        label: 'App Version',
        description: '1.0.0 (Build 24)',
        type: 'info'
      },
      {
        label: 'Last Updated',
        description: 'January 15, 2024',
        type: 'info'
      },
      {
        label: 'Device ID',
        description: 'DV-2024-001',
        type: 'info'
      }
    ]
  }
];
