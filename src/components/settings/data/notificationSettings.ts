import { Bell } from 'lucide-react';
import { SettingSection, NotificationSettings } from '../types';

export const createNotificationSection = (
  notifications: NotificationSettings,
  setNotifications: React.Dispatch<React.SetStateAction<NotificationSettings>>
): SettingSection => ({
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
});