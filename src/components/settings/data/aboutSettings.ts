import { Info } from 'lucide-react';
import { SettingSection } from '../types';

export const createAboutSection = (): SettingSection => ({
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
});