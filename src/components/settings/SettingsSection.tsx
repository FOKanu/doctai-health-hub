
import React from 'react';
import { SettingsItem } from './SettingsItem';
import { SettingSection } from './types';

interface SettingsSectionProps {
  section: SettingSection;
}

export const SettingsSection = ({ section }: SettingsSectionProps) => {
  const SectionIcon = section.icon;
  
  return (
    <div className="mb-6">
      <div className="flex items-center space-x-2 mb-3">
        <SectionIcon className="w-5 h-5 text-gray-600" />
        <h2 className="text-lg font-semibold text-gray-800">{section.title}</h2>
      </div>
      
      <div className="bg-white rounded-lg shadow-sm">
        {section.items.map((item, itemIndex) => (
          <SettingsItem
            key={itemIndex}
            item={item}
            isLast={itemIndex === section.items.length - 1}
          />
        ))}
      </div>
    </div>
  );
};
