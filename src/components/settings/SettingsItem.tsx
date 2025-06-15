
import React from 'react';
import { ChevronRight } from 'lucide-react';
import { ToggleSwitch } from './ToggleSwitch';
import { SettingItem } from './types';

interface SettingsItemProps {
  item: SettingItem;
  isLast: boolean;
}

export const SettingsItem = ({ item, isLast }: SettingsItemProps) => {
  return (
    <div
      className={`p-4 ${!isLast ? 'border-b border-gray-100' : ''}`}
    >
      <div className="flex items-center justify-between">
        <div className="flex-1">
          <h3 className="font-medium text-gray-800">{item.label}</h3>
          <p className="text-sm text-gray-500 mt-1">{item.description}</p>
        </div>
        
        <div className="ml-4">
          {item.type === 'toggle' && (
            <ToggleSwitch
              enabled={item.value as boolean}
              onChange={item.onChange as (value: boolean) => void}
            />
          )}
          {item.type === 'action' && (
            <button
              onClick={item.action}
              className="p-1 text-gray-400 hover:text-gray-600 rounded"
            >
              <ChevronRight className="w-5 h-5" />
            </button>
          )}
          {item.type === 'info' && (
            <div className="text-right">
              <span className="text-sm text-gray-500">{item.description}</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
