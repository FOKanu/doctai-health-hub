
import React from 'react';

interface ToggleSwitchProps {
  enabled: boolean;
  onChange: (value: boolean) => void;
}

export const ToggleSwitch = ({ enabled, onChange }: ToggleSwitchProps) => (
  <button
    onClick={() => onChange(!enabled)}
    className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
      enabled ? 'bg-blue-600' : 'bg-gray-300'
    }`}
  >
    <span
      className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
        enabled ? 'translate-x-6' : 'translate-x-1'
      }`}
    />
  </button>
);
