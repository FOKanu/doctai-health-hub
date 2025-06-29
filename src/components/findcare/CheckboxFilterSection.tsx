
import React from 'react';

interface CheckboxFilterSectionProps {
  title: string;
  options: string[] | { value: string; label: string }[];
  selectedValues: string[];
  onToggle: (value: string) => void;
  maxHeight?: string;
}

const CheckboxFilterSection: React.FC<CheckboxFilterSectionProps> = ({
  title,
  options,
  selectedValues,
  onToggle,
  maxHeight = 'auto'
}) => {
  return (
    <div>
      <h3 className="font-medium mb-3">{title}</h3>
      <div className={`space-y-2 ${maxHeight !== 'auto' ? `max-h-${maxHeight} overflow-y-auto` : ''}`}>
        {options.map(option => {
          const value = typeof option === 'string' ? option : option.value;
          const label = typeof option === 'string' ? option : option.label;
          
          return (
            <label key={value} className="flex items-center">
              <input
                type="checkbox"
                checked={selectedValues.includes(value)}
                onChange={() => onToggle(value)}
                className="mr-2 rounded"
              />
              <span className="text-sm">{label}</span>
            </label>
          );
        })}
      </div>
    </div>
  );
};

export default CheckboxFilterSection;
