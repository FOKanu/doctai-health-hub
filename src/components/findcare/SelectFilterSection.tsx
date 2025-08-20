
import React from 'react';

interface SelectFilterSectionProps {
  title: string;
  value: string | number;
  onChange: (value: string) => void;
  options: { value: string | number; label: string }[];
}

const SelectFilterSection: React.FC<SelectFilterSectionProps> = ({
  title,
  value,
  onChange,
  options
}) => {
  return (
    <div>
      <h3 className="font-medium mb-3">{title}</h3>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full p-2 border border-gray-300 rounded-lg"
      >
        {options.map(option => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </div>
  );
};

export default SelectFilterSection;
