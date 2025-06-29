
import React from 'react';
import { X, Filter } from 'lucide-react';

interface FilterHeaderProps {
  onClose: () => void;
}

const FilterHeader: React.FC<FilterHeaderProps> = ({ onClose }) => {
  return (
    <div className="p-4 border-b flex items-center justify-between">
      <div className="flex items-center">
        <Filter className="w-5 h-5 mr-2 text-gray-600" />
        <h2 className="text-lg font-semibold">Filters</h2>
      </div>
      <button onClick={onClose} className="md:hidden p-1 hover:bg-gray-100 rounded">
        <X className="w-5 h-5" />
      </button>
    </div>
  );
};

export default FilterHeader;
