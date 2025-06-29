import React from 'react';
import { X, Filter } from 'lucide-react';
import { FilterOptions } from '../../types/healthcare';

interface FilterSidebarProps {
  filters: FilterOptions;
  onFiltersChange: (filters: FilterOptions) => void;
  isOpen: boolean;
  onClose: () => void;
}

const FilterSidebar: React.FC<FilterSidebarProps> = ({
  filters,
  onFiltersChange,
  isOpen,
  onClose
}) => {
  const providerTypes = [
    { value: 'doctor', label: 'Doctor' },
    { value: 'dentist', label: 'Dentist' },
    { value: 'physiotherapist', label: 'Physiotherapist' },
    { value: 'psychotherapist', label: 'Psychotherapist' },
    { value: 'hospital', label: 'Hospital' },
    { value: 'clinic', label: 'Clinic' },
    { value: 'pharmacy', label: 'Pharmacy' }
  ];

  const specialties = [
    'Cardiology', 'Dermatology', 'Neurology', 'Orthopedics', 'Pediatrics',
    'Psychiatry', 'Gynecology', 'Urology', 'Oncology', 'Endocrinology',
    'General Practice', 'Emergency Medicine', 'Radiology', 'Anesthesiology'
  ];

  const languages = [
    'English', 'German', 'Spanish', 'French', 'Italian', 'Turkish', 'Arabic', 'Russian'
  ];

  const insuranceOptions = [
    'Public Health Insurance', 'Private Insurance', 'Self-Pay', 'Medicare', 'Medicaid'
  ];

  const updateFilter = (key: keyof FilterOptions, value: any) => {
    onFiltersChange({ ...filters, [key]: value });
  };

  const toggleArrayFilter = (key: keyof FilterOptions, value: string) => {
    const currentArray = filters[key] as string[];
    const updatedArray = currentArray.includes(value)
      ? currentArray.filter(item => item !== value)
      : [...currentArray, value];
    updateFilter(key, updatedArray);
  };

  return (
    <>
      {/* Mobile Overlay */}
      {isOpen && (
        <div className="fixed inset-0 bg-black bg-opacity-50 z-30 md:hidden" onClick={onClose} />
      )}

      {/* Sidebar */}
      <div className={`
        fixed md:sticky top-0 left-0 h-full md:h-auto w-80 bg-white shadow-lg z-20
        transform transition-transform duration-300 ease-in-out
        ${isOpen ? 'translate-x-0' : '-translate-x-full md:translate-x-0'}
        overflow-y-auto
      `}>
        <div className="p-4 border-b flex items-center justify-between">
          <div className="flex items-center">
            <Filter className="w-5 h-5 mr-2 text-gray-600" />
            <h2 className="text-lg font-semibold">Filters</h2>
          </div>
          <button onClick={onClose} className="md:hidden p-1 hover:bg-gray-100 rounded">
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="p-4 space-y-6">
          {/* Provider Type */}
          <div>
            <h3 className="font-medium mb-3">Provider Type</h3>
            <div className="space-y-2">
              {providerTypes.map(type => (
                <label key={type.value} className="flex items-center">
                  <input
                    type="checkbox"
                    checked={filters.providerType.includes(type.value)}
                    onChange={() => toggleArrayFilter('providerType', type.value)}
                    className="mr-2 rounded"
                  />
                  <span className="text-sm">{type.label}</span>
                </label>
              ))}
            </div>
          </div>

          {/* Specialties */}
          <div>
            <h3 className="font-medium mb-3">Specialties</h3>
            <div className="space-y-2 max-h-48 overflow-y-auto">
              {specialties.map(specialty => (
                <label key={specialty} className="flex items-center">
                  <input
                    type="checkbox"
                    checked={filters.specialties.includes(specialty)}
                    onChange={() => toggleArrayFilter('specialties', specialty)}
                    className="mr-2 rounded"
                  />
                  <span className="text-sm">{specialty}</span>
                </label>
              ))}
            </div>
          </div>

          {/* Distance */}
          <div>
            <h3 className="font-medium mb-3">Distance</h3>
            <div className="space-y-2">
              <input
                type="range"
                min="1"
                max="50"
                value={filters.maxDistance}
                onChange={(e) => updateFilter('maxDistance', parseInt(e.target.value))}
                className="w-full"
              />
              <div className="flex justify-between text-sm text-gray-600">
                <span>1 km</span>
                <span>{filters.maxDistance} km</span>
                <span>50 km</span>
              </div>
            </div>
          </div>

          {/* Rating */}
          <div>
            <h3 className="font-medium mb-3">Minimum Rating</h3>
            <select
              value={filters.minRating}
              onChange={(e) => updateFilter('minRating', parseFloat(e.target.value))}
              className="w-full p-2 border border-gray-300 rounded-lg"
            >
              <option value={0}>Any Rating</option>
              <option value={3}>3.0+ Stars</option>
              <option value={4}>4.0+ Stars</option>
              <option value={4.5}>4.5+ Stars</option>
            </select>
          </div>

          {/* Languages */}
          <div>
            <h3 className="font-medium mb-3">Languages Spoken</h3>
            <div className="space-y-2">
              {languages.map(language => (
                <label key={language} className="flex items-center">
                  <input
                    type="checkbox"
                    checked={filters.languages.includes(language)}
                    onChange={() => toggleArrayFilter('languages', language)}
                    className="mr-2 rounded"
                  />
                  <span className="text-sm">{language}</span>
                </label>
              ))}
            </div>
          </div>

          {/* Availability */}
          <div>
            <h3 className="font-medium mb-3">Availability</h3>
            <select
              value={filters.availability}
              onChange={(e) => updateFilter('availability', e.target.value)}
              className="w-full p-2 border border-gray-300 rounded-lg"
            >
              <option value="any">Any Time</option>
              <option value="today">Available Today</option>
              <option value="this_week">This Week</option>
            </select>
          </div>

          {/* Insurance */}
          <div>
            <h3 className="font-medium mb-3">Insurance Accepted</h3>
            <div className="space-y-2">
              {insuranceOptions.map(insurance => (
                <label key={insurance} className="flex items-center">
                  <input
                    type="checkbox"
                    checked={filters.insuranceAccepted.includes(insurance)}
                    onChange={() => toggleArrayFilter('insuranceAccepted', insurance)}
                    className="mr-2 rounded"
                  />
                  <span className="text-sm">{insurance}</span>
                </label>
              ))}
            </div>
          </div>

          {/* Price Range */}
          <div>
            <h3 className="font-medium mb-3">Price Range</h3>
            <div className="space-y-2">
              {['low', 'medium', 'high'].map(price => (
                <label key={price} className="flex items-center">
                  <input
                    type="checkbox"
                    checked={filters.priceRange.includes(price)}
                    onChange={() => toggleArrayFilter('priceRange', price)}
                    className="mr-2 rounded"
                  />
                  <span className="text-sm capitalize">{price}</span>
                </label>
              ))}
            </div>
          </div>

          {/* Clear Filters */}
          <button
            onClick={() => onFiltersChange({
              providerType: [],
              specialties: [],
              maxDistance: 25,
              minRating: 0,
              languages: [],
              availability: 'any',
              insuranceAccepted: [],
              priceRange: []
            })}
            className="w-full py-2 text-blue-600 border border-blue-600 rounded-lg hover:bg-blue-50 transition-colors"
          >
            Clear All Filters
          </button>
        </div>
      </div>
    </>
  );
};

export default FilterSidebar;
