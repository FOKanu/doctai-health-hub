
import React from 'react';
import { FilterOptions } from '../../types/healthcare';
import FilterHeader from './FilterHeader';
import CheckboxFilterSection from './CheckboxFilterSection';
import DistanceFilterSection from './DistanceFilterSection';
import SelectFilterSection from './SelectFilterSection';
import { providerTypes, specialties, languages, insuranceOptions } from './filterData';

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

  const clearAllFilters = () => {
    onFiltersChange({
      providerType: [],
      specialties: [],
      maxDistance: 25,
      minRating: 0,
      languages: [],
      availability: 'any',
      insuranceAccepted: [],
      priceRange: []
    });
  };

  const ratingOptions = [
    { value: 0, label: 'Any Rating' },
    { value: 3, label: '3.0+ Stars' },
    { value: 4, label: '4.0+ Stars' },
    { value: 4.5, label: '4.5+ Stars' }
  ];

  const availabilityOptions = [
    { value: 'any', label: 'Any Time' },
    { value: 'today', label: 'Available Today' },
    { value: 'this_week', label: 'This Week' }
  ];

  const priceRangeOptions = ['low', 'medium', 'high'];

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
        <FilterHeader onClose={onClose} />

        <div className="p-4 space-y-6">
          <CheckboxFilterSection
            title="Provider Type"
            options={providerTypes}
            selectedValues={filters.providerType}
            onToggle={(value: string) => toggleArrayFilter('providerType', value)}
          />

          <CheckboxFilterSection
            title="Specialties"
            options={specialties}
            selectedValues={filters.specialties}
            onToggle={(value: string) => toggleArrayFilter('specialties', value)}
            maxHeight="48"
          />

          <DistanceFilterSection
            maxDistance={filters.maxDistance}
            onDistanceChange={(distance: number) => updateFilter('maxDistance', distance)}
          />

          <SelectFilterSection
            title="Minimum Rating"
            value={filters.minRating}
            onChange={(value: string) => updateFilter('minRating', parseFloat(value))}
            options={ratingOptions}
          />

          <CheckboxFilterSection
            title="Languages Spoken"
            options={languages}
            selectedValues={filters.languages}
            onToggle={(value: string) => toggleArrayFilter('languages', value)}
          />

          <SelectFilterSection
            title="Availability"
            value={filters.availability}
            onChange={(value: string) => updateFilter('availability', value)}
            options={availabilityOptions}
          />

          <CheckboxFilterSection
            title="Insurance Accepted"
            options={insuranceOptions}
            selectedValues={filters.insuranceAccepted}
            onToggle={(value: string) => toggleArrayFilter('insuranceAccepted', value)}
          />

          <CheckboxFilterSection
            title="Price Range"
            options={priceRangeOptions}
            selectedValues={filters.priceRange}
            onToggle={(value) => toggleArrayFilter('priceRange', value)}
          />

          <button
            onClick={clearAllFilters}
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
