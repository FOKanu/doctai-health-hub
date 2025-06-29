
import React from 'react';

interface DistanceFilterSectionProps {
  maxDistance: number;
  onDistanceChange: (distance: number) => void;
}

const DistanceFilterSection: React.FC<DistanceFilterSectionProps> = ({
  maxDistance,
  onDistanceChange
}) => {
  return (
    <div>
      <h3 className="font-medium mb-3">Distance</h3>
      <div className="space-y-2">
        <input
          type="range"
          min="1"
          max="50"
          value={maxDistance}
          onChange={(e) => onDistanceChange(parseInt(e.target.value))}
          className="w-full"
        />
        <div className="flex justify-between text-sm text-gray-600">
          <span>1 km</span>
          <span>{maxDistance} km</span>
          <span>50 km</span>
        </div>
      </div>
    </div>
  );
};

export default DistanceFilterSection;
