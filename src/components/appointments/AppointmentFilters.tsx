
import React from 'react';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Filter } from 'lucide-react';

interface AppointmentFiltersProps {
  filterType: string;
  setFilterType: (value: string) => void;
  filterTimeRange: string;
  setFilterTimeRange: (value: string) => void;
}

export const AppointmentFilters = ({
  filterType,
  setFilterType,
  filterTimeRange,
  setFilterTimeRange
}: AppointmentFiltersProps) => {
  return (
    <div className="flex gap-2">
      <Select value={filterType} onValueChange={setFilterType}>
        <SelectTrigger className="w-40">
          <Filter className="w-4 h-4 mr-2" />
          <SelectValue placeholder="Filter by type" />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="all">All Types</SelectItem>
          <SelectItem value="medical">🔵 Medical</SelectItem>
          <SelectItem value="fitness">🟢 Fitness</SelectItem>
          <SelectItem value="dental">🟣 Dental</SelectItem>
          <SelectItem value="therapy">🔶 Therapy</SelectItem>
          <SelectItem value="custom">Custom</SelectItem>
        </SelectContent>
      </Select>

      <Select value={filterTimeRange} onValueChange={setFilterTimeRange}>
        <SelectTrigger className="w-32">
          <SelectValue placeholder="Time range" />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="today">Today</SelectItem>
          <SelectItem value="7d">7 Days</SelectItem>
          <SelectItem value="30d">30 Days</SelectItem>
          <SelectItem value="custom">Custom</SelectItem>
        </SelectContent>
      </Select>
    </div>
  );
};
