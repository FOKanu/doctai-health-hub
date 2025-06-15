
import React, { useState } from 'react';
import { Calendar, Plus, Filter, Search, Download } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { AppointmentOverview } from './appointments/AppointmentOverview';
import { AppointmentCalendar } from './appointments/AppointmentCalendar';
import { AppointmentFilters } from './appointments/AppointmentFilters';
import { NewAppointmentDialog } from './appointments/NewAppointmentDialog';
import { AppointmentReminders } from './appointments/AppointmentReminders';

const AppointmentsScreen = () => {
  const [viewMode, setViewMode] = useState<'month' | 'week'>('month');
  const [selectedDate, setSelectedDate] = useState<Date>(new Date());
  const [showNewAppointment, setShowNewAppointment] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [filterType, setFilterType] = useState('all');
  const [filterTimeRange, setFilterTimeRange] = useState('30d');

  return (
    <div className="space-y-6">
      {/* Header Section */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Appointments</h1>
          <p className="text-gray-600">Manage your schedule and health appointments</p>
        </div>
        <div className="flex gap-2">
          <Button
            onClick={() => setShowNewAppointment(true)}
            className="bg-blue-600 hover:bg-blue-700"
          >
            <Plus className="w-4 h-4 mr-2" />
            New Appointment
          </Button>
          <Button variant="outline">
            <Download className="w-4 h-4 mr-2" />
            Export
          </Button>
        </div>
      </div>

      {/* Overview Panel */}
      <AppointmentOverview />

      {/* Search and Filters */}
      <Card>
        <CardContent className="p-4">
          <div className="flex flex-col sm:flex-row gap-4">
            <div className="flex-1">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
                <Input
                  placeholder="Search appointments..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="pl-10"
                />
              </div>
            </div>
            <AppointmentFilters
              filterType={filterType}
              setFilterType={setFilterType}
              filterTimeRange={filterTimeRange}
              setFilterTimeRange={setFilterTimeRange}
            />
          </div>
        </CardContent>
      </Card>

      {/* Calendar View Controls */}
      <div className="flex justify-between items-center">
        <div className="flex gap-2">
          <Button
            variant={viewMode === 'month' ? 'default' : 'outline'}
            onClick={() => setViewMode('month')}
          >
            Month
          </Button>
          <Button
            variant={viewMode === 'week' ? 'default' : 'outline'}
            onClick={() => setViewMode('week')}
          >
            Week
          </Button>
        </div>
        <div className="text-sm text-gray-600">
          {selectedDate.toLocaleDateString('en-US', { 
            month: 'long', 
            year: 'numeric' 
          })}
        </div>
      </div>

      {/* Calendar Component */}
      <AppointmentCalendar
        viewMode={viewMode}
        selectedDate={selectedDate}
        onDateSelect={setSelectedDate}
        searchTerm={searchTerm}
        filterType={filterType}
        filterTimeRange={filterTimeRange}
      />

      {/* Smart Reminders */}
      <AppointmentReminders />

      {/* New Appointment Dialog */}
      <NewAppointmentDialog
        open={showNewAppointment}
        onOpenChange={setShowNewAppointment}
      />
    </div>
  );
};

export default AppointmentsScreen;
