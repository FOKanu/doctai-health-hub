
import React, { useState } from 'react';
import { Calendar } from '@/components/ui/calendar';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Clock, MapPin, Edit, Trash2, MoreHorizontal } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from '@/components/ui/dropdown-menu';

interface AppointmentCalendarProps {
  viewMode: 'month' | 'week';
  selectedDate: Date;
  onDateSelect: (date: Date) => void;
  searchTerm: string;
  filterType: string;
  filterTimeRange: string;
}

export const AppointmentCalendar = ({
  viewMode,
  selectedDate,
  onDateSelect,
  searchTerm,
  filterType,
  filterTimeRange
}: AppointmentCalendarProps) => {
  // Mock appointments data
  const appointments = [
    {
      id: 1,
      title: 'Medical Checkup',
      type: 'medical',
      date: new Date(),
      time: '2:30 PM',
      duration: '30 min',
      location: 'Downtown Medical Center',
      provider: 'Dr. Smith',
      isRecurring: false
    },
    {
      id: 2,
      title: 'Gym Session',
      type: 'fitness',
      date: new Date(Date.now() + 86400000), // Tomorrow
      time: '6:00 AM',
      duration: '1 hour',
      location: 'FitLife Gym',
      provider: 'Personal Trainer',
      isRecurring: true
    },
    {
      id: 3,
      title: 'Dental Cleaning',
      type: 'dental',
      date: new Date(Date.now() + 7 * 86400000), // Next week
      time: '10:00 AM',
      duration: '45 min',
      location: 'Smile Dental Clinic',
      provider: 'Dr. Johnson',
      isRecurring: false
    }
  ];

  const getCategoryColor = (type: string) => {
    const colors = {
      medical: 'bg-blue-500',
      fitness: 'bg-green-500',
      dental: 'bg-purple-500',
      therapy: 'bg-orange-500'
    };
    return colors[type as keyof typeof colors] || 'bg-gray-500';
  };

  const getCategoryBadgeColor = (type: string) => {
    const colors = {
      medical: 'bg-blue-100 text-blue-800',
      fitness: 'bg-green-100 text-green-800',
      dental: 'bg-purple-100 text-purple-800',
      therapy: 'bg-orange-100 text-orange-800'
    };
    return colors[type as keyof typeof colors] || 'bg-gray-100 text-gray-800';
  };

  const filteredAppointments = appointments.filter(apt => {
    const matchesSearch = apt.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         apt.provider.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesType = filterType === 'all' || apt.type === filterType;
    return matchesSearch && matchesType;
  });

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      {/* Calendar */}
      <Card className="lg:col-span-2">
        <CardContent className="p-4">
          <Calendar
            mode="single"
            selected={selectedDate}
            onSelect={(date) => date && onDateSelect(date)}
            className="w-full"
          />
        </CardContent>
      </Card>

      {/* Appointments List */}
      <Card>
        <CardContent className="p-4">
          <h3 className="font-semibold mb-4">
            Appointments for {selectedDate.toLocaleDateString()}
          </h3>
          <div className="space-y-3">
            {filteredAppointments.map((appointment) => (
              <div
                key={appointment.id}
                className="border rounded-lg p-3 hover:shadow-md transition-shadow"
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <div className={`w-3 h-3 rounded-full ${getCategoryColor(appointment.type)}`} />
                      <Badge className={getCategoryBadgeColor(appointment.type)}>
                        {appointment.title}
                      </Badge>
                      {appointment.isRecurring && (
                        <Badge variant="outline" className="text-xs">
                          Recurring
                        </Badge>
                      )}
                    </div>
                    <div className="space-y-1 text-sm text-gray-600">
                      <div className="flex items-center gap-1">
                        <Clock className="w-3 h-3" />
                        {appointment.time} ({appointment.duration})
                      </div>
                      <div className="flex items-center gap-1">
                        <MapPin className="w-3 h-3" />
                        {appointment.location}
                      </div>
                      <p className="text-xs">{appointment.provider}</p>
                    </div>
                  </div>
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <Button variant="ghost" size="sm">
                        <MoreHorizontal className="w-4 h-4" />
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="end">
                      <DropdownMenuItem>
                        <Edit className="w-4 h-4 mr-2" />
                        Edit
                      </DropdownMenuItem>
                      <DropdownMenuItem>
                        <Calendar className="w-4 h-4 mr-2" />
                        Reschedule
                      </DropdownMenuItem>
                      <DropdownMenuItem className="text-red-600">
                        <Trash2 className="w-4 h-4 mr-2" />
                        Cancel
                      </DropdownMenuItem>
                    </DropdownMenuContent>
                  </DropdownMenu>
                </div>
              </div>
            ))}
            {filteredAppointments.length === 0 && (
              <p className="text-gray-500 text-center py-4">
                No appointments found for this date
              </p>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
