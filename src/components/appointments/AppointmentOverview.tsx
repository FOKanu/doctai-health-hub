
import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Calendar, Clock, AlertTriangle, CheckCircle } from 'lucide-react';
import { Badge } from '@/components/ui/badge';

export const AppointmentOverview = () => {
  // Mock data - in a real app, this would come from your state/API
  const stats = {
    total: 42,
    upcoming7: 3,
    upcoming30: 8,
    missed: 2,
    cancelled: 1
  };

  const nextAppointment = {
    type: 'Medical Checkup',
    date: 'Tomorrow',
    time: '2:30 PM',
    location: 'Downtown Medical Center',
    provider: 'Dr. Smith',
    category: 'medical'
  };

  const getCategoryColor = (category: string) => {
    const colors = {
      medical: 'bg-blue-100 text-blue-800',
      dental: 'bg-purple-100 text-purple-800',
      fitness: 'bg-green-100 text-green-800',
      therapy: 'bg-orange-100 text-orange-800'
    };
    return colors[category as keyof typeof colors] || 'bg-gray-100 text-gray-800';
  };

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
      {/* Stats Cards */}
      <Card>
        <CardContent className="p-4">
          <div className="flex items-center space-x-2">
            <Calendar className="w-5 h-5 text-blue-600" />
            <div>
              <p className="text-sm font-medium text-gray-600">Total Appointments</p>
              <p className="text-2xl font-bold text-gray-900">{stats.total}</p>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardContent className="p-4">
          <div className="flex items-center space-x-2">
            <Clock className="w-5 h-5 text-green-600" />
            <div>
              <p className="text-sm font-medium text-gray-600">Next 7 Days</p>
              <p className="text-2xl font-bold text-gray-900">{stats.upcoming7}</p>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardContent className="p-4">
          <div className="flex items-center space-x-2">
            <CheckCircle className="w-5 h-5 text-purple-600" />
            <div>
              <p className="text-sm font-medium text-gray-600">Next 30 Days</p>
              <p className="text-2xl font-bold text-gray-900">{stats.upcoming30}</p>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardContent className="p-4">
          <div className="flex items-center space-x-2">
            <AlertTriangle className="w-5 h-5 text-red-600" />
            <div>
              <p className="text-sm font-medium text-gray-600">Missed/Cancelled</p>
              <p className="text-2xl font-bold text-gray-900">{stats.missed + stats.cancelled}</p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Next Appointment Card - spans full width on mobile, 2 columns on larger screens */}
      <Card className="md:col-span-2 lg:col-span-4">
        <CardHeader>
          <CardTitle className="text-lg">Next Appointment</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Badge className={getCategoryColor(nextAppointment.category)}>
                  {nextAppointment.type}
                </Badge>
              </div>
              <p className="text-lg font-semibold">{nextAppointment.date} at {nextAppointment.time}</p>
              <p className="text-gray-600">{nextAppointment.location}</p>
              <p className="text-sm text-gray-500">with {nextAppointment.provider}</p>
            </div>
            <div className="flex gap-2">
              <Badge variant="outline" className="text-green-600 border-green-600">
                Reminder Set
              </Badge>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
