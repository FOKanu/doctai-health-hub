
import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Calendar, Pill, Clock, TrendingUp, Target } from 'lucide-react';

interface AppointmentsMedicationsProps {
  dateRange: string;
}

export const AppointmentsMedications: React.FC<AppointmentsMedicationsProps> = ({ dateRange }) => {
  const appointmentData = [
    { month: 'Jan', scheduled: 4, completed: 4, noShow: 0 },
    { month: 'Feb', scheduled: 3, completed: 2, noShow: 1 },
    { month: 'Mar', scheduled: 5, completed: 5, noShow: 0 },
    { month: 'Apr', scheduled: 2, completed: 2, noShow: 0 },
  ];

  const medicationAdherence = [
    { date: '2024-02-08', adherence: 95 },
    { date: '2024-02-09', adherence: 100 },
    { date: '2024-02-10', adherence: 87 },
    { date: '2024-02-11', adherence: 92 },
    { date: '2024-02-12', adherence: 100 },
    { date: '2024-02-13', adherence: 95 },
    { date: '2024-02-14', adherence: 100 },
  ];

  const upcomingAppointments = [
    { id: 1, type: 'Dermatology', date: '2024-02-20', time: '10:00 AM', doctor: 'Dr. Sarah Chen' },
    { id: 2, type: 'Cardiology', date: '2024-02-25', time: '2:30 PM', doctor: 'Dr. Michael Roberts' },
    { id: 3, type: 'General', date: '2024-03-05', time: '11:15 AM', doctor: 'Dr. Emily Johnson' },
  ];

  const currentMedications = [
    { name: 'Lisinopril', dosage: '10mg', frequency: 'Daily', adherence: 98, streak: 28 },
    { name: 'Metformin', dosage: '500mg', frequency: 'Twice daily', adherence: 95, streak: 25 },
    { name: 'Vitamin D3', dosage: '1000 IU', frequency: 'Daily', adherence: 92, streak: 21 },
  ];

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Appointments Timeline */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Calendar className="w-5 h-5 text-blue-600" />
              Appointment History
            </CardTitle>
            <CardDescription>
              Scheduled vs completed appointments over time
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={appointmentData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="month" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="scheduled" fill="#3b82f6" name="Scheduled" />
                <Bar dataKey="completed" fill="#10b981" name="Completed" />
                <Bar dataKey="noShow" fill="#ef4444" name="No-show" />
              </BarChart>
            </ResponsiveContainer>
            
            <div className="mt-4 grid grid-cols-3 gap-4 text-center">
              <div>
                <div className="text-lg font-bold text-blue-600">14</div>
                <div className="text-xs text-gray-500">Total Scheduled</div>
              </div>
              <div>
                <div className="text-lg font-bold text-green-600">13</div>
                <div className="text-xs text-gray-500">Completed</div>
              </div>
              <div>
                <div className="text-lg font-bold text-gray-600">23 days</div>
                <div className="text-xs text-gray-500">Avg Time Between</div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Medication Adherence */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Pill className="w-5 h-5 text-green-600" />
              Medication Adherence
            </CardTitle>
            <CardDescription>
              Daily medication compliance rate
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={medicationAdherence}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="date" 
                  tickFormatter={(value) => new Date(value).toLocaleDateString('en-US', { weekday: 'short' })}
                />
                <YAxis domain={[80, 100]} />
                <Tooltip 
                  labelFormatter={(value) => new Date(value).toLocaleDateString()}
                  formatter={(value) => [`${value}%`, 'Adherence']}
                />
                <Line 
                  type="monotone" 
                  dataKey="adherence" 
                  stroke="#10b981" 
                  strokeWidth={3}
                  dot={{ fill: '#10b981', strokeWidth: 2, r: 4 }}
                />
              </LineChart>
            </ResponsiveContainer>
            
            <div className="mt-4 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Target className="w-4 h-4 text-green-600" />
                <span className="text-sm font-medium">Current Streak: 7 days</span>
              </div>
              <div className="text-sm text-gray-600">
                Avg: 95.6%
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Upcoming Appointments */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Clock className="w-5 h-5 text-purple-600" />
            Upcoming Appointments
          </CardTitle>
          <CardDescription>
            Your scheduled medical appointments
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {upcomingAppointments.map((appointment) => (
              <div key={appointment.id} className="flex items-center justify-between p-3 border rounded-lg">
                <div className="flex items-center gap-3">
                  <div className="p-2 bg-blue-100 rounded-full">
                    <Calendar className="w-4 h-4 text-blue-600" />
                  </div>
                  <div>
                    <div className="font-medium">{appointment.type}</div>
                    <div className="text-sm text-gray-600">{appointment.doctor}</div>
                  </div>
                </div>
                <div className="text-right">
                  <div className="font-medium">{appointment.date}</div>
                  <div className="text-sm text-gray-600">{appointment.time}</div>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Current Medications */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Pill className="w-5 h-5 text-green-600" />
            Current Medications
          </CardTitle>
          <CardDescription>
            Medication adherence tracking and streak counters
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {currentMedications.map((med, index) => (
              <div key={index} className="flex items-center justify-between p-4 border rounded-lg">
                <div className="flex items-center gap-3">
                  <div className="p-2 bg-green-100 rounded-full">
                    <Pill className="w-4 h-4 text-green-600" />
                  </div>
                  <div>
                    <div className="font-medium">{med.name}</div>
                    <div className="text-sm text-gray-600">{med.dosage} • {med.frequency}</div>
                  </div>
                </div>
                <div className="text-right">
                  <div className="flex items-center gap-2">
                    <span className="text-lg font-bold text-green-600">{med.adherence}%</span>
                    <TrendingUp className="w-4 h-4 text-green-600" />
                  </div>
                  <div className="text-sm text-gray-600">{med.streak} day streak</div>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
