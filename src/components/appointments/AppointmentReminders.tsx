
import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Bell, Lightbulb, AlertCircle } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';

export const AppointmentReminders = () => {
  const reminders = [
    {
      id: 1,
      type: 'upcoming',
      title: 'Medical Checkup Tomorrow',
      message: 'Don\'t forget your appointment with Dr. Smith at 2:30 PM',
      time: '24 hours',
      priority: 'high'
    },
    {
      id: 2,
      type: 'suggestion',
      title: 'Schedule Dental Checkup',
      message: 'It\'s been 6 months since your last dental visit',
      priority: 'medium'
    },
    {
      id: 3,
      type: 'routine',
      title: 'Weekly Gym Session',
      message: 'Time for your regular workout routine',
      time: '2 hours',
      priority: 'low'
    }
  ];

  const getPriorityColor = (priority: string) => {
    const colors = {
      high: 'bg-red-100 text-red-800 border-red-200',
      medium: 'bg-yellow-100 text-yellow-800 border-yellow-200',
      low: 'bg-blue-100 text-blue-800 border-blue-200'
    };
    return colors[priority as keyof typeof colors] || 'bg-gray-100 text-gray-800';
  };

  const getIcon = (type: string) => {
    switch (type) {
      case 'upcoming':
        return <Bell className="w-4 h-4" />;
      case 'suggestion':
        return <Lightbulb className="w-4 h-4" />;
      case 'routine':
        return <AlertCircle className="w-4 h-4" />;
      default:
        return <Bell className="w-4 h-4" />;
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Bell className="w-5 h-5" />
          Smart Reminders & Recommendations
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-3">
          {reminders.map((reminder) => (
            <div
              key={reminder.id}
              className={`border rounded-lg p-3 ${getPriorityColor(reminder.priority)}`}
            >
              <div className="flex items-start justify-between">
                <div className="flex items-start gap-3">
                  {getIcon(reminder.type)}
                  <div className="flex-1">
                    <h4 className="font-medium">{reminder.title}</h4>
                    <p className="text-sm opacity-90 mt-1">{reminder.message}</p>
                    {reminder.time && (
                      <p className="text-xs opacity-75 mt-1">in {reminder.time}</p>
                    )}
                  </div>
                </div>
                <div className="flex gap-2">
                  <Badge variant="outline" className="text-xs">
                    {reminder.priority}
                  </Badge>
                  <Button size="sm" variant="ghost">
                    Dismiss
                  </Button>
                </div>
              </div>
            </div>
          ))}
        </div>
        
        <div className="mt-4 p-3 bg-blue-50 rounded-lg">
          <h4 className="font-medium text-blue-900 mb-2">Notification Settings</h4>
          <div className="space-y-2 text-sm">
            <label className="flex items-center gap-2">
              <input type="checkbox" defaultChecked className="rounded" />
              <span>SMS Reminders</span>
            </label>
            <label className="flex items-center gap-2">
              <input type="checkbox" defaultChecked className="rounded" />
              <span>Email Notifications</span>
            </label>
            <label className="flex items-center gap-2">
              <input type="checkbox" defaultChecked className="rounded" />
              <span>Push Notifications</span>
            </label>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};
