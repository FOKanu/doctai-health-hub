
import React from 'react';
import { AlertCircle, Bell } from 'lucide-react';

interface Alert {
  icon: React.ComponentType<any>;
  text: string;
  type: string;
  time: string;
}

export const HealthAlertsSection: React.FC = () => {
  const alerts: Alert[] = [
    { icon: AlertCircle, text: 'Weekly rescan reminder for lesion #3', type: 'warning', time: '2 hours ago' },
    { icon: Bell, text: 'Prescription renewal due in 3 days', type: 'info', time: '1 day ago' },
  ];

  if (alerts.length === 0) return null;

  return (
    <div className="space-y-4">
      <h2 className="text-xl font-semibold text-gray-900">Health Alerts</h2>
      <div className="space-y-3">
        {alerts.map((alert, index) => (
          <div key={index} className={`p-4 rounded-lg border-l-4 ${
            alert.type === 'warning' ? 'bg-orange-50 border-orange-400' : 'bg-blue-50 border-blue-400'
          }`}>
            <div className="flex items-start space-x-3">
              <alert.icon className={`w-5 h-5 mt-0.5 ${
                alert.type === 'warning' ? 'text-orange-500' : 'text-blue-500'
              }`} />
              <div className="flex-1">
                <p className="text-sm font-medium text-gray-900">{alert.text}</p>
                <p className="text-xs text-gray-500 mt-1">{alert.time}</p>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
