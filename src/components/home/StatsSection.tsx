
import React from 'react';
import { Camera, Activity, Calendar } from 'lucide-react';

interface Stat {
  label: string;
  value: string;
  change: string;
  icon: React.ComponentType<any>;
  color: string;
}

export const StatsSection: React.FC = () => {
  const stats: Stat[] = [
    { label: 'Total Scans', value: '24', change: '+3 this week', icon: Camera, color: 'text-blue-600' },
    { label: 'Risk Assessments', value: '18', change: '2 high priority', icon: Activity, color: 'text-green-600' },
    { label: 'Appointments', value: '3', change: 'Next: Tomorrow', icon: Calendar, color: 'text-purple-600' },
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
      {stats.map((stat, index) => (
        <div key={index} className="bg-white rounded-lg p-6 shadow-sm border border-gray-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">{stat.label}</p>
              <p className="text-2xl font-bold text-gray-900 mt-1">{stat.value}</p>
              <p className="text-xs text-gray-500 mt-1">{stat.change}</p>
            </div>
            <stat.icon className={`w-8 h-8 ${stat.color}`} />
          </div>
        </div>
      ))}
    </div>
  );
};
