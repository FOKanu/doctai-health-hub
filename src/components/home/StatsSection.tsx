
import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Camera, Activity, Calendar } from 'lucide-react';

interface Stat {
  label: string;
  value: string;
  change: string;
  icon: React.ComponentType<{ className?: string }>;
  color: string;
  path: string;
}

export const StatsSection: React.FC = () => {
  const navigate = useNavigate();

  const stats: Stat[] = [
    {
      label: 'Total Scans',
      value: '24',
      change: '+3 this week',
      icon: Camera,
      color: 'text-primary',
      path: '/total-scans'
    },
    {
      label: 'Risk Assessments',
      value: '18',
      change: '2 high priority',
      icon: Activity,
      color: 'text-secondary-foreground',
      path: '/risk-assessments'
    },
    {
      label: 'Appointments',
      value: '3',
      change: 'Next: Tomorrow',
      icon: Calendar,
      color: 'text-accent-foreground',
      path: '/appointments'
    },
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
      {stats.map((stat, index) => (
        <button
          key={index}
          onClick={() => navigate(stat.path)}
          className="bg-card rounded-lg p-6 shadow-sm border border-border hover:shadow-md hover:border-border/70 transition-all duration-200 text-left group"
        >
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-muted-foreground">{stat.label}</p>
              <p className="text-2xl font-bold text-foreground mt-1">{stat.value}</p>
              <p className="text-xs text-muted-foreground mt-1">{stat.change}</p>
            </div>
            <stat.icon className={`w-8 h-8 ${stat.color} group-hover:scale-110 transition-transform duration-200`} />
          </div>
        </button>
      ))}
    </div>
  );
};
