
import React from 'react';
import { useNavigate } from 'react-router-dom';
import { BarChart, Activity, Apple, Brain } from 'lucide-react';

interface HealthAction {
  icon: React.ComponentType<{ className?: string }>;
  title: string;
  subtitle: string;
  color: string;
  path: string;
}

export const HealthManagementSection: React.FC = () => {
  const navigate = useNavigate();

  const healthManagementActions: HealthAction[] = [
    { icon: BarChart, title: 'Results', subtitle: 'View your test results', color: 'bg-primary hover:bg-primary/90', path: '/results' },
    { icon: Activity, title: 'Fitness', subtitle: 'Track your workouts & fitness', color: 'bg-secondary hover:bg-secondary/90', path: '/fitness' },
    { icon: Apple, title: 'Diet Plan', subtitle: 'Personalized nutrition', color: 'bg-accent hover:bg-accent/90', path: '/diet' },
    { icon: Brain, title: 'AI Recommendations', subtitle: 'Smart health insights', color: 'bg-primary hover:bg-primary/90', path: '/analytics' },
  ];

  return (
    <div className="space-y-4">
      <h2 className="text-xl font-semibold text-foreground">Health Management</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {healthManagementActions.map((action, index) => (
          <button
            key={index}
            onClick={() => navigate(action.path)}
            className="bg-card p-6 rounded-lg shadow-sm border border-border hover:shadow-md hover:border-border/70 transition-all duration-200 text-left group"
          >
            <div className={`w-12 h-12 ${action.color} rounded-lg flex items-center justify-center mb-4 group-hover:scale-110 transition-transform duration-200`}>
              <action.icon className="w-6 h-6 text-primary-foreground" />
            </div>
            <h3 className="font-semibold text-foreground mb-2">{action.title}</h3>
            <p className="text-sm text-muted-foreground">{action.subtitle}</p>
          </button>
        ))}
      </div>
    </div>
  );
};
