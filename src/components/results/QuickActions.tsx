
import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { 
  Camera, 
  RefreshCw, 
  Target, 
  Brain, 
  BarChart3,
  Plus,
  Calendar,
  Settings
} from 'lucide-react';

interface QuickActionsProps {
  onNavigate: (path: string) => void;
}

export const QuickActions: React.FC<QuickActionsProps> = ({ onNavigate }) => {
  const actions = [
    {
      title: 'Rescan',
      description: 'Take a new skin health scan',
      icon: Camera,
      action: () => onNavigate('/scan'),
      variant: 'default' as const,
      className: 'bg-blue-600 hover:bg-blue-700'
    },
    {
      title: 'Update Activity',
      description: 'Log your latest workout',
      icon: Plus,
      action: () => onNavigate('/fitness'),
      variant: 'outline' as const
    },
    {
      title: 'Adjust Goals',
      description: 'Modify your health targets',
      icon: Target,
      action: () => onNavigate('/settings'),
      variant: 'outline' as const
    },
    {
      title: 'View Analytics',
      description: 'Detailed health insights',
      icon: BarChart3,
      action: () => onNavigate('/analytics'),
      variant: 'outline' as const
    },
    {
      title: 'AI Recommendations',
      description: 'Get personalized advice',
      icon: Brain,
      action: () => onNavigate('/analytics'),
      variant: 'outline' as const
    },
    {
      title: 'Book Appointment',
      description: 'Schedule with specialist',
      icon: Calendar,
      action: () => onNavigate('/appointments'),
      variant: 'outline' as const
    }
  ];

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <RefreshCw className="w-5 h-5" />
          Quick Actions
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
          {actions.map((action, index) => (
            <Button
              key={index}
              variant={action.variant}
              onClick={action.action}
              className={`h-auto p-4 flex flex-col items-center space-y-2 text-center ${action.className || ''}`}
            >
              <action.icon className="w-6 h-6" />
              <div>
                <div className="font-semibold text-sm">{action.title}</div>
                <div className="text-xs opacity-80 hidden md:block">{action.description}</div>
              </div>
            </Button>
          ))}
        </div>
      </CardContent>
    </Card>
  );
};
