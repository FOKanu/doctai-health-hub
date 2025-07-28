import React from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '@/contexts/AuthContext';
import { UserRole } from '@/contexts/AuthContext';
import {
  Home,
  Users,
  Calendar,
  Pill,
  FileText,
  Settings,
  Activity,
  Stethoscope,
  Brain,
  Shield,
  MessageSquare,
  Terminal,
  Server,
  Database,
  Code,
  Bug,
  Monitor,
  Zap
} from 'lucide-react';

interface RoleBasedMobileNavigationProps {
  role: UserRole;
}

export function RoleBasedMobileNavigation({ role }: RoleBasedMobileNavigationProps) {
  const navigate = useNavigate();
  const location = useLocation();

  const getNavigationItems = () => {
    switch (role) {
      case 'provider':
        return [
          { icon: Home, label: 'Dashboard', path: '/provider/dashboard' },
          { icon: Users, label: 'Patients', path: '/provider/patients' },
          { icon: Stethoscope, label: 'Clinical', path: '/provider/clinical' },
          { icon: Brain, label: 'AI Support', path: '/provider/ai-support' },
          { icon: Shield, label: 'Compliance', path: '/provider/compliance' },
          { icon: Settings, label: 'Settings', path: '/provider/settings' }
        ];
      case 'engineer':
        return [
          { icon: Home, label: 'Dashboard', path: '/engineer/dashboard' },
          { icon: Code, label: 'Dev Tools', path: '/engineer/dev-tools' },
          { icon: Database, label: 'Data', path: '/engineer/data' },
          { icon: Shield, label: 'Security', path: '/engineer/security' },
          { icon: Terminal, label: 'Logs', path: '/engineer/logs' },
          { icon: Settings, label: 'Settings', path: '/engineer/settings' }
        ];
      case 'patient':
      default:
        return [
          { icon: Home, label: 'Home', path: '/' },
          { icon: Calendar, label: 'Appointments', path: '/appointments' },
          { icon: Pill, label: 'Medications', path: '/medications' },
          { icon: FileText, label: 'Records', path: '/medical-records' },
          { icon: Activity, label: 'Analytics', path: '/analytics' },
          { icon: Settings, label: 'Settings', path: '/settings' }
        ];
    }
  };

  const navigationItems = getNavigationItems();

  return (
    <nav className="md:hidden fixed bottom-0 left-0 right-0 bg-white border-t border-gray-200 z-50 safe-area-pb">
      <div className="flex justify-around py-2">
        {navigationItems.map((item, index) => {
          const isActive = location.pathname === item.path;
          return (
            <button
              key={index}
              onClick={() => navigate(item.path)}
              className={`flex flex-col items-center py-2 px-3 rounded-lg transition-colors duration-200 ${
                isActive
                  ? role === 'engineer'
                    ? 'text-blue-400 bg-blue-900/20'
                    : 'text-blue-600 bg-blue-50'
                  : 'text-gray-500 hover:text-gray-700'
              }`}
            >
              <item.icon className="w-5 h-5 mb-1" />
              <span className="text-xs font-medium">{item.label}</span>
            </button>
          );
        })}
      </div>
    </nav>
  );
}
