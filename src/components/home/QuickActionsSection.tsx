
import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Camera, Upload } from 'lucide-react';

interface QuickAction {
  icon: React.ComponentType<any>;
  title: string;
  subtitle: string;
  color: string;
  path: string;
}

export const QuickActionsSection: React.FC = () => {
  const navigate = useNavigate();

  const quickActions: QuickAction[] = [
    { icon: Camera, title: 'Skin Scan', subtitle: 'AI-powered lesion detection', color: 'bg-blue-600 hover:bg-blue-700', path: '/scan' },
    { icon: Upload, title: 'Upload Medical Image', subtitle: 'CT, MRI, EEG analysis', color: 'bg-green-600 hover:bg-green-700', path: '/upload' },
  ];

  return (
    <div className="space-y-4">
      <h2 className="text-xl font-semibold text-gray-900">Core Health Actions</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {quickActions.map((action, index) => (
          <button
            key={index}
            onClick={() => navigate(action.path)}
            className={`${action.color} text-white p-6 rounded-xl shadow-lg hover:shadow-xl transform hover:scale-105 transition-all duration-200 text-left`}
          >
            <action.icon className="w-10 h-10 mb-4" />
            <h3 className="font-semibold text-lg mb-2">{action.title}</h3>
            <p className="text-sm opacity-90">{action.subtitle}</p>
          </button>
        ))}
      </div>
    </div>
  );
};
