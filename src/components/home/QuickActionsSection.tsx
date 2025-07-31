
import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Camera, Upload, TrendingUp } from 'lucide-react';
import { BodyPartSelectionDialog, BodyPart } from '../BodyPartSelectionDialog';

interface QuickAction {
  icon: React.ComponentType<{ className?: string }>;
  title: string;
  subtitle: string;
  color: string;
  path: string;
}

export const QuickActionsSection: React.FC = () => {
  const navigate = useNavigate();
  const [showBodyPartDialog, setShowBodyPartDialog] = useState(false);
  const [selectedBodyPart, setSelectedBodyPart] = useState<BodyPart | null>(null);

  const quickActions: QuickAction[] = [
    { icon: Camera, title: 'Skin Scan', subtitle: 'AI-powered lesion detection', color: 'bg-primary hover:bg-primary/90', path: '/patient/scan' },
    { icon: Upload, title: 'Upload Medical Image', subtitle: 'CT, MRI, EEG analysis', color: 'bg-secondary hover:bg-secondary/90', path: '/patient/upload' },
    { icon: TrendingUp, title: 'Track Progression', subtitle: 'Time-series health analysis', color: 'bg-accent hover:bg-accent/90', path: '/patient/analytics?tab=progression' },
  ];

  const handleActionClick = (action: QuickAction) => {
    if (action.path === '/scan') {
      setShowBodyPartDialog(true);
    } else {
      navigate(action.path);
    }
  };

  const handleBodyPartSelect = (bodyPart: BodyPart) => {
    setSelectedBodyPart(bodyPart);
    setShowBodyPartDialog(false);

    // Navigate to scan screen with body part data
    navigate('/patient/scan', {
      state: {
        scanMetaData: { bodyPart }
      }
    });
  };

  return (
    <>
      <div className="space-y-4">
        <h2 className="text-xl font-semibold text-foreground">Core Health Actions</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {quickActions.map((action, index) => (
            <button
              key={index}
              onClick={() => handleActionClick(action)}
              className={`${action.color} text-primary-foreground p-6 rounded-xl shadow-lg hover:shadow-xl transform hover:scale-105 transition-all duration-200 text-left`}
            >
              <action.icon className="w-10 h-10 mb-4" />
              <h3 className="font-semibold text-lg mb-2">{action.title}</h3>
              <p className="text-sm opacity-90">{action.subtitle}</p>
            </button>
          ))}
        </div>
      </div>

      <BodyPartSelectionDialog
        open={showBodyPartDialog}
        onOpenChange={setShowBodyPartDialog}
        onBodyPartSelect={handleBodyPartSelect}
        selectedBodyPart={selectedBodyPart}
      />
    </>
  );
};
