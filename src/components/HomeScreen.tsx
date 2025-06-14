
import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Camera, Upload, User, Calendar, Pill, FileText, AlertCircle, Bell } from 'lucide-react';

const HomeScreen = () => {
  const navigate = useNavigate();

  const mainActions = [
    { icon: Camera, title: 'Start New Scan', subtitle: 'Check skin lesions', color: 'bg-blue-500', path: '/scan' },
    { icon: Upload, title: 'Upload Medical Image', subtitle: 'CT, MRI, EEG analysis', color: 'bg-green-500', path: '/upload' },
    { icon: User, title: 'AI Specialist Recommendation', subtitle: 'Get expert guidance', color: 'bg-purple-500', path: '/specialists' },
    { icon: Calendar, title: 'Book Appointment', subtitle: 'Find German doctors', color: 'bg-orange-500', path: '/appointments' },
    { icon: Pill, title: 'Manage Medications', subtitle: 'Track prescriptions', color: 'bg-pink-500', path: '/medications' },
    { icon: FileText, title: 'View Treatment Plan', subtitle: 'Your health journey', color: 'bg-indigo-500', path: '/treatments' },
  ];

  const alerts = [
    { icon: AlertCircle, text: 'Weekly rescan reminder for lesion #3', type: 'warning' },
    { icon: Bell, text: 'Prescription renewal due in 3 days', type: 'info' },
  ];

  return (
    <div className="p-4 pb-20">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-800 mb-2">Welcome to DoctAI</h1>
        <p className="text-gray-600">Your AI-powered health companion</p>
      </div>

      {/* Alerts Section */}
      {alerts.length > 0 && (
        <div className="mb-6">
          <h2 className="text-lg font-semibold text-gray-800 mb-3">Alerts</h2>
          {alerts.map((alert, index) => (
            <div key={index} className={`p-3 rounded-lg mb-2 flex items-center ${
              alert.type === 'warning' ? 'bg-orange-50 border-l-4 border-orange-400' : 'bg-blue-50 border-l-4 border-blue-400'
            }`}>
              <alert.icon className={`w-5 h-5 mr-3 ${
                alert.type === 'warning' ? 'text-orange-500' : 'text-blue-500'
              }`} />
              <span className="text-sm text-gray-700">{alert.text}</span>
            </div>
          ))}
        </div>
      )}

      {/* Main Actions Grid */}
      <div className="space-y-4">
        <h2 className="text-lg font-semibold text-gray-800">Quick Actions</h2>
        <div className="grid grid-cols-2 gap-4">
          {mainActions.map((action, index) => (
            <button
              key={index}
              onClick={() => navigate(action.path)}
              className={`${action.color} text-white p-4 rounded-xl shadow-lg hover:shadow-xl transform hover:scale-105 transition-all duration-200`}
            >
              <action.icon className="w-8 h-8 mb-2" />
              <h3 className="font-semibold text-sm mb-1">{action.title}</h3>
              <p className="text-xs opacity-90">{action.subtitle}</p>
            </button>
          ))}
        </div>
      </div>

      {/* Recent Activity */}
      <div className="mt-6">
        <h2 className="text-lg font-semibold text-gray-800 mb-3">Recent Activity</h2>
        <div className="bg-gray-50 rounded-lg p-4">
          <div className="flex items-center text-sm text-gray-600">
            <Calendar className="w-4 h-4 mr-2" />
            <span>Last scan: 3 days ago - Low risk detected</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default HomeScreen;
