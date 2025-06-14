
import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Camera, Upload, User, Calendar, Pill, FileText, AlertCircle, Bell, TrendingUp, Activity } from 'lucide-react';

const HomeScreen = () => {
  const navigate = useNavigate();

  const quickActions = [
    { icon: Camera, title: 'Skin Scan', subtitle: 'AI-powered lesion detection', color: 'bg-blue-600 hover:bg-blue-700', path: '/scan' },
    { icon: Upload, title: 'Upload Medical Image', subtitle: 'CT, MRI, EEG analysis', color: 'bg-green-600 hover:bg-green-700', path: '/upload' },
  ];

  const secondaryActions = [
    { icon: User, title: 'Find Specialists', subtitle: 'Get AI recommendations', color: 'bg-purple-600 hover:bg-purple-700', path: '/specialists' },
    { icon: Calendar, title: 'Book Appointment', subtitle: 'German healthcare providers', color: 'bg-orange-600 hover:bg-orange-700', path: '/appointments' },
    { icon: Pill, title: 'Manage Medications', subtitle: 'Track & renew prescriptions', color: 'bg-pink-600 hover:bg-pink-700', path: '/medications' },
    { icon: FileText, title: 'Treatment Plans', subtitle: 'Your health journey', color: 'bg-indigo-600 hover:bg-indigo-700', path: '/treatments' },
  ];

  const alerts = [
    { icon: AlertCircle, text: 'Weekly rescan reminder for lesion #3', type: 'warning', time: '2 hours ago' },
    { icon: Bell, text: 'Prescription renewal due in 3 days', type: 'info', time: '1 day ago' },
  ];

  const stats = [
    { label: 'Total Scans', value: '24', change: '+3 this week', icon: Camera, color: 'text-blue-600' },
    { label: 'Risk Assessments', value: '18', change: '2 high priority', icon: Activity, color: 'text-green-600' },
    { label: 'Appointments', value: '3', change: 'Next: Tomorrow', icon: Calendar, color: 'text-purple-600' },
  ];

  return (
    <div className="space-y-6">
      {/* Welcome Section */}
      <div className="bg-gradient-to-r from-blue-600 to-blue-700 rounded-xl p-6 text-white">
        <h1 className="text-2xl md:text-3xl font-bold mb-2">Welcome back!</h1>
        <p className="text-blue-100 mb-4">Your AI-powered health monitoring is active</p>
        <div className="flex items-center space-x-2 text-sm">
          <TrendingUp className="w-4 h-4" />
          <span>Health score improving by 12% this month</span>
        </div>
      </div>

      {/* Quick Stats */}
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

      {/* Core Actions */}
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

      {/* Alerts */}
      {alerts.length > 0 && (
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
      )}

      {/* Additional Services */}
      <div className="space-y-4">
        <h2 className="text-xl font-semibold text-gray-900">Health Management</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {secondaryActions.map((action, index) => (
            <button
              key={index}
              onClick={() => navigate(action.path)}
              className="bg-white p-6 rounded-lg shadow-sm border border-gray-200 hover:shadow-md hover:border-gray-300 transition-all duration-200 text-left group"
            >
              <div className={`w-12 h-12 ${action.color} rounded-lg flex items-center justify-center mb-4 group-hover:scale-110 transition-transform duration-200`}>
                <action.icon className="w-6 h-6 text-white" />
              </div>
              <h3 className="font-semibold text-gray-900 mb-2">{action.title}</h3>
              <p className="text-sm text-gray-600">{action.subtitle}</p>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};

export default HomeScreen;
