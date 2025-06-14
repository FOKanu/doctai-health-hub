import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Camera, Upload, TrendingUp, Activity, AlertCircle, Bell, Calendar, BarChart, Apple, Brain, Trophy, Target, Star, Zap, Award, Heart, Weight, Droplets, Moon, Thermometer, Plus } from 'lucide-react';

const HomeScreen = () => {
  const navigate = useNavigate();

  const quickActions = [
    { icon: Camera, title: 'Skin Scan', subtitle: 'AI-powered lesion detection', color: 'bg-blue-600 hover:bg-blue-700', path: '/scan' },
    { icon: Upload, title: 'Upload Medical Image', subtitle: 'CT, MRI, EEG analysis', color: 'bg-green-600 hover:bg-green-700', path: '/upload' },
  ];

  const healthManagementActions = [
    { icon: BarChart, title: 'Results', subtitle: 'View your test results', color: 'bg-purple-600 hover:bg-purple-700', path: '/results' },
    { icon: Activity, title: 'Fitness Metrics', subtitle: 'Track your health metrics', color: 'bg-orange-600 hover:bg-orange-700', path: '/fitness' },
    { icon: Apple, title: 'Diet Plan', subtitle: 'Personalized nutrition', color: 'bg-pink-600 hover:bg-pink-700', path: '/diet' },
    { icon: Brain, title: 'AI Recommendations', subtitle: 'Smart health insights', color: 'bg-indigo-600 hover:bg-indigo-700', path: '/recommendations' },
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

  // Gamification Elements
  const achievements = [
    { icon: Trophy, title: 'Health Warrior', description: 'Completed 30 health checks', unlocked: true },
    { icon: Target, title: 'Consistency Champion', description: 'Logged health data for 7 days straight', unlocked: true },
    { icon: Star, title: 'Early Bird', description: 'Complete morning health routine', unlocked: false },
    { icon: Award, title: 'Wellness Expert', description: 'Achieve 90% health score', unlocked: false },
  ];

  const healthScore = 78;
  const weeklyGoals = [
    { title: 'Daily Steps', current: 8500, target: 10000, icon: Activity, unit: 'steps' },
    { title: 'Water Intake', current: 6, target: 8, icon: Droplets, unit: 'glasses' },
    { title: 'Sleep Hours', current: 7.2, target: 8, icon: Moon, unit: 'hours' },
    { title: 'Heart Rate', current: 72, target: 65, icon: Heart, unit: 'bpm' },
    { title: 'Weight Goal', current: 75.2, target: 73.0, icon: Weight, unit: 'kg' },
    { title: 'Body Temp', current: 98.6, target: 98.6, icon: Thermometer, unit: '°F' },
  ];

  return (
    <div className="space-y-6">
      {/* Welcome Section with Health Score */}
      <div className="bg-gradient-to-r from-blue-600 to-blue-700 rounded-xl p-6 text-white">
        <div className="flex justify-between items-start">
          <div>
            <h1 className="text-2xl md:text-3xl font-bold mb-2">Welcome back!</h1>
            <p className="text-blue-100 mb-4">Your AI-powered health monitoring is active</p>
            <div className="flex items-center space-x-2 text-sm">
              <TrendingUp className="w-4 h-4" />
              <span>Health score improving by 12% this month</span>
            </div>
          </div>
          <div className="text-center">
            <div className="w-20 h-20 bg-white/20 rounded-full flex items-center justify-center mb-2">
              <span className="text-2xl font-bold">{healthScore}</span>
            </div>
            <p className="text-sm text-blue-200">Health Score</p>
          </div>
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

      {/* Weekly Goals Progress */}
      <div className="bg-white rounded-lg p-6 shadow-sm border border-gray-200">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold text-gray-900">Weekly Health Goals</h2>
          <button className="flex items-center space-x-2 text-blue-600 hover:text-blue-700 text-sm font-medium">
            <Plus className="w-4 h-4" />
            <span>Add Metric</span>
          </button>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {weeklyGoals.map((goal, index) => {
            const progress = (goal.current / goal.target) * 100;
            const isOnTrack = progress >= 80;
            return (
              <div key={index} className="bg-gray-50 rounded-lg p-4 space-y-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <div className={`p-2 rounded-lg ${isOnTrack ? 'bg-green-100' : 'bg-orange-100'}`}>
                      <goal.icon className={`w-4 h-4 ${isOnTrack ? 'text-green-600' : 'text-orange-600'}`} />
                    </div>
                    <span className="text-sm font-medium text-gray-900">{goal.title}</span>
                  </div>
                  <span className="text-xs text-gray-500 bg-white px-2 py-1 rounded">
                    {goal.current} / {goal.target} {goal.unit}
                  </span>
                </div>
                <div className="space-y-2">
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div 
                      className={`h-2 rounded-full transition-all duration-300 ${
                        isOnTrack ? 'bg-green-500' : 'bg-orange-500'
                      }`}
                      style={{ width: `${Math.min(progress, 100)}%` }}
                    ></div>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span className={`font-medium ${isOnTrack ? 'text-green-600' : 'text-orange-600'}`}>
                      {Math.round(progress)}% complete
                    </span>
                    <span className="text-gray-500">
                      {isOnTrack ? 'On track' : 'Needs attention'}
                    </span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Achievements */}
      <div className="bg-white rounded-lg p-6 shadow-sm border border-gray-200">
        <h2 className="text-xl font-semibold text-gray-900 mb-4">Recent Achievements</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {achievements.map((achievement, index) => (
            <div key={index} className={`p-4 rounded-lg border-2 ${
              achievement.unlocked 
                ? 'border-yellow-300 bg-yellow-50' 
                : 'border-gray-200 bg-gray-50'
            }`}>
              <div className="flex items-center space-x-3">
                <achievement.icon className={`w-8 h-8 ${
                  achievement.unlocked ? 'text-yellow-600' : 'text-gray-400'
                }`} />
                <div>
                  <h3 className={`font-semibold text-sm ${
                    achievement.unlocked ? 'text-gray-900' : 'text-gray-500'
                  }`}>
                    {achievement.title}
                  </h3>
                  <p className={`text-xs ${
                    achievement.unlocked ? 'text-gray-600' : 'text-gray-400'
                  }`}>
                    {achievement.description}
                  </p>
                </div>
              </div>
            </div>
          ))}
        </div>
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

      {/* Health Management */}
      <div className="space-y-4">
        <h2 className="text-xl font-semibold text-gray-900">Health Management</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {healthManagementActions.map((action, index) => (
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
