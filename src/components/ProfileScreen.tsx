
import React from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowLeft, User, Settings, Bell, Shield, HelpCircle, LogOut, Edit, Camera } from 'lucide-react';

const ProfileScreen = () => {
  const navigate = useNavigate();

  const userInfo = {
    name: 'John Doe',
    email: 'john.doe@email.com',
    phone: '+49 123 456 7890',
    dateOfBirth: '1985-03-15',
    insuranceProvider: 'TK - Techniker Krankenkasse',
    emergencyContact: 'Jane Doe (+49 987 654 3210)',
    avatar: 'https://images.unsplash.com/photo-1472099645785-5658abf4ff4e?w=150&h=150&fit=crop&crop=face'
  };

  const menuItems = [
    { icon: User, label: 'Personal Information', action: () => {} },
    { icon: Bell, label: 'Notifications', action: () => {} },
    { icon: Shield, label: 'Privacy & Security', action: () => {} },
    { icon: Settings, label: 'App Settings', action: () => {} },
    { icon: HelpCircle, label: 'Help & Support', action: () => {} },
  ];

  return (
    <div className="min-h-screen bg-gray-50 pb-20">
      {/* Header */}
      <div className="bg-white shadow-sm">
        <div className="flex items-center p-4">
          <button
            onClick={() => navigate('/')}
            className="p-2 -ml-2 rounded-full hover:bg-gray-100"
          >
            <ArrowLeft className="w-6 h-6" />
          </button>
          <h1 className="text-xl font-semibold ml-2">Profile</h1>
        </div>
      </div>

      <div className="p-4">
        {/* User Profile Card */}
        <div className="bg-white rounded-lg shadow-sm p-6 mb-6">
          <div className="flex items-center space-x-4 mb-4">
            <div className="relative">
              <img
                src={userInfo.avatar}
                alt="Profile"
                className="w-20 h-20 rounded-full object-cover"
              />
              <button className="absolute bottom-0 right-0 p-1 bg-blue-600 text-white rounded-full hover:bg-blue-700 transition-colors">
                <Camera className="w-3 h-3" />
              </button>
            </div>
            
            <div className="flex-1">
              <h2 className="text-xl font-semibold text-gray-800">{userInfo.name}</h2>
              <p className="text-gray-600">{userInfo.email}</p>
              <p className="text-sm text-gray-500">{userInfo.phone}</p>
            </div>
            
            <button className="p-2 text-blue-600 hover:bg-blue-50 rounded-full">
              <Edit className="w-5 h-5" />
            </button>
          </div>

          <div className="grid grid-cols-1 gap-3 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-500">Date of Birth:</span>
              <span className="font-medium">{userInfo.dateOfBirth}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">Insurance:</span>
              <span className="font-medium">{userInfo.insuranceProvider}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">Emergency Contact:</span>
              <span className="font-medium">{userInfo.emergencyContact}</span>
            </div>
          </div>
        </div>

        {/* Health Stats */}
        <div className="bg-white rounded-lg shadow-sm p-4 mb-6">
          <h3 className="font-semibold text-gray-800 mb-3">Health Overview</h3>
          <div className="grid grid-cols-2 gap-4">
            <div className="text-center p-3 bg-blue-50 rounded-lg">
              <div className="text-2xl font-bold text-blue-600 mb-1">23</div>
              <div className="text-xs text-gray-500">Total Scans</div>
            </div>
            <div className="text-center p-3 bg-green-50 rounded-lg">
              <div className="text-2xl font-bold text-green-600 mb-1">5</div>
              <div className="text-xs text-gray-500">Active Medications</div>
            </div>
            <div className="text-center p-3 bg-purple-50 rounded-lg">
              <div className="text-2xl font-bold text-purple-600 mb-1">2</div>
              <div className="text-xs text-gray-500">Upcoming Appointments</div>
            </div>
            <div className="text-center p-3 bg-orange-50 rounded-lg">
              <div className="text-2xl font-bold text-orange-600 mb-1">1</div>
              <div className="text-xs text-gray-500">Active Reminders</div>
            </div>
          </div>
        </div>

        {/* Menu Items */}
        <div className="bg-white rounded-lg shadow-sm mb-6">
          {menuItems.map((item, index) => (
            <button
              key={index}
              onClick={item.action}
              className="w-full flex items-center justify-between p-4 border-b border-gray-100 last:border-b-0 hover:bg-gray-50 transition-colors"
            >
              <div className="flex items-center space-x-3">
                <item.icon className="w-5 h-5 text-gray-600" />
                <span className="font-medium text-gray-800">{item.label}</span>
              </div>
              <ArrowLeft className="w-4 h-4 text-gray-400 rotate-180" />
            </button>
          ))}
        </div>

        {/* App Information */}
        <div className="bg-white rounded-lg shadow-sm p-4 mb-6">
          <h3 className="font-semibold text-gray-800 mb-3">App Information</h3>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-500">Version:</span>
              <span className="font-medium">1.0.0</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">Language:</span>
              <span className="font-medium">English</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">Data Storage:</span>
              <span className="font-medium">Local + Cloud</span>
            </div>
          </div>
        </div>

        {/* Sign Out Button */}
        <button className="w-full bg-red-50 text-red-600 py-4 rounded-lg font-semibold hover:bg-red-100 transition-colors flex items-center justify-center">
          <LogOut className="w-5 h-5 mr-2" />
          Sign Out
        </button>
      </div>
    </div>
  );
};

export default ProfileScreen;
