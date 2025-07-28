
import React from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowLeft, User, Settings, Bell, Shield, HelpCircle, LogOut, Edit, Camera } from 'lucide-react';
import { useAuth } from '@/contexts/AuthContext';

const ProfileScreen = () => {
  const navigate = useNavigate();
  const { logout: authLogout, user: authUser } = useAuth();

  const userInfo = {
    name: authUser?.name || 'John Doe',
    email: authUser?.email || 'john.doe@email.com',
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

  const handleLogout = () => {
    // Use the auth context logout method
    authLogout();

    // Navigate to login page
    navigate('/login');
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-4xl mx-auto px-4 py-4">
          <div className="flex items-center space-x-4">
            <button
              onClick={() => navigate(-1)}
              className="p-2 rounded-lg hover:bg-gray-100 transition-colors"
            >
              <ArrowLeft className="w-5 h-5 text-gray-600" />
            </button>
            <h1 className="text-xl font-semibold text-gray-900">Profile</h1>
          </div>
        </div>
      </div>

      <div className="max-w-4xl mx-auto px-4 py-6">
        {/* Profile Header */}
        <div className="bg-white rounded-lg shadow-sm p-6 mb-6">
          <div className="flex items-center space-x-6">
            <div className="relative">
              <img
                src={userInfo.avatar}
                alt={userInfo.name}
                className="w-20 h-20 rounded-full object-cover"
              />
              <button className="absolute bottom-0 right-0 p-1 bg-blue-600 text-white rounded-full hover:bg-blue-700 transition-colors">
                <Camera className="w-4 h-4" />
              </button>
            </div>
            <div className="flex-1">
              <div className="flex items-center space-x-3 mb-2">
                <h2 className="text-2xl font-bold text-gray-900">{userInfo.name}</h2>
                <button className="p-1 text-gray-400 hover:text-gray-600">
                  <Edit className="w-4 h-4" />
                </button>
              </div>
              <p className="text-gray-600">{userInfo.email}</p>
            </div>
          </div>
        </div>

        {/* Personal Information */}
        <div className="bg-white rounded-lg shadow-sm p-6 mb-6">
          <h3 className="font-semibold text-gray-800 mb-4">Personal Information</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Phone</label>
              <p className="text-gray-900">{userInfo.phone}</p>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Date of Birth</label>
              <p className="text-gray-900">{userInfo.dateOfBirth}</p>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Insurance Provider</label>
              <p className="text-gray-900">{userInfo.insuranceProvider}</p>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Emergency Contact</label>
              <p className="text-gray-900">{userInfo.emergencyContact}</p>
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
        <button
          onClick={handleLogout}
          className="w-full bg-red-50 text-red-600 py-4 rounded-lg font-semibold hover:bg-red-100 transition-colors flex items-center justify-center"
        >
          <LogOut className="w-5 h-5 mr-2" />
          Sign Out
        </button>
      </div>
    </div>
  );
};

export default ProfileScreen;
