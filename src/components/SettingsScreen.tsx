import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowLeft, Bell, Shield, Globe, Moon, Smartphone, Database, HelpCircle, Info, ChevronRight } from 'lucide-react';

const SettingsScreen = () => {
  const navigate = useNavigate();
  const [notifications, setNotifications] = useState({
    appointments: true,
    medications: true,
    labResults: true,
    reminders: false
  });
  const [privacy, setPrivacy] = useState({
    dataSharing: false,
    analytics: true,
    biometrics: true
  });

  const settingSections = [
    {
      title: 'Notifications',
      icon: Bell,
      items: [
        {
          label: 'Appointment Reminders',
          description: 'Get notified about upcoming appointments',
          type: 'toggle',
          value: notifications.appointments,
          onChange: (value: boolean) => setNotifications({...notifications, appointments: value})
        },
        {
          label: 'Medication Reminders',
          description: 'Reminders to take your medications',
          type: 'toggle',
          value: notifications.medications,
          onChange: (value: boolean) => setNotifications({...notifications, medications: value})
        },
        {
          label: 'Lab Results',
          description: 'Notifications when new results are available',
          type: 'toggle',
          value: notifications.labResults,
          onChange: (value: boolean) => setNotifications({...notifications, labResults: value})
        },
        {
          label: 'Health Reminders',
          description: 'General health and wellness reminders',
          type: 'toggle',
          value: notifications.reminders,
          onChange: (value: boolean) => setNotifications({...notifications, reminders: value})
        }
      ]
    },
    {
      title: 'Privacy & Security',
      icon: Shield,
      items: [
        {
          label: 'Data Sharing',
          description: 'Share anonymized data for research',
          type: 'toggle',
          value: privacy.dataSharing,
          onChange: (value: boolean) => setPrivacy({...privacy, dataSharing: value})
        },
        {
          label: 'Usage Analytics',
          description: 'Help improve the app with usage data',
          type: 'toggle',
          value: privacy.analytics,
          onChange: (value: boolean) => setPrivacy({...privacy, analytics: value})
        },
        {
          label: 'Biometric Authentication',
          description: 'Use fingerprint or face recognition',
          type: 'toggle',
          value: privacy.biometrics,
          onChange: (value: boolean) => setPrivacy({...privacy, biometrics: value})
        },
        {
          label: 'Change Password',
          description: 'Update your account password',
          type: 'action',
          action: () => console.log('Change password')
        },
        {
          label: 'Two-Factor Authentication',
          description: 'Add an extra layer of security',
          type: 'action',
          action: () => console.log('Setup 2FA')
        }
      ]
    },
    {
      title: 'Preferences',
      icon: Globe,
      items: [
        {
          label: 'Language',
          description: 'English (US)',
          type: 'action',
          action: () => console.log('Change language')
        },
        {
          label: 'Date Format',
          description: 'MM/DD/YYYY',
          type: 'action',
          action: () => console.log('Change date format')
        },
        {
          label: 'Time Zone',
          description: 'Eastern Time (ET)',
          type: 'action',
          action: () => console.log('Change timezone')
        },
        {
          label: 'Units',
          description: 'Imperial (lbs, ft, °F)',
          type: 'action',
          action: () => console.log('Change units')
        }
      ]
    },
    {
      title: 'Data & Storage',
      icon: Database,
      items: [
        {
          label: 'Sync Settings',
          description: 'Manage cloud synchronization',
          type: 'action',
          action: () => console.log('Sync settings')
        },
        {
          label: 'Export Data',
          description: 'Download your health data',
          type: 'action',
          action: () => console.log('Export data')
        },
        {
          label: 'Storage Usage',
          description: '2.3 GB of 5 GB used',
          type: 'action',
          action: () => console.log('Storage usage')
        },
        {
          label: 'Clear Cache',
          description: 'Free up space by clearing cached data',
          type: 'action',
          action: () => console.log('Clear cache')
        }
      ]
    },
    {
      title: 'Support',
      icon: HelpCircle,
      items: [
        {
          label: 'Help Center',
          description: 'Get answers to common questions',
          type: 'action',
          action: () => console.log('Help center')
        },
        {
          label: 'Contact Support',
          description: 'Get help from our support team',
          type: 'action',
          action: () => console.log('Contact support')
        },
        {
          label: 'Report a Bug',
          description: 'Help us improve the app',
          type: 'action',
          action: () => console.log('Report bug')
        },
        {
          label: 'Privacy Policy',
          description: 'Review our privacy policy',
          type: 'action',
          action: () => console.log('Privacy policy')
        },
        {
          label: 'Terms of Service',
          description: 'Review terms and conditions',
          type: 'action',
          action: () => console.log('Terms of service')
        }
      ]
    },
    {
      title: 'About',
      icon: Info,
      items: [
        {
          label: 'App Version',
          description: '1.0.0 (Build 24)',
          type: 'info'
        },
        {
          label: 'Last Updated',
          description: 'January 15, 2024',
          type: 'info'
        },
        {
          label: 'Device ID',
          description: 'DV-2024-001',
          type: 'info'
        }
      ]
    }
  ];

  const ToggleSwitch = ({ enabled, onChange }: { enabled: boolean; onChange: (value: boolean) => void }) => (
    <button
      onClick={() => onChange(!enabled)}
      className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
        enabled ? 'bg-blue-600' : 'bg-gray-300'
      }`}
    >
      <span
        className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
          enabled ? 'translate-x-6' : 'translate-x-1'
        }`}
      />
    </button>
  );

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
          <h1 className="text-xl font-semibold ml-2">Settings</h1>
        </div>
      </div>

      <div className="p-4">
        {settingSections.map((section, sectionIndex) => {
          const SectionIcon = section.icon;
          return (
            <div key={sectionIndex} className="mb-6">
              <div className="flex items-center space-x-2 mb-3">
                <SectionIcon className="w-5 h-5 text-gray-600" />
                <h2 className="text-lg font-semibold text-gray-800">{section.title}</h2>
              </div>
              
              <div className="bg-white rounded-lg shadow-sm">
                {section.items.map((item, itemIndex) => (
                  <div
                    key={itemIndex}
                    className={`p-4 ${itemIndex < section.items.length - 1 ? 'border-b border-gray-100' : ''}`}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex-1">
                        <h3 className="font-medium text-gray-800">{item.label}</h3>
                        <p className="text-sm text-gray-500 mt-1">{item.description}</p>
                      </div>
                      
                      <div className="ml-4">
                        {item.type === 'toggle' && (
                          <ToggleSwitch
                            enabled={item.value as boolean}
                            onChange={item.onChange as (value: boolean) => void}
                          />
                        )}
                        {item.type === 'action' && (
                          <button
                            onClick={item.action}
                            className="p-1 text-gray-400 hover:text-gray-600 rounded"
                          >
                            <ChevronRight className="w-5 h-5" />
                          </button>
                        )}
                        {item.type === 'info' && (
                          <div className="text-right">
                            <span className="text-sm text-gray-500">{item.description}</span>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          );
        })}

        {/* App Info */}
        <div className="text-center text-gray-500 text-sm mt-8">
          <p>DoctAI Health Assistant</p>
          <p>Made with ❤️ for your health</p>
        </div>
      </div>
    </div>
  );
};

export default SettingsScreen;
