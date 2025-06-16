import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Switch } from '@/components/ui/switch';
import { Smartphone, Watch, Wifi, WifiOff, RefreshCw, CheckCircle2, AlertCircle, Clock, Activity } from 'lucide-react';

export const FitnessDataSync: React.FC = () => {
  const [syncSettings, setSyncSettings] = useState({
    phoneSteps: true,
    phoneHeartRate: false,
    appleHealth: false,
    googleFit: true,
    smartwatch: false,
    autoSync: true
  });

  const dataSources = [
    {
      id: 'phone',
      name: 'Phone Sensors',
      description: 'Steps, movement, and basic health metrics',
      icon: Smartphone,
      connected: true,
      lastSync: '2 minutes ago',
      data: ['Steps: 8,547', 'Distance: 4.2 km', 'Floors: 12'],
      status: 'active'
    },
    {
      id: 'appleHealth',
      name: 'Apple Health',
      description: 'Comprehensive health data from iOS devices',
      icon: Activity,
      connected: syncSettings.appleHealth,
      lastSync: syncSettings.appleHealth ? '15 minutes ago' : 'Never',
      data: ['Heart Rate', 'Workouts', 'Sleep', 'Nutrition'],
      status: syncSettings.appleHealth ? 'active' : 'inactive'
    },
    {
      id: 'googleFit',
      name: 'Google Fit',
      description: 'Activity and wellness data from Google services',
      icon: Activity,
      connected: syncSettings.googleFit,
      lastSync: syncSettings.googleFit ? '5 minutes ago' : 'Never',
      data: ['Activities', 'Goals', 'Heart Points', 'Move Minutes'],
      status: syncSettings.googleFit ? 'active' : 'inactive'
    },
    {
      id: 'smartwatch',
      name: 'Smart Watch',
      description: 'Real-time fitness tracking from wearable devices',
      icon: Watch,
      connected: syncSettings.smartwatch,
      lastSync: syncSettings.smartwatch ? '1 minute ago' : 'Never',
      data: ['Real-time HR', 'Workout Sessions', 'Sleep Tracking'],
      status: syncSettings.smartwatch ? 'active' : 'inactive'
    }
  ];

  const recentSyncActivity = [
    {
      source: 'Phone Sensors',
      action: 'Steps data synced',
      time: '2 min ago',
      status: 'success'
    },
    {
      source: 'Google Fit',
      action: 'Workout completed',
      time: '1 hour ago',
      status: 'success'
    },
    {
      source: 'Smart Watch',
      action: 'Connection failed',
      time: '2 hours ago',
      status: 'error'
    },
    {
      source: 'Apple Health',
      action: 'Sleep data imported',
      time: '8 hours ago',
      status: 'success'
    }
  ];

  const handleToggleSync = (sourceId: string) => {
    setSyncSettings(prev => ({
      ...prev,
      [sourceId]: !prev[sourceId]
    }));
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'active': return <CheckCircle2 className="w-4 h-4 text-green-500" />;
      case 'inactive': return <WifiOff className="w-4 h-4 text-gray-400" />;
      case 'error': return <AlertCircle className="w-4 h-4 text-red-500" />;
      default: return <Wifi className="w-4 h-4 text-blue-500" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'bg-green-100 text-green-800';
      case 'inactive': return 'bg-gray-100 text-gray-800';
      case 'error': return 'bg-red-100 text-red-800';
      default: return 'bg-blue-100 text-blue-800';
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-gray-900">Data Sync & Integration</h2>
          <p className="text-gray-600">Connect your devices and apps to automatically track fitness data</p>
        </div>
        <Button className="bg-blue-600 hover:bg-blue-700">
          <RefreshCw className="w-4 h-4 mr-2" />
          Sync All
        </Button>
      </div>

      {/* Auto Sync Setting */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-lg">Automatic Sync</CardTitle>
              <CardDescription>
                Enable automatic data synchronization every 15 minutes
              </CardDescription>
            </div>
            <Switch
              checked={syncSettings.autoSync}
              onCheckedChange={(checked) => setSyncSettings(prev => ({ ...prev, autoSync: checked }))}
            />
          </div>
        </CardHeader>
      </Card>

      {/* Data Sources */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {dataSources.map((source) => {
          const IconComponent = source.icon;
          return (
            <Card key={source.id} className="hover:shadow-md transition-shadow">
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <div className="p-2 bg-blue-100 rounded-lg">
                      <IconComponent className="w-5 h-5 text-blue-600" />
                    </div>
                    <div>
                      <CardTitle className="text-lg">{source.name}</CardTitle>
                      <CardDescription>{source.description}</CardDescription>
                    </div>
                  </div>
                  <div className="flex items-center space-x-2">
                    {getStatusIcon(source.status)}
                    <Badge className={getStatusColor(source.status)}>
                      {source.status}
                    </Badge>
                  </div>
                </div>
              </CardHeader>
              <CardContent className="space-y-4">
                {/* Connection Toggle */}
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">Enable Sync</span>
                  <Switch
                    checked={source.connected}
                    onCheckedChange={() => handleToggleSync(source.id)}
                    disabled={source.id === 'phone'} // Phone sensors always enabled
                  />
                </div>

                {/* Last Sync */}
                <div className="flex items-center space-x-2 text-sm text-gray-600">
                  <Clock className="w-4 h-4" />
                  <span>Last sync: {source.lastSync}</span>
                </div>

                {/* Available Data */}
                <div>
                  <h4 className="text-sm font-medium text-gray-700 mb-2">Available Data:</h4>
                  <div className="flex flex-wrap gap-1">
                    {source.data.map((item, index) => (
                      <Badge key={index} variant="outline" className="text-xs">
                        {item}
                      </Badge>
                    ))}
                  </div>
                </div>

                {/* Action Button */}
                {source.id !== 'phone' && (
                  <Button 
                    variant={source.connected ? "outline" : "default"} 
                    size="sm" 
                    className="w-full"
                  >
                    {source.connected ? 'Disconnect' : 'Connect'}
                  </Button>
                )}
              </CardContent>
            </Card>
          );
        })}
      </div>

      {/* Sync Activity */}
      <Card>
        <CardHeader>
          <CardTitle>Recent Sync Activity</CardTitle>
          <CardDescription>
            Latest data synchronization events and status updates
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {recentSyncActivity.map((activity, index) => (
              <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <div className="flex items-center space-x-3">
                  {activity.status === 'success' ? (
                    <CheckCircle2 className="w-4 h-4 text-green-500" />
                  ) : (
                    <AlertCircle className="w-4 h-4 text-red-500" />
                  )}
                  <div>
                    <div className="font-medium text-sm">{activity.action}</div>
                    <div className="text-xs text-gray-600">{activity.source}</div>
                  </div>
                </div>
                <div className="text-xs text-gray-500">{activity.time}</div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Privacy Notice */}
      <Card className="bg-blue-50 border-blue-200">
        <CardContent className="p-4">
          <div className="flex items-start space-x-3">
            <AlertCircle className="w-5 h-5 text-blue-600 mt-0.5" />
            <div className="text-sm text-blue-800">
              <p className="font-medium mb-1">Privacy & Security</p>
              <p>All synced data is encrypted and stored securely. You can disconnect any source at any time. We never share your personal health data with third parties.</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
