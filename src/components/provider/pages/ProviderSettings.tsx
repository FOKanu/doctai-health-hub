import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { Textarea } from '@/components/ui/textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Separator } from '@/components/ui/separator';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  Settings, 
  User, 
  Bell, 
  Link, 
  Save, 
  Globe,
  Mail,
  MessageSquare,
  Phone,
  CheckCircle,
  AlertTriangle,
  Zap
} from 'lucide-react';

interface ProviderSettingsData {
  // Profile
  displayName: string;
  email: string;
  phone: string;
  specialty: string;
  bio: string;
  licenseNumber: string;
  
  // Status
  isOnline: boolean;
  
  // Notifications
  emailNotifications: boolean;
  smsNotifications: boolean;
  pushNotifications: boolean;
  appointmentReminders: boolean;
  patientMessages: boolean;
  emergencyAlerts: boolean;
  
  // Integrations
  calendars: {
    google: boolean;
    outlook: boolean;
    apple: boolean;
  };
  emr: {
    epic: boolean;
    cerner: boolean;
    allscripts: boolean;
  };
  lab: {
    quest: boolean;
    labcorp: boolean;
  };
}

const defaultSettings: ProviderSettingsData = {
  displayName: 'Dr. Sarah Weber',
  email: 'sarah.weber@doctai.com',
  phone: '+49 30 12345678',
  specialty: 'Dermatology',
  bio: 'Board-certified dermatologist with 10+ years of experience in medical and cosmetic dermatology.',
  licenseNumber: 'DE-DERM-2013-001',
  isOnline: true,
  emailNotifications: true,
  smsNotifications: false,
  pushNotifications: true,
  appointmentReminders: true,
  patientMessages: true,
  emergencyAlerts: true,
  calendars: {
    google: true,
    outlook: false,
    apple: false
  },
  emr: {
    epic: false,
    cerner: true,
    allscripts: false
  },
  lab: {
    quest: true,
    labcorp: false
  }
};

// Mock localStorage functions
const getSettings = (): ProviderSettingsData => {
  const stored = localStorage.getItem('providerSettings');
  return stored ? JSON.parse(stored) : defaultSettings;
};

const saveSettings = (settings: ProviderSettingsData) => {
  localStorage.setItem('providerSettings', JSON.stringify(settings));
  // Dispatch custom event for global status updates
  window.dispatchEvent(new CustomEvent('providerStatusChange', { 
    detail: { isOnline: settings.isOnline } 
  }));
};

export function ProviderSettings() {
  const [settings, setSettings] = useState<ProviderSettingsData>(defaultSettings);
  const [isSaving, setIsSaving] = useState(false);
  const [savedTab, setSavedTab] = useState<string | null>(null);

  useEffect(() => {
    const loadedSettings = getSettings();
    setSettings(loadedSettings);
  }, []);

  const handleSave = async (tabName: string) => {
    setIsSaving(true);
    setSavedTab(tabName);
    
    try {
      saveSettings(settings);
      await new Promise(resolve => setTimeout(resolve, 500)); // Mock save delay
    } finally {
      setIsSaving(false);
      setTimeout(() => setSavedTab(null), 2000);
    }
  };

  const updateSettings = (updates: Partial<ProviderSettingsData>) => {
    setSettings(prev => ({ ...prev, ...updates }));
  };

  const updateNestedSettings = (section: keyof ProviderSettingsData, field: string, value: boolean) => {
    setSettings(prev => ({
      ...prev,
      [section]: {
        ...prev[section as keyof ProviderSettingsData] as object,
        [field]: value
      }
    }));
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Settings</h1>
          <p className="text-gray-600 mt-1">Manage your profile and preferences</p>
        </div>
        <div className="flex items-center space-x-2">
          <Globe className={`w-4 h-4 ${settings.isOnline ? 'text-green-600' : 'text-gray-400'}`} />
          <Badge variant={settings.isOnline ? 'default' : 'secondary'}>
            {settings.isOnline ? 'Online' : 'Offline'}
          </Badge>
        </div>
      </div>

      <Tabs defaultValue="profile" className="space-y-6">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="profile" className="flex items-center space-x-2">
            <User className="w-4 h-4" />
            <span>Profile</span>
          </TabsTrigger>
          <TabsTrigger value="notifications" className="flex items-center space-x-2">
            <Bell className="w-4 h-4" />
            <span>Notifications</span>
          </TabsTrigger>
          <TabsTrigger value="integrations" className="flex items-center space-x-2">
            <Link className="w-4 h-4" />
            <span>Integrations</span>
          </TabsTrigger>
        </TabsList>

        {/* Profile Tab */}
        <TabsContent value="profile">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <User className="w-5 h-5" />
                  <span>Profile Information</span>
                </div>
                {savedTab === 'profile' && (
                  <Badge className="bg-green-100 text-green-800">
                    <CheckCircle className="w-3 h-3 mr-1" />
                    Saved
                  </Badge>
                )}
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Online Status */}
              <div className="flex items-center justify-between p-4 bg-blue-50 border border-blue-200 rounded-lg">
                <div className="flex items-center space-x-3">
                  <Globe className={`w-5 h-5 ${settings.isOnline ? 'text-green-600' : 'text-gray-400'}`} />
                  <div>
                    <h4 className="font-medium">Online Status</h4>
                    <p className="text-sm text-gray-600">
                      {settings.isOnline ? 'Available for consultations' : 'Not available'}
                    </p>
                  </div>
                </div>
                <Switch
                  checked={settings.isOnline}
                  onCheckedChange={(checked) => updateSettings({ isOnline: checked })}
                />
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="displayName">Display Name</Label>
                  <Input
                    id="displayName"
                    value={settings.displayName}
                    onChange={(e) => updateSettings({ displayName: e.target.value })}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="email">Email Address</Label>
                  <Input
                    id="email"
                    type="email"
                    value={settings.email}
                    onChange={(e) => updateSettings({ email: e.target.value })}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="phone">Phone Number</Label>
                  <Input
                    id="phone"
                    value={settings.phone}
                    onChange={(e) => updateSettings({ phone: e.target.value })}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="specialty">Specialty</Label>
                  <Select value={settings.specialty} onValueChange={(value) => updateSettings({ specialty: value })}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="Dermatology">Dermatology</SelectItem>
                      <SelectItem value="Cardiology">Cardiology</SelectItem>
                      <SelectItem value="Neurology">Neurology</SelectItem>
                      <SelectItem value="Orthopedics">Orthopedics</SelectItem>
                      <SelectItem value="Ophthalmology">Ophthalmology</SelectItem>
                      <SelectItem value="General Practice">General Practice</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2 md:col-span-2">
                  <Label htmlFor="licenseNumber">Medical License Number</Label>
                  <Input
                    id="licenseNumber"
                    value={settings.licenseNumber}
                    onChange={(e) => updateSettings({ licenseNumber: e.target.value })}
                  />
                </div>
              </div>

              <div className="space-y-2">
                <Label htmlFor="bio">Professional Bio</Label>
                <Textarea
                  id="bio"
                  value={settings.bio}
                  onChange={(e) => updateSettings({ bio: e.target.value })}
                  rows={4}
                  placeholder="Brief description of your background and expertise..."
                />
              </div>

              <Button 
                onClick={() => handleSave('profile')} 
                disabled={isSaving}
                className="w-full"
              >
                <Save className="w-4 h-4 mr-2" />
                {isSaving ? 'Saving...' : 'Save Profile'}
              </Button>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Notifications Tab */}
        <TabsContent value="notifications">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <Bell className="w-5 h-5" />
                  <span>Notification Preferences</span>
                </div>
                {savedTab === 'notifications' && (
                  <Badge className="bg-green-100 text-green-800">
                    <CheckCircle className="w-3 h-3 mr-1" />
                    Saved
                  </Badge>
                )}
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <Alert>
                <AlertTriangle className="h-4 w-4" />
                <AlertDescription>
                  These preferences control how and when you receive notifications. Emergency alerts cannot be disabled.
                </AlertDescription>
              </Alert>

              <div className="space-y-4">
                <h4 className="font-medium flex items-center space-x-2">
                  <MessageSquare className="w-4 h-4" />
                  <span>Notification Channels</span>
                </h4>
                
                <div className="space-y-4 pl-6">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      <Mail className="w-4 h-4 text-blue-600" />
                      <div>
                        <p className="font-medium">Email Notifications</p>
                        <p className="text-sm text-gray-600">Receive notifications via email</p>
                      </div>
                    </div>
                    <Switch
                      checked={settings.emailNotifications}
                      onCheckedChange={(checked) => updateSettings({ emailNotifications: checked })}
                    />
                  </div>

                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      <Phone className="w-4 h-4 text-green-600" />
                      <div>
                        <p className="font-medium">SMS Notifications</p>
                        <p className="text-sm text-gray-600">Receive text messages for urgent items</p>
                      </div>
                    </div>
                    <Switch
                      checked={settings.smsNotifications}
                      onCheckedChange={(checked) => updateSettings({ smsNotifications: checked })}
                    />
                  </div>

                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      <Bell className="w-4 h-4 text-purple-600" />
                      <div>
                        <p className="font-medium">Push Notifications</p>
                        <p className="text-sm text-gray-600">Browser and app push notifications</p>
                      </div>
                    </div>
                    <Switch
                      checked={settings.pushNotifications}
                      onCheckedChange={(checked) => updateSettings({ pushNotifications: checked })}
                    />
                  </div>
                </div>
              </div>

              <Separator />

              <div className="space-y-4">
                <h4 className="font-medium">Notification Types</h4>
                
                <div className="space-y-4 pl-6">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="font-medium">Appointment Reminders</p>
                      <p className="text-sm text-gray-600">Reminders about upcoming appointments</p>
                    </div>
                    <Switch
                      checked={settings.appointmentReminders}
                      onCheckedChange={(checked) => updateSettings({ appointmentReminders: checked })}
                    />
                  </div>

                  <div className="flex items-center justify-between">
                    <div>
                      <p className="font-medium">Patient Messages</p>
                      <p className="text-sm text-gray-600">New messages from patients</p>
                    </div>
                    <Switch
                      checked={settings.patientMessages}
                      onCheckedChange={(checked) => updateSettings({ patientMessages: checked })}
                    />
                  </div>

                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <p className="font-medium">Emergency Alerts</p>
                      <Badge variant="destructive" className="text-xs">Required</Badge>
                    </div>
                    <Switch
                      checked={settings.emergencyAlerts}
                      disabled
                    />
                  </div>
                </div>
              </div>

              <Button 
                onClick={() => handleSave('notifications')} 
                disabled={isSaving}
                className="w-full"
              >
                <Save className="w-4 h-4 mr-2" />
                {isSaving ? 'Saving...' : 'Save Notification Preferences'}
              </Button>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Integrations Tab */}
        <TabsContent value="integrations">
          <div className="space-y-6">
            {/* Calendar Integrations */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <Link className="w-5 h-5" />
                    <span>Calendar Integrations</span>
                  </div>
                  {savedTab === 'integrations' && (
                    <Badge className="bg-green-100 text-green-800">
                      <CheckCircle className="w-3 h-3 mr-1" />
                      Saved
                    </Badge>
                  )}
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between p-3 border rounded-lg">
                  <div className="flex items-center space-x-3">
                    <div className="w-8 h-8 bg-blue-100 rounded flex items-center justify-center">
                      <span className="text-xs font-bold text-blue-600">G</span>
                    </div>
                    <div>
                      <p className="font-medium">Google Calendar</p>
                      <p className="text-sm text-gray-600">Sync appointments with Google Calendar</p>
                    </div>
                  </div>
                  <Switch
                    checked={settings.calendars.google}
                    onCheckedChange={(checked) => updateNestedSettings('calendars', 'google', checked)}
                  />
                </div>

                <div className="flex items-center justify-between p-3 border rounded-lg">
                  <div className="flex items-center space-x-3">
                    <div className="w-8 h-8 bg-blue-100 rounded flex items-center justify-center">
                      <span className="text-xs font-bold text-blue-600">O</span>
                    </div>
                    <div>
                      <p className="font-medium">Microsoft Outlook</p>
                      <p className="text-sm text-gray-600">Sync with Outlook calendar</p>
                    </div>
                  </div>
                  <Switch
                    checked={settings.calendars.outlook}
                    onCheckedChange={(checked) => updateNestedSettings('calendars', 'outlook', checked)}
                  />
                </div>

                <div className="flex items-center justify-between p-3 border rounded-lg">
                  <div className="flex items-center space-x-3">
                    <div className="w-8 h-8 bg-gray-100 rounded flex items-center justify-center">
                      <span className="text-xs font-bold text-gray-600">A</span>
                    </div>
                    <div>
                      <p className="font-medium">Apple Calendar</p>
                      <p className="text-sm text-gray-600">Sync with Apple Calendar (iCloud)</p>
                    </div>
                  </div>
                  <Switch
                    checked={settings.calendars.apple}
                    onCheckedChange={(checked) => updateNestedSettings('calendars', 'apple', checked)}
                  />
                </div>
              </CardContent>
            </Card>

            {/* EMR Integrations */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Zap className="w-5 h-5" />
                  <span>Electronic Medical Records</span>
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between p-3 border rounded-lg">
                  <div className="flex items-center space-x-3">
                    <div className="w-8 h-8 bg-purple-100 rounded flex items-center justify-center">
                      <span className="text-xs font-bold text-purple-600">E</span>
                    </div>
                    <div>
                      <p className="font-medium">Epic</p>
                      <p className="text-sm text-gray-600">Connect to Epic EMR system</p>
                    </div>
                  </div>
                  <Switch
                    checked={settings.emr.epic}
                    onCheckedChange={(checked) => updateNestedSettings('emr', 'epic', checked)}
                  />
                </div>

                <div className="flex items-center justify-between p-3 border rounded-lg">
                  <div className="flex items-center space-x-3">
                    <div className="w-8 h-8 bg-green-100 rounded flex items-center justify-center">
                      <span className="text-xs font-bold text-green-600">C</span>
                    </div>
                    <div>
                      <p className="font-medium">Cerner</p>
                      <p className="text-sm text-gray-600">Connect to Cerner EMR system</p>
                    </div>
                  </div>
                  <Switch
                    checked={settings.emr.cerner}
                    onCheckedChange={(checked) => updateNestedSettings('emr', 'cerner', checked)}
                  />
                </div>

                <div className="flex items-center justify-between p-3 border rounded-lg">
                  <div className="flex items-center space-x-3">
                    <div className="w-8 h-8 bg-orange-100 rounded flex items-center justify-center">
                      <span className="text-xs font-bold text-orange-600">A</span>
                    </div>
                    <div>
                      <p className="font-medium">Allscripts</p>
                      <p className="text-sm text-gray-600">Connect to Allscripts EMR system</p>
                    </div>
                  </div>
                  <Switch
                    checked={settings.emr.allscripts}
                    onCheckedChange={(checked) => updateNestedSettings('emr', 'allscripts', checked)}
                  />
                </div>
              </CardContent>
            </Card>

            {/* Lab Integrations */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Settings className="w-5 h-5" />
                  <span>Laboratory Systems</span>
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between p-3 border rounded-lg">
                  <div className="flex items-center space-x-3">
                    <div className="w-8 h-8 bg-red-100 rounded flex items-center justify-center">
                      <span className="text-xs font-bold text-red-600">Q</span>
                    </div>
                    <div>
                      <p className="font-medium">Quest Diagnostics</p>
                      <p className="text-sm text-gray-600">Receive lab results from Quest</p>
                    </div>
                  </div>
                  <Switch
                    checked={settings.lab.quest}
                    onCheckedChange={(checked) => updateNestedSettings('lab', 'quest', checked)}
                  />
                </div>

                <div className="flex items-center justify-between p-3 border rounded-lg">
                  <div className="flex items-center space-x-3">
                    <div className="w-8 h-8 bg-blue-100 rounded flex items-center justify-center">
                      <span className="text-xs font-bold text-blue-600">L</span>
                    </div>
                    <div>
                      <p className="font-medium">LabCorp</p>
                      <p className="text-sm text-gray-600">Receive lab results from LabCorp</p>
                    </div>
                  </div>
                  <Switch
                    checked={settings.lab.labcorp}
                    onCheckedChange={(checked) => updateNestedSettings('lab', 'labcorp', checked)}
                  />
                </div>
              </CardContent>
            </Card>

            <Button 
              onClick={() => handleSave('integrations')} 
              disabled={isSaving}
              className="w-full"
            >
              <Save className="w-4 h-4 mr-2" />
              {isSaving ? 'Saving...' : 'Save Integration Settings'}
            </Button>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}