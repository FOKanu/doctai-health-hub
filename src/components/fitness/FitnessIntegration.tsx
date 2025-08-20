import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Alert, AlertDescription } from '@/components/ui/alert';
import {
  Watch,
  Heart,
  Footprints,
  Moon,
  Flame,
  Scale,
  RefreshCw,
  CheckCircle,
  XCircle,
  ExternalLink,
  TrendingUp,
  TrendingDown,
  Minus
} from 'lucide-react';
import { FitnessIntegrationService, FitnessDevice, HealthMetricsSummary } from '@/services/fitness/fitnessIntegrationService';

interface FitnessIntegrationProps {
  userId: string;
  className?: string;
}

export const FitnessIntegration: React.FC<FitnessIntegrationProps> = ({ userId, className }) => {
  const [fitnessService] = useState(() => new FitnessIntegrationService());
  const [connectedDevices, setConnectedDevices] = useState<FitnessDevice[]>([]);
  const [healthSummary, setHealthSummary] = useState<HealthMetricsSummary | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [syncResults, setSyncResults] = useState<any[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadConnectedDevices();
    loadHealthSummary();
  }, [userId]);

  const loadConnectedDevices = () => {
    const devices = fitnessService.getConnectedDevices(userId);
    setConnectedDevices(devices);
  };

  const loadHealthSummary = async () => {
    try {
      const summary = await fitnessService.getHealthMetricsSummary(userId);
      setHealthSummary(summary);
    } catch (error) {
      console.error('Error loading health summary:', error);
    }
  };

  const handleConnectGoogleFit = async () => {
    try {
      setIsLoading(true);
      setError(null);

      // Initialize Google Fit with your app's credentials
      const config = {
        clientId: import.meta.env.VITE_GOOGLE_FIT_CLIENT_ID || '',
        clientSecret: import.meta.env.VITE_GOOGLE_FIT_CLIENT_SECRET || '',
        redirectUri: `${window.location.origin}/fitness/callback`,
        scopes: [
          'https://www.googleapis.com/auth/fitness.activity.read',
          'https://www.googleapis.com/auth/fitness.heart_rate.read',
          'https://www.googleapis.com/auth/fitness.sleep.read',
          'https://www.googleapis.com/auth/fitness.body.read',
        ],
      };

      const authUrl = await fitnessService.initializeGoogleFit(config);
      window.open(authUrl, '_blank', 'width=600,height=600');
    } catch (error) {
      setError('Failed to initialize Google Fit connection');
      console.error('Error connecting Google Fit:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleConnectFitbit = async () => {
    try {
      setIsLoading(true);
      setError(null);

      // Initialize Fitbit with your app's credentials
      const config = {
        clientId: import.meta.env.VITE_FITBIT_CLIENT_ID || '',
        clientSecret: import.meta.env.VITE_FITBIT_CLIENT_SECRET || '',
        redirectUri: `${window.location.origin}/fitness/callback`,
        scopes: [
          'activity',
          'heartrate',
          'sleep',
          'weight',
          'profile',
        ],
      };

      const authUrl = await fitnessService.initializeFitbit(config);
      window.open(authUrl, '_blank', 'width=600,height=600');
    } catch (error) {
      setError('Failed to initialize Fitbit connection');
      console.error('Error connecting Fitbit:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSyncAllDevices = async () => {
    try {
      setIsLoading(true);
      setError(null);

      const today = new Date().toISOString().split('T')[0];
      const weekAgo = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString().split('T')[0];

      const results = await fitnessService.syncAllDevices(userId, {
        start: weekAgo,
        end: today,
      });

      setSyncResults(results);
      loadConnectedDevices();
      loadHealthSummary();
    } catch (error) {
      setError('Failed to sync devices');
      console.error('Error syncing devices:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleDisconnectDevice = async (deviceId: string) => {
    try {
      await fitnessService.disconnectDevice(deviceId);
      loadConnectedDevices();
    } catch (error) {
      setError('Failed to disconnect device');
      console.error('Error disconnecting device:', error);
    }
  };

  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case 'increasing':
        return <TrendingUp className="w-4 h-4 text-red-500" />;
      case 'decreasing':
        return <TrendingDown className="w-4 h-4 text-green-500" />;
      default:
        return <Minus className="w-4 h-4 text-gray-500" />;
    }
  };

  const getTrendColor = (trend: string) => {
    switch (trend) {
      case 'increasing':
        return 'text-red-600';
      case 'decreasing':
        return 'text-green-600';
      default:
        return 'text-gray-600';
    }
  };

  return (
    <div className={className}>
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Watch className="h-5 w-5" />
            Smart Watch Integration
          </CardTitle>
          <CardDescription>
            Connect your Google Fit or Fitbit device to sync health metrics
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Connection Status */}
          <div className="space-y-4">
            <h3 className="font-medium">Connected Devices</h3>
            {connectedDevices.length === 0 ? (
              <Alert>
                <AlertDescription>
                  No devices connected. Connect your smartwatch to start syncing health data.
                </AlertDescription>
              </Alert>
            ) : (
              <div className="space-y-3">
                {connectedDevices.map((device) => (
                  <div key={device.id} className="flex items-center justify-between p-3 border rounded-lg">
                    <div className="flex items-center gap-3">
                      <div className="p-2 bg-blue-100 rounded-lg">
                        <Watch className="w-4 h-4 text-blue-600" />
                      </div>
                      <div>
                        <p className="font-medium">{device.name}</p>
                        <p className="text-sm text-gray-600">
                          Last sync: {device.lastSync ? new Date(device.lastSync).toLocaleString() : 'Never'}
                        </p>
                        <div className="flex gap-1 mt-1">
                          {device.metrics.map((metric) => (
                            <Badge key={metric} variant="secondary" className="text-xs">
                              {metric.replace('_', ' ')}
                            </Badge>
                          ))}
                        </div>
                      </div>
                    </div>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => handleDisconnectDevice(device.id)}
                    >
                      Disconnect
                    </Button>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Connect New Devices */}
          <div className="space-y-4">
            <h3 className="font-medium">Connect New Device</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <Button
                onClick={handleConnectGoogleFit}
                disabled={isLoading}
                className="flex items-center gap-2"
              >
                <ExternalLink className="w-4 h-4" />
                Connect Google Fit
              </Button>
              <Button
                onClick={handleConnectFitbit}
                disabled={isLoading}
                className="flex items-center gap-2"
              >
                <ExternalLink className="w-4 h-4" />
                Connect Fitbit
              </Button>
            </div>
          </div>

          {/* Sync Controls */}
          {connectedDevices.length > 0 && (
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="font-medium">Sync Data</h3>
                <Button
                  onClick={handleSyncAllDevices}
                  disabled={isLoading}
                  className="flex items-center gap-2"
                >
                  <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
                  {isLoading ? 'Syncing...' : 'Sync All Devices'}
                </Button>
              </div>

              {/* Sync Results */}
              {syncResults.length > 0 && (
                <div className="space-y-2">
                  {syncResults.map((result, index) => (
                    <div key={index} className="flex items-center gap-2 p-2 border rounded">
                      {result.success ? (
                        <CheckCircle className="w-4 h-4 text-green-500" />
                      ) : (
                        <XCircle className="w-4 h-4 text-red-500" />
                      )}
                      <span className="text-sm">
                        {result.deviceId}: {result.success ? `${result.metricsSynced} metrics synced` : result.errors.join(', ')}
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Health Metrics Summary */}
          {healthSummary && (
            <div className="space-y-4">
              <h3 className="font-medium">Health Metrics Summary</h3>
              <Tabs defaultValue="overview" className="w-full">
                <TabsList className="grid w-full grid-cols-4">
                  <TabsTrigger value="overview">Overview</TabsTrigger>
                  <TabsTrigger value="activity">Activity</TabsTrigger>
                  <TabsTrigger value="sleep">Sleep</TabsTrigger>
                  <TabsTrigger value="vitals">Vitals</TabsTrigger>
                </TabsList>

                <TabsContent value="overview" className="space-y-4">
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                    {/* Heart Rate */}
                    <Card>
                      <CardContent className="p-4">
                        <div className="flex items-center gap-2 mb-2">
                          <Heart className="w-4 h-4 text-red-500" />
                          <span className="text-sm font-medium">Heart Rate</span>
                          {getTrendIcon(healthSummary.heartRate.trend)}
                        </div>
                        <p className="text-2xl font-bold">{healthSummary.heartRate.current || 0} BPM</p>
                        <p className="text-xs text-gray-600">
                          Avg: {Math.round(healthSummary.heartRate.average || 0)} BPM
                        </p>
                      </CardContent>
                    </Card>

                    {/* Steps */}
                    <Card>
                      <CardContent className="p-4">
                        <div className="flex items-center gap-2 mb-2">
                          <Footprints className="w-4 h-4 text-blue-500" />
                          <span className="text-sm font-medium">Steps</span>
                        </div>
                        <p className="text-2xl font-bold">{healthSummary.steps.today.toLocaleString()}</p>
                        <div className="mt-2">
                          <Progress value={healthSummary.steps.progress} className="h-2" />
                          <p className="text-xs text-gray-600 mt-1">
                            {healthSummary.steps.progress.toFixed(0)}% of daily goal
                          </p>
                        </div>
                      </CardContent>
                    </Card>

                    {/* Sleep */}
                    <Card>
                      <CardContent className="p-4">
                        <div className="flex items-center gap-2 mb-2">
                          <Moon className="w-4 h-4 text-purple-500" />
                          <span className="text-sm font-medium">Sleep</span>
                        </div>
                        <p className="text-2xl font-bold">{healthSummary.sleep.lastNight.toFixed(1)}h</p>
                        <p className="text-xs text-gray-600">
                          Efficiency: {healthSummary.sleep.efficiency.toFixed(0)}%
                        </p>
                      </CardContent>
                    </Card>

                    {/* Calories */}
                    <Card>
                      <CardContent className="p-4">
                        <div className="flex items-center gap-2 mb-2">
                          <Flame className="w-4 h-4 text-orange-500" />
                          <span className="text-sm font-medium">Calories</span>
                        </div>
                        <p className="text-2xl font-bold">{healthSummary.calories.burned}</p>
                        <p className="text-xs text-gray-600">
                          {healthSummary.calories.remaining} remaining
                        </p>
                      </CardContent>
                    </Card>
                  </div>
                </TabsContent>

                <TabsContent value="activity" className="space-y-4">
                  <Card>
                    <CardHeader>
                      <CardTitle>Activity Summary</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-4">
                        <div className="flex justify-between items-center">
                          <span>Today's Steps</span>
                          <span className="font-bold">{healthSummary.steps.today.toLocaleString()}</span>
                        </div>
                        <div className="flex justify-between items-center">
                          <span>Weekly Steps</span>
                          <span className="font-bold">{healthSummary.steps.weekly.toLocaleString()}</span>
                        </div>
                        <div className="flex justify-between items-center">
                          <span>Daily Goal</span>
                          <span className="font-bold">{healthSummary.steps.goal.toLocaleString()}</span>
                        </div>
                        <Progress value={healthSummary.steps.progress} className="h-3" />
                      </div>
                    </CardContent>
                  </Card>
                </TabsContent>

                <TabsContent value="sleep" className="space-y-4">
                  <Card>
                    <CardHeader>
                      <CardTitle>Sleep Analysis</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-4">
                        <div className="flex justify-between items-center">
                          <span>Last Night</span>
                          <span className="font-bold">{healthSummary.sleep.lastNight.toFixed(1)} hours</span>
                        </div>
                        <div className="flex justify-between items-center">
                          <span>Average Sleep</span>
                          <span className="font-bold">{healthSummary.sleep.average.toFixed(1)} hours</span>
                        </div>
                        <div className="flex justify-between items-center">
                          <span>Sleep Quality</span>
                          <span className="font-bold">{healthSummary.sleep.quality}/10</span>
                        </div>
                        <div className="flex justify-between items-center">
                          <span>Sleep Efficiency</span>
                          <span className="font-bold">{healthSummary.sleep.efficiency.toFixed(0)}%</span>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </TabsContent>

                <TabsContent value="vitals" className="space-y-4">
                  <Card>
                    <CardHeader>
                      <CardTitle>Vital Signs</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-4">
                        <div className="flex justify-between items-center">
                          <span>Current Heart Rate</span>
                          <div className="flex items-center gap-2">
                            <span className="font-bold">{healthSummary.heartRate.current} BPM</span>
                            {getTrendIcon(healthSummary.heartRate.trend)}
                          </div>
                        </div>
                        <div className="flex justify-between items-center">
                          <span>Average Heart Rate</span>
                          <span className="font-bold">{Math.round(healthSummary.heartRate.average)} BPM</span>
                        </div>
                        <div className="flex justify-between items-center">
                          <span>Heart Rate Range</span>
                          <span className="font-bold">
                            {healthSummary.heartRate.min} - {healthSummary.heartRate.max} BPM
                          </span>
                        </div>
                        {healthSummary.weight.current > 0 && (
                          <>
                            <div className="flex justify-between items-center">
                              <span>Current Weight</span>
                              <div className="flex items-center gap-2">
                                <span className="font-bold">{healthSummary.weight.current} kg</span>
                                {getTrendIcon(healthSummary.weight.trend)}
                              </div>
                            </div>
                            <div className="flex justify-between items-center">
                              <span>Weight Change</span>
                              <span className={`font-bold ${getTrendColor(healthSummary.weight.trend)}`}>
                                {healthSummary.weight.change > 0 ? '+' : ''}{healthSummary.weight.change.toFixed(1)} kg
                              </span>
                            </div>
                          </>
                        )}
                      </div>
                    </CardContent>
                  </Card>
                </TabsContent>
              </Tabs>
            </div>
          )}

          {/* Error Display */}
          {error && (
            <Alert>
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}
        </CardContent>
      </Card>
    </div>
  );
};
