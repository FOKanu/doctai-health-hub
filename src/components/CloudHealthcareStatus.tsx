
import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Separator } from '@/components/ui/separator';
import {
  Cloud,
  CheckCircle,
  XCircle,
  AlertTriangle,
  Settings,
  Activity,
  Shield,
  Database
} from 'lucide-react';
import { getCloudHealthcareStatus } from '@/services/predictionService';
import { CLOUD_HEALTHCARE_CONFIG, validateCloudHealthcareConfig } from '@/services/cloudHealthcare/config';

export function CloudHealthcareStatus() {
  const status = getCloudHealthcareStatus();
  const configValidation = validateCloudHealthcareConfig();

  const getStatusIcon = (available: boolean) => {
    return available ? (
      <CheckCircle className="h-4 w-4 text-green-500" />
    ) : (
      <XCircle className="h-4 w-4 text-red-500" />
    );
  };

  const getStatusBadge = (available: boolean) => {
    return available ? (
      <Badge variant="default" className="bg-green-100 text-green-800">
        Available
      </Badge>
    ) : (
      <Badge variant="secondary" className="bg-gray-100 text-gray-600">
        Unavailable
      </Badge>
    );
  };

  const getProviderStatus = (provider: string) => {
    switch (provider) {
      case 'google':
        return {
          name: 'Google Cloud Healthcare',
          enabled: CLOUD_HEALTHCARE_CONFIG.GOOGLE.ENABLED,
          configured: Boolean(CLOUD_HEALTHCARE_CONFIG.GOOGLE.PROJECT_ID),
          icon: <Cloud className="h-4 w-4" />
        };
      case 'azure':
        return {
          name: 'Azure Health Bot',
          enabled: CLOUD_HEALTHCARE_CONFIG.AZURE.ENABLED,
          configured: !!CLOUD_HEALTHCARE_CONFIG.AZURE.ENDPOINT,
          icon: <Activity className="h-4 w-4" />
        };
      case 'watson':
        return {
          name: 'IBM Watson Health',
          enabled: CLOUD_HEALTHCARE_CONFIG.WATSON.ENABLED,
          configured: !!CLOUD_HEALTHCARE_CONFIG.WATSON.API_KEY,
          icon: <Shield className="h-4 w-4" />
        };
      default:
        return {
          name: 'Unknown Provider',
          enabled: false,
          configured: false,
          icon: <Database className="h-4 w-4" />
        };
    }
  };

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Cloud className="h-5 w-5" />
          Cloud Healthcare APIs
        </CardTitle>
        <CardDescription>
          Status and configuration of cloud healthcare providers
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Overall Status */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            {getStatusIcon(status.available)}
            <span className="font-medium">Overall Status</span>
          </div>
          {getStatusBadge(status.available)}
        </div>

        {!status.available && (
          <div className="text-sm text-muted-foreground">
            {status.reason}
          </div>
        )}

        <Separator />

        {/* Configuration Validation */}
        {!configValidation.isValid && (
          <div className="space-y-2">
            <div className="flex items-center gap-2 text-amber-600">
              <AlertTriangle className="h-4 w-4" />
              <span className="font-medium">Configuration Issues</span>
            </div>
            <ul className="text-sm text-muted-foreground space-y-1">
              {configValidation.errors.map((error, index) => (
                <li key={index} className="flex items-start gap-2">
                  <span className="text-amber-500">•</span>
                  {error}
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Available Providers */}
        {status.available && (
          <div className="space-y-3">
            <h4 className="font-medium">Available Providers</h4>
            <div className="grid gap-2">
              {status.providers.map((provider) => {
                const providerStatus = getProviderStatus(provider);
                const isPrimary = provider === status.primaryProvider;

                return (
                  <div key={provider} className="flex items-center justify-between p-2 border rounded-lg">
                    <div className="flex items-center gap-2">
                      {providerStatus.icon}
                      <span className="text-sm font-medium">{providerStatus.name}</span>
                      {isPrimary && (
                        <Badge variant="outline" className="text-xs">
                          Primary
                        </Badge>
                      )}
                    </div>
                    <div className="flex items-center gap-2">
                      {getStatusIcon(providerStatus.enabled && providerStatus.configured)}
                      <span className="text-xs text-muted-foreground">
                        {providerStatus.enabled ? 'Enabled' : 'Disabled'}
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        <Separator />

        {/* Feature Flags */}
        <div className="space-y-3">
          <h4 className="font-medium">Feature Flags</h4>
          <div className="grid gap-2 text-sm">
            <div className="flex items-center justify-between">
              <span>Fallback to Custom ML</span>
              {getStatusIcon(CLOUD_HEALTHCARE_CONFIG.ENABLE_FALLBACK)}
            </div>
            <div className="flex items-center justify-between">
              <span>Consensus Analysis</span>
              {getStatusIcon(CLOUD_HEALTHCARE_CONFIG.ENABLE_CONSENSUS)}
            </div>
            <div className="flex items-center justify-between">
              <span>Debug Mode</span>
              {getStatusIcon(CLOUD_HEALTHCARE_CONFIG.DEBUG_MODE)}
            </div>
          </div>
        </div>

        {/* Performance Settings */}
        <div className="space-y-3">
          <h4 className="font-medium">Performance Settings</h4>
          <div className="grid gap-2 text-sm">
            <div className="flex items-center justify-between">
              <span>Timeout</span>
              <span className="text-muted-foreground">{CLOUD_HEALTHCARE_CONFIG.TIMEOUT_MS}ms</span>
            </div>
            <div className="flex items-center justify-between">
              <span>Max Retries</span>
              <span className="text-muted-foreground">{CLOUD_HEALTHCARE_CONFIG.MAX_RETRIES}</span>
            </div>
          </div>
        </div>

        {/* Actions */}
        <div className="flex gap-2 pt-2">
          <Button variant="outline" size="sm" className="flex-1">
            <Settings className="h-4 w-4 mr-2" />
            Configure
          </Button>
          <Button variant="outline" size="sm" className="flex-1">
            <Activity className="h-4 w-4 mr-2" />
            Test Connection
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
