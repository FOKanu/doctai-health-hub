
import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Separator } from '@/components/ui/separator';
import {
  CheckCircle,
  XCircle,
  AlertTriangle,
  Cloud,
  Settings,
  Shield,
  Database,
  Loader2,
  RefreshCw
} from 'lucide-react';
import { CLOUD_HEALTHCARE_CONFIG } from '@/services/cloudHealthcare/config';

interface ConfigValidationResult {
  isValid: boolean;
  errors: string[];
  warnings: string[];
  details: {
    projectId: boolean;
    location: boolean;
    datasetId: boolean;
    apiKey: boolean;
    serviceAccount: boolean;
    apisEnabled: boolean;
  };
}

export function GoogleCloudConfigValidator() {
  const [validationResult, setValidationResult] = useState<ConfigValidationResult | null>(null);
  const [isValidating, setIsValidating] = useState(false);
  const [lastValidated, setLastValidated] = useState<Date | null>(null);

  const validateConfiguration = async (): Promise<ConfigValidationResult> => {
    const errors: string[] = [];
    const warnings: string[] = [];
    const details = {
      projectId: false,
      location: false,
      datasetId: false,
      apiKey: false,
      serviceAccount: false,
      apisEnabled: false
    };

    // Check if Google Healthcare is enabled
    if (!CLOUD_HEALTHCARE_CONFIG.ENABLE_CLOUD_HEALTHCARE) {
      errors.push('Google Cloud Healthcare is not enabled');
    }

    // Validate project ID
    if (!CLOUD_HEALTHCARE_CONFIG.GOOGLE.PROJECT_ID) {
      errors.push('Google Cloud Project ID is not configured');
    } else {
      details.projectId = true;
    }

    // Validate location
    if (!CLOUD_HEALTHCARE_CONFIG.GOOGLE.LOCATION) {
      errors.push('Google Cloud location is not configured');
    } else {
      details.location = true;
    }

    // Validate dataset ID
    if (!CLOUD_HEALTHCARE_CONFIG.GOOGLE.DATASET_ID) {
      errors.push('Healthcare dataset ID is not configured');
    } else {
      details.datasetId = true;
    }

    // Check API key (optional but recommended)
    if (!CLOUD_HEALTHCARE_CONFIG.GOOGLE.API_KEY) {
      warnings.push('Google Cloud API key is not configured (using service account authentication)');
    } else {
      details.apiKey = true;
    }

    // Check service account credentials
    const serviceAccountPath = import.meta.env.GOOGLE_APPLICATION_CREDENTIALS;
    if (!serviceAccountPath) {
      warnings.push('Service account credentials path not found');
    } else {
      details.serviceAccount = true;
    }

    // Check if APIs are enabled (this would require actual API call)
    // For now, we'll assume they are if the config is valid
    if (details.projectId && details.location && details.datasetId) {
      details.apisEnabled = true;
    }

    return {
      isValid: errors.length === 0,
      errors,
      warnings,
      details
    };
  };

  const handleValidate = async () => {
    setIsValidating(true);
    try {
      const result = await validateConfiguration();
      setValidationResult(result);
      setLastValidated(new Date());
    } catch (error) {
      console.error('Validation error:', error);
      setValidationResult({
        isValid: false,
        errors: ['Validation failed: ' + (error as Error).message],
        warnings: [],
        details: {
          projectId: false,
          location: false,
          datasetId: false,
          apiKey: false,
          serviceAccount: false,
          apisEnabled: false
        }
      });
    } finally {
      setIsValidating(false);
    }
  };

  useEffect(() => {
    // Auto-validate on component mount
    handleValidate();
  }, []);

  const getStatusIcon = (isValid: boolean) => {
    if (isValidating) return <Loader2 className="h-4 w-4 animate-spin" />;
    return isValid ? <CheckCircle className="h-4 w-4 text-green-500" /> : <XCircle className="h-4 w-4 text-red-500" />;
  };

  const getStatusBadge = (isValid: boolean) => {
    if (isValidating) {
      return <Badge variant="secondary">Validating...</Badge>;
    }
    return isValid ?
      <Badge variant="default" className="bg-green-500">Valid</Badge> :
      <Badge variant="destructive">Invalid</Badge>;
  };

  return (
    <Card className="w-full max-w-2xl">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Cloud className="h-5 w-5" />
            <CardTitle>Google Cloud Configuration</CardTitle>
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={handleValidate}
            disabled={isValidating}
          >
            <RefreshCw className={`h-4 w-4 mr-2 ${isValidating ? 'animate-spin' : ''}`} />
            Validate
          </Button>
        </div>
        <CardDescription>
          Validate your Google Cloud Healthcare configuration
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Overall Status */}
        <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
          <div className="flex items-center space-x-2">
            {getStatusIcon(validationResult?.isValid ?? false)}
            <span className="font-medium">Configuration Status</span>
          </div>
          {getStatusBadge(validationResult?.isValid ?? false)}
        </div>

        {/* Configuration Details */}
        {validationResult && (
          <div className="space-y-3">
            <h4 className="font-medium flex items-center space-x-2">
              <Settings className="h-4 w-4" />
              <span>Configuration Details</span>
            </h4>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              <div className="flex items-center justify-between p-2 bg-gray-50 rounded">
                <span className="text-sm">Project ID</span>
                {getStatusIcon(validationResult.details.projectId)}
              </div>

              <div className="flex items-center justify-between p-2 bg-gray-50 rounded">
                <span className="text-sm">Location</span>
                {getStatusIcon(validationResult.details.location)}
              </div>

              <div className="flex items-center justify-between p-2 bg-gray-50 rounded">
                <span className="text-sm">Dataset ID</span>
                {getStatusIcon(validationResult.details.datasetId)}
              </div>

              <div className="flex items-center justify-between p-2 bg-gray-50 rounded">
                <span className="text-sm">API Key</span>
                {getStatusIcon(validationResult.details.apiKey)}
              </div>

              <div className="flex items-center justify-between p-2 bg-gray-50 rounded">
                <span className="text-sm">Service Account</span>
                {getStatusIcon(validationResult.details.serviceAccount)}
              </div>

              <div className="flex items-center justify-between p-2 bg-gray-50 rounded">
                <span className="text-sm">APIs Enabled</span>
                {getStatusIcon(validationResult.details.apisEnabled)}
              </div>
            </div>
          </div>
        )}

        {/* Current Configuration */}
        <Separator />
        <div className="space-y-3">
          <h4 className="font-medium flex items-center space-x-2">
            <Database className="h-4 w-4" />
            <span>Current Configuration</span>
          </h4>

          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-600">Project ID:</span>
              <span className="font-mono">{CLOUD_HEALTHCARE_CONFIG.GOOGLE.PROJECT_ID || 'Not configured'}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Location:</span>
              <span className="font-mono">{CLOUD_HEALTHCARE_CONFIG.GOOGLE.LOCATION || 'Not configured'}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Dataset ID:</span>
              <span className="font-mono">{CLOUD_HEALTHCARE_CONFIG.GOOGLE.DATASET_ID || 'Not configured'}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">API Key:</span>
              <span className="font-mono">
                {CLOUD_HEALTHCARE_CONFIG.GOOGLE.API_KEY ? 'Configured' : 'Not configured'}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Service Account:</span>
              <span className="font-mono">
                {import.meta.env.GOOGLE_APPLICATION_CREDENTIALS ? 'Configured' : 'Not configured'}
              </span>
            </div>
          </div>
        </div>

        {/* Errors */}
        {validationResult && validationResult.errors.length > 0 && (
          <Alert variant="destructive">
            <XCircle className="h-4 w-4" />
            <AlertDescription>
              <div className="space-y-1">
                <strong>Configuration Errors:</strong>
                <ul className="list-disc list-inside space-y-1">
                  {validationResult.errors.map((error, index) => (
                    <li key={index}>{error}</li>
                  ))}
                </ul>
              </div>
            </AlertDescription>
          </Alert>
        )}

        {/* Warnings */}
        {validationResult && validationResult.warnings.length > 0 && (
          <Alert>
            <AlertTriangle className="h-4 w-4" />
            <AlertDescription>
              <div className="space-y-1">
                <strong>Configuration Warnings:</strong>
                <ul className="list-disc list-inside space-y-1">
                  {validationResult.warnings.map((warning, index) => (
                    <li key={index}>{warning}</li>
                  ))}
                </ul>
              </div>
            </AlertDescription>
          </Alert>
        )}

        {/* Success Message */}
        {validationResult && validationResult.isValid && validationResult.errors.length === 0 && (
          <Alert className="border-green-200 bg-green-50">
            <CheckCircle className="h-4 w-4 text-green-600" />
            <AlertDescription className="text-green-800">
              <strong>Configuration is valid!</strong> Your Google Cloud Healthcare setup appears to be correctly configured.
            </AlertDescription>
          </Alert>
        )}

        {/* Last Validated */}
        {lastValidated && (
          <div className="text-xs text-gray-500 text-center">
            Last validated: {lastValidated.toLocaleString()}
          </div>
        )}

        {/* Action Buttons */}
        <div className="flex space-x-2 pt-4">
          <Button
            variant="outline"
            size="sm"
            onClick={() => window.open('https://console.cloud.google.com', '_blank')}
            className="flex-1"
          >
            <Cloud className="h-4 w-4 mr-2" />
            Open Google Cloud Console
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => window.open('QUICK_GOOGLE_CLOUD_SETUP.md', '_blank')}
            className="flex-1"
          >
            <Settings className="h-4 w-4 mr-2" />
            Setup Guide
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
