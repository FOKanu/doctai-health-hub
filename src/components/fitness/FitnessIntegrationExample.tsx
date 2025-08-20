import React from 'react';
import { FitnessIntegration } from './FitnessIntegration';

/**
 * Example component showing how to integrate fitness features
 * into your main application
 */
export const FitnessIntegrationExample: React.FC = () => {
  // In a real app, you would get the userId from your auth context
  const userId = 'example-user-id'; // Replace with actual user ID from auth

  return (
    <div className="container mx-auto p-6">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">
          Smart Watch Integration
        </h1>
        <p className="text-gray-600">
          Connect your Google Fit or Fitbit device to automatically sync health metrics
          and get personalized health insights.
        </p>
      </div>

      <FitnessIntegration userId={userId} />
    </div>
  );
};

/**
 * Example of how to add fitness integration to your main dashboard
 */
export const DashboardWithFitness: React.FC = () => {
  const userId = 'example-user-id'; // Replace with actual user ID

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* Your existing dashboard components */}
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-xl font-semibold mb-4">Health Overview</h2>
        {/* Your existing health overview content */}
      </div>

      {/* Fitness Integration */}
      <div className="bg-white rounded-lg shadow">
        <FitnessIntegration userId={userId} />
      </div>
    </div>
  );
};

/**
 * Example of how to add fitness integration to a settings page
 */
export const SettingsWithFitness: React.FC = () => {
  const userId = 'example-user-id'; // Replace with actual user ID

  return (
    <div className="space-y-6">
      {/* Other settings sections */}
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-xl font-semibold mb-4">Account Settings</h2>
        {/* Account settings content */}
      </div>

      {/* Fitness Integration Settings */}
      <div className="bg-white rounded-lg shadow">
        <FitnessIntegration userId={userId} />
      </div>
    </div>
  );
};
