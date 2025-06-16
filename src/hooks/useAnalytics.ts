import { useState, useEffect } from 'react';
import { analyticsService } from '../services/analyticsService';
import { AnalyticsData, ScanResult } from '../types/analysis';
import { useAuth } from './useAuth';

export const useAnalytics = () => {
  const [analyticsData, setAnalyticsData] = useState<AnalyticsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const { user } = useAuth();

  const fetchAnalyticsData = async () => {
    if (!user) {
      setError('User not authenticated');
      setLoading(false);
      return;
    }

    try {
      setLoading(true);
      const data = await analyticsService.getAnalyticsData(user.id);
      setAnalyticsData(data);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch analytics data');
    } finally {
      setLoading(false);
    }
  };

  const getScanDetails = async (scanId: string): Promise<ScanResult | null> => {
    try {
      const details = await analyticsService.getScanDetails(scanId);
      return details;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch scan details');
      return null;
    }
  };

  useEffect(() => {
    fetchAnalyticsData();
  }, [user]);

  return {
    analyticsData,
    loading,
    error,
    refreshData: fetchAnalyticsData,
    getScanDetails
  };
};
