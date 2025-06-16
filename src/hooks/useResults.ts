import { useState } from 'react';
import { ScanResult } from '../types/analysis';
import { useNavigate } from 'react-router-dom';
import { analyticsService } from '../services/analyticsService';

export const useResults = (initialResult: ScanResult) => {
  const [result, setResult] = useState<ScanResult>(initialResult);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();

  const handleShare = async () => {
    try {
      setLoading(true);
      // Implement sharing logic here
      // This could involve generating a shareable link or sending to a doctor
      console.log('Sharing result:', result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to share result');
    } finally {
      setLoading(false);
    }
  };

  const handleSchedule = () => {
    // Navigate to scheduling page with result context
    navigate('/schedule', { state: { result } });
  };

  const handleMonitor = () => {
    // Navigate to monitoring page with result context
    navigate('/monitor', { state: { result } });
  };

  const handleViewAnalytics = () => {
    // Navigate to analytics page
    navigate('/analytics');
  };

  const updateResult = async (scanId: string) => {
    try {
      setLoading(true);
      const updatedResult = await analyticsService.getScanDetails(scanId);
      if (updatedResult) {
        setResult(updatedResult);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update result');
    } finally {
      setLoading(false);
    }
  };

  return {
    result,
    loading,
    error,
    handleShare,
    handleSchedule,
    handleMonitor,
    handleViewAnalytics,
    updateResult
  };
};
