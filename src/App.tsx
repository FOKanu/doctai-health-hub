import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { useState, useEffect } from 'react';
import { AuthProvider } from './contexts/AuthContext';
import ScanScreen from './components/ScanScreen';
import ResultsScreen from './components/ResultsScreen';
import AnalyticsScreen from './components/AnalyticsScreen';
import { useAnalytics } from './hooks/useAnalytics';
import { useResults } from './hooks/useResults';
import { ScanResult, AnalyticsData } from './types/analysis';

function App() {
  return (
    <Router>
      <AuthProvider>
        <div className="min-h-screen bg-gray-100">
          <Routes>
            <Route path="/" element={<ScanScreen />} />
            <Route
              path="/results/:scanId"
              element={
                <ResultsScreenWrapper />
              }
            />
            <Route
              path="/analytics"
              element={
                <AnalyticsScreenWrapper />
              }
            />
          </Routes>
        </div>
      </AuthProvider>
    </Router>
  );
}

// Wrapper components to handle data fetching and state management
function ResultsScreenWrapper() {
  const { analyticsData, getScanDetails } = useAnalytics();
  const scanId = window.location.pathname.split('/').pop() || '';
  const [result, setResult] = useState<ScanResult | null>(null);

  useEffect(() => {
    const fetchResult = async () => {
      const scanResult = await getScanDetails(scanId);
      if (scanResult) {
        setResult(scanResult);
      }
    };
    fetchResult();
  }, [scanId, getScanDetails]);

  if (!result) {
    return <div>Loading...</div>;
  }

  const {
    handleShare,
    handleSchedule,
    handleMonitor
  } = useResults(result);

  return (
    <ResultsScreen
      result={result}
      onShare={handleShare}
      onSchedule={handleSchedule}
      onMonitor={handleMonitor}
    />
  );
}

function AnalyticsScreenWrapper() {
  const { analyticsData, loading, error, refreshData } = useAnalytics();

  if (loading) {
    return <div>Loading analytics data...</div>;
  }

  if (error) {
    return <div>Error: {error}</div>;
  }

  if (!analyticsData) {
    return <div>No analytics data available</div>;
  }

  return (
    <AnalyticsScreen
      data={analyticsData}
      onViewDetails={(scanId: string) => {
        // Navigate to results screen for the selected scan
        window.location.href = `/results/${scanId}`;
      }}
    />
  );
}

export default App;
