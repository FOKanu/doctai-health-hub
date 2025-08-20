
import React from 'react';
import { RiskDistributionChart } from './risk/RiskDistributionChart';
import { RiskProgressionChart } from './risk/RiskProgressionChart';
import { RiskAlerts } from './risk/RiskAlerts';

interface RiskPredictionsProps {
  dateRange: string;
}

export const RiskPredictions: React.FC<RiskPredictionsProps> = ({ dateRange }) => {
  const riskDistribution = [
    { name: 'Low Risk', value: 65, color: '#10b981', count: 13 },
    { name: 'Medium Risk', value: 25, color: '#f59e0b', count: 5 },
    { name: 'High Risk', value: 10, color: '#ef4444', count: 2 },
  ];

  const riskProgression = [
    { date: '2024-01-01', low: 70, medium: 25, high: 5 },
    { date: '2024-01-15', low: 68, medium: 27, high: 5 },
    { date: '2024-02-01', low: 65, medium: 25, high: 10 },
    { date: '2024-02-15', low: 65, medium: 25, high: 10 },
  ];

  const alerts = [
    {
      id: 1,
      type: 'skin',
      title: 'Mole Changes Detected',
      description: 'Asymmetrical changes noted in lesion ID: MSK-2024-003',
      risk: 'high',
      date: '2024-02-14',
      action: 'Schedule dermatologist appointment',
    },
    {
      id: 2,
      type: 'cardiovascular',
      title: 'Elevated Resting Heart Rate',
      description: 'Average resting HR increased by 15% over 7 days',
      risk: 'medium',
      date: '2024-02-13',
      action: 'Monitor for 3 more days',
    },
    {
      id: 3,
      type: 'metabolic',
      title: 'Sleep Pattern Disruption',
      description: 'Sleep quality decreased by 20% this week',
      risk: 'medium',
      date: '2024-02-12',
      action: 'Review sleep hygiene',
    },
  ];

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <RiskDistributionChart data={riskDistribution} />
        <RiskProgressionChart />
      </div>
      <RiskAlerts alerts={alerts} />
    </div>
  );
};
