export interface RiskIndicator {
  id: string;
  type: 'skin_lesion' | 'family_history' | 'sun_exposure' | 'age_risk' | 'medical_history';
  severity: 'low' | 'medium' | 'high';
  date: string;
  description: string;
  recommendation: string;
  status: 'pending' | 'monitoring' | 'resolved';
  location?: string;
  confidence: number;
  followUpRequired?: boolean;
  followUpDate?: string;
}

export interface RiskSummary {
  high: { count: number; percentage: number };
  medium: { count: number; percentage: number };
  low: { count: number; percentage: number };
}

export const getRiskIndicators = async (): Promise<RiskIndicator[]> => {
  // In a real app, this would fetch from your database
  return [
    {
      id: '1',
      type: 'skin_lesion',
      severity: 'high',
      date: '2024-06-15',
      description: 'Irregular borders detected on left arm mole',
      recommendation: 'Immediate dermatologist consultation',
      status: 'pending',
      location: 'Left Arm',
      confidence: 95,
      followUpRequired: true,
      followUpDate: '2024-06-22'
    },
    {
      id: '2',
      type: 'family_history',
      severity: 'medium',
      date: '2024-06-10',
      description: 'Family history of melanoma',
      recommendation: 'Regular screening every 6 months',
      status: 'monitoring',
      location: 'N/A',
      confidence: 85
    },
    {
      id: '3',
      type: 'sun_exposure',
      severity: 'medium',
      date: '2024-06-08',
      description: 'High UV exposure detected',
      recommendation: 'Increase sun protection measures',
      status: 'monitoring',
      location: 'Multiple areas',
      confidence: 78
    },
    {
      id: '4',
      type: 'age_risk',
      severity: 'low',
      date: '2024-06-05',
      description: 'Age-related risk factors',
      recommendation: 'Annual skin examination',
      status: 'monitoring',
      location: 'N/A',
      confidence: 65
    }
  ];
};

export const getRiskSummary = async (): Promise<RiskSummary> => {
  const indicators = await getRiskIndicators();
  const total = indicators.length;

  const high = indicators.filter(i => i.severity === 'high').length;
  const medium = indicators.filter(i => i.severity === 'medium').length;
  const low = indicators.filter(i => i.severity === 'low').length;

  return {
    high: { count: high, percentage: total > 0 ? Math.round((high / total) * 100) : 0 },
    medium: { count: medium, percentage: total > 0 ? Math.round((medium / total) * 100) : 0 },
    low: { count: low, percentage: total > 0 ? Math.round((low / total) * 100) : 0 }
  };
};

export const updateRiskStatus = async (id: string, status: RiskIndicator['status']): Promise<void> => {
  // In a real app, this would update the database
  console.log(`Updating risk indicator ${id} status to ${status}`);
};
