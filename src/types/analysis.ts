export type RiskLevel = 'low' | 'medium' | 'high';

export interface ScanResult {
  id: string;
  image: string;
  timestamp: string;
  findings: string[];
  riskLevel: RiskLevel;
  confidence: number;
  recommendations: string[];
  type: 'skin_lesion' | 'ct_scan' | 'mri' | 'xray' | 'eeg';
  metadata: {
    size: number;
    width: number;
    height: number;
    format: string;
    device_info?: {
      model: string;
      os: string;
      browser: string;
    };
  };
}

export interface AnalyticsData {
  trends: {
    riskLevels: {
      dates: string[];
      values: number[];
    };
    confidence: {
      dates: string[];
      values: number[];
    };
  };
  comparisons: {
    previousScans: ScanResult[];
    changes: {
      type: 'improvement' | 'deterioration' | 'stable';
      percentage: number;
    };
  };
  insights: {
    patterns: string[];
    recommendations: string[];
    progress: {
      startDate: string;
      currentStatus: string;
      improvement: number;
    };
  };
}
