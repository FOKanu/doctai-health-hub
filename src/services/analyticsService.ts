
import { supabase } from './supabaseClient';
import { AnalyticsData, ScanResult } from '../types/analysis';

export const analyticsService = {
  async getAnalyticsData(userId: string): Promise<AnalyticsData> {
    try {
      // Get all scans for the user
      const { data: scans, error: scansError } = await supabase
        .from('image_metadata')
        .select('*')
        .eq('user_id', userId)
        .order('created_at', { ascending: false });

      if (scansError) throw scansError;

      // Process scans into analytics data
      const processedData: AnalyticsData = {
        trends: {
          riskLevels: {
            dates: scans.map(scan => scan.created_at),
            values: scans.map(scan => {
              // Convert risk level to numeric value
              switch (scan.analysis_result?.riskLevel) {
                case 'high': return 3;
                case 'medium': return 2;
                case 'low': return 1;
                default: return 0;
              }
            })
          },
          confidence: {
            dates: scans.map(scan => scan.created_at),
            values: scans.map(scan => scan.analysis_result?.confidence || 0)
          }
        },
        comparisons: {
          previousScans: scans.map(scan => ({
            id: scan.id,
            image: scan.url,
            timestamp: scan.created_at,
            findings: scan.analysis_result?.findings || [],
            riskLevel: scan.analysis_result?.riskLevel || 'low',
            confidence: scan.analysis_result?.confidence || 0,
            recommendations: scan.analysis_result?.recommendations || [],
            type: scan.type,
            metadata: scan.metadata
          })),
          changes: calculateChanges(scans)
        },
        insights: {
          patterns: generatePatterns(scans),
          recommendations: generateRecommendations(scans),
          progress: calculateProgress(scans)
        }
      };

      return processedData;
    } catch (error) {
      console.error('Error fetching analytics data:', error);
      throw error;
    }
  },

  async getScanDetails(scanId: string): Promise<ScanResult> {
    try {
      const { data, error } = await supabase
        .from('image_metadata')
        .select('*')
        .eq('id', scanId)
        .single();

      if (error) throw error;

      return {
        id: data.id,
        image: data.url,
        timestamp: data.created_at,
        findings: data.analysis_result?.findings || [],
        riskLevel: data.analysis_result?.riskLevel || 'low',
        confidence: data.analysis_result?.confidence || 0,
        recommendations: data.analysis_result?.recommendations || [],
        type: data.type,
        metadata: data.metadata
      };
    } catch (error) {
      console.error('Error fetching scan details:', error);
      throw error;
    }
  }
};

// Helper functions
function calculateChanges(scans: any[]): { type: 'improvement' | 'deterioration' | 'stable'; percentage: number } {
  if (scans.length < 2) {
    return {
      type: 'stable',
      percentage: 0
    };
  }

  const latest = scans[0];
  const previous = scans[1];

  const latestRisk = getRiskValue(latest.analysis_result?.riskLevel);
  const previousRisk = getRiskValue(previous.analysis_result?.riskLevel);

  const change = ((latestRisk - previousRisk) / previousRisk) * 100;

  let type: 'improvement' | 'deterioration' | 'stable' = 'stable';
  if (change > 0) {
    type = 'deterioration';
  } else if (change < 0) {
    type = 'improvement';
  }

  return {
    type,
    percentage: Math.abs(change)
  };
}

function getRiskValue(riskLevel: string): number {
  switch (riskLevel) {
    case 'high': return 3;
    case 'medium': return 2;
    case 'low': return 1;
    default: return 0;
  }
}

function generatePatterns(scans: any[]): string[] {
  const patterns: string[] = [];

  // Add pattern detection logic here
  // Example patterns:
  if (scans.length > 0) {
    patterns.push('Regular monitoring shows consistent risk levels');
    patterns.push('Most scans are performed during morning hours');
  }

  return patterns;
}

function generateRecommendations(scans: any[]): string[] {
  const recommendations: string[] = [];

  // Add recommendation generation logic here
  // Example recommendations:
  if (scans.length > 0) {
    recommendations.push('Continue regular monitoring schedule');
    recommendations.push('Consider consulting a specialist for detailed analysis');
  }

  return recommendations;
}

function calculateProgress(scans: any[]): {
  startDate: string;
  currentStatus: string;
  improvement: number;
} {
  if (scans.length === 0) {
    return {
      startDate: new Date().toISOString(),
      currentStatus: 'No scans available',
      improvement: 0
    };
  }

  const firstScan = scans[scans.length - 1];
  const latestScan = scans[0];

  const firstRisk = getRiskValue(firstScan.analysis_result?.riskLevel);
  const latestRisk = getRiskValue(latestScan.analysis_result?.riskLevel);

  const improvement = ((firstRisk - latestRisk) / firstRisk) * 100;

  return {
    startDate: firstScan.created_at,
    currentStatus: `Current risk level: ${latestScan.analysis_result?.riskLevel}`,
    improvement: Math.max(0, improvement)
  };
}
