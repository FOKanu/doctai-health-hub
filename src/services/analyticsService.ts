
import { supabase } from '@/integrations/supabase/client';
import { AnalyticsData, ScanResult } from '../types/analysis';

interface AnalysisResult {
  riskLevel?: string;
  confidence?: number;
  findings?: string[];
  recommendations?: string[];
}

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
              const analysisResult = scan.analysis_result as AnalysisResult;
              // Convert risk level to numeric value
              switch (analysisResult?.riskLevel) {
                case 'high': return 3;
                case 'medium': return 2;
                case 'low': return 1;
                default: return 0;
              }
            })
          },
          confidence: {
            dates: scans.map(scan => scan.created_at),
            values: scans.map(scan => {
              const analysisResult = scan.analysis_result as AnalysisResult;
              return analysisResult?.confidence || 0;
            })
          }
        },
        comparisons: {
          previousScans: scans.map(scan => {
            const analysisResult = scan.analysis_result as AnalysisResult;
            const metadata = scan.metadata as any;
            return {
              id: scan.id,
              image: scan.url,
              timestamp: scan.created_at,
              findings: analysisResult?.findings || [],
              riskLevel: (analysisResult?.riskLevel || 'low') as 'low' | 'medium' | 'high',
              confidence: analysisResult?.confidence || 0,
              recommendations: analysisResult?.recommendations || [],
              type: scan.type,
              metadata: metadata || {
                size: 0,
                width: 0,
                height: 0,
                format: ''
              }
            };
          }),
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

      const analysisResult = data.analysis_result as AnalysisResult;
      const metadata = data.metadata as any;

      return {
        id: data.id,
        image: data.url,
        timestamp: data.created_at,
        findings: analysisResult?.findings || [],
        riskLevel: (analysisResult?.riskLevel || 'low') as 'low' | 'medium' | 'high',
        confidence: analysisResult?.confidence || 0,
        recommendations: analysisResult?.recommendations || [],
        type: data.type,
        metadata: metadata || {
          size: 0,
          width: 0,
          height: 0,
          format: ''
        }
      };
    } catch (error) {
      console.error('Error fetching scan details:', error);
      throw error;
    }
  }
};

// Helper functions
function calculateChanges(scans: any[]): {
  type: 'improvement' | 'deterioration' | 'stable';
  percentage: number;
} {
  if (scans.length < 2) {
    return {
      type: 'stable',
      percentage: 0
    };
  }

  const latest = scans[0];
  const previous = scans[1];

  const latestAnalysis = latest.analysis_result as AnalysisResult;
  const previousAnalysis = previous.analysis_result as AnalysisResult;

  const latestRisk = getRiskValue(latestAnalysis?.riskLevel);
  const previousRisk = getRiskValue(previousAnalysis?.riskLevel);

  const change = ((latestRisk - previousRisk) / previousRisk) * 100;

  return {
    type: change > 0 ? 'deterioration' : change < 0 ? 'improvement' : 'stable',
    percentage: Math.abs(change)
  };
}

function getRiskValue(riskLevel?: string): number {
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

  const firstAnalysis = firstScan.analysis_result as AnalysisResult;
  const latestAnalysis = latestScan.analysis_result as AnalysisResult;

  const firstRisk = getRiskValue(firstAnalysis?.riskLevel);
  const latestRisk = getRiskValue(latestAnalysis?.riskLevel);

  const improvement = ((firstRisk - latestRisk) / firstRisk) * 100;

  return {
    startDate: firstScan.created_at,
    currentStatus: `Current risk level: ${latestAnalysis?.riskLevel}`,
    improvement: Math.max(0, improvement)
  };
}
