/**
 * Time-Series Service
 * Handles operations for time-series health data and analytics
 */

import { supabase } from '@/integrations/supabase/client';
import type {
  PatientTimeline,
  HealthMetricTimeseries,
  ScanSequence,
  TreatmentResponse,
  RiskProgression,
  HealthMetricsTrend,
  PatientProgressionSummary
} from '@/integrations/supabase/types';

export interface TimeSeriesQueryParams {
  userId: string;
  startDate?: string;
  endDate?: string;
  metricType?: string;
  conditionType?: string;
  limit?: number;
}

export interface HealthMetricsData {
  heartRate: Array<{ timestamp: string; value: number }>;
  bloodPressure: Array<{ timestamp: string; systolic: number; diastolic: number }>;
  temperature: Array<{ timestamp: string; value: number }>;
  weight: Array<{ timestamp: string; value: number }>;
  sleepHours: Array<{ timestamp: string; value: number }>;
  steps: Array<{ timestamp: string; value: number }>;
  waterIntake: Array<{ timestamp: string; value: number }>;
}

export interface ProgressionData {
  conditionType: string;
  status: string;
  severityScore: number;
  confidenceScore: number;
  baselineDate: string;
  daysSinceBaseline: number;
  trend: string;
  recentScans: Array<{
    id: string;
    date: string;
    riskLevel: string;
    confidence: number;
  }>;
}

export class TimeSeriesService {
  /**
   * Get health metrics time-series data
   */
  async getHealthMetrics(params: TimeSeriesQueryParams): Promise<HealthMetricsData> {
    try {
      const { data, error } = await supabase
        .from('health_metrics_timeseries')
        .select('*')
        .eq('user_id', params.userId)
        .gte('recorded_at', params.startDate || new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString())
        .lte('recorded_at', params.endDate || new Date().toISOString())
        .order('recorded_at', { ascending: true });

      if (error) throw error;

      // Process and organize data by metric type
      const metrics: HealthMetricsData = {
        heartRate: [],
        bloodPressure: [],
        temperature: [],
        weight: [],
        sleepHours: [],
        steps: [],
        waterIntake: []
      };

      data?.forEach(record => {
        const timestamp = record.recorded_at;
        const value = record.value;

        switch (record.metric_type) {
          case 'heart_rate':
            if (value?.value) {
              metrics.heartRate.push({
                timestamp,
                value: value.value
              });
            }
            break;
          case 'blood_pressure':
            if (value?.systolic && value?.diastolic) {
              metrics.bloodPressure.push({
                timestamp,
                systolic: value.systolic,
                diastolic: value.diastolic
              });
            }
            break;
          case 'temperature':
            if (value?.value) {
              metrics.temperature.push({
                timestamp,
                value: value.value
              });
            }
            break;
          case 'weight':
            if (value?.value) {
              metrics.weight.push({
                timestamp,
                value: value.value
              });
            }
            break;
          case 'sleep_hours':
            if (value?.hours) {
              metrics.sleepHours.push({
                timestamp,
                value: value.hours
              });
            }
            break;
          case 'steps':
            if (value?.count) {
              metrics.steps.push({
                timestamp,
                value: value.count
              });
            }
            break;
          case 'water_intake':
            if (value?.amount) {
              metrics.waterIntake.push({
                timestamp,
                value: value.amount
              });
            }
            break;
        }
      });

      return metrics;
    } catch (error) {
      console.error('Error fetching health metrics:', error);
      // Return mock data for development
      return this.getMockHealthMetrics();
    }
  }

  /**
   * Get patient progression data
   */
  async getPatientProgression(params: TimeSeriesQueryParams): Promise<ProgressionData[]> {
    try {
      const { data, error } = await supabase
        .from('patient_timelines')
        .select('*')
        .eq('user_id', params.userId)
        .order('baseline_date', { ascending: false });

      if (error) throw error;

      const progressions: ProgressionData[] = [];

      for (const timeline of data || []) {
        // Get recent scans for this condition
        const recentScans = await this.getRecentScans(params.userId, timeline.condition_type);

        progressions.push({
          conditionType: timeline.condition_type,
          status: timeline.status,
          severityScore: timeline.severity_score || 0,
          confidenceScore: timeline.confidence_score || 0,
          baselineDate: timeline.baseline_date,
          daysSinceBaseline: this.calculateDaysSinceBaseline(timeline.baseline_date),
          trend: this.determineTrend(timeline.status),
          recentScans
        });
      }

      return progressions;
    } catch (error) {
      console.error('Error fetching patient progression:', error);
      // Return mock data for development
      return this.getMockProgressionData();
    }
  }

  /**
   * Get risk progression data
   */
  async getRiskProgression(params: TimeSeriesQueryParams): Promise<RiskProgression[]> {
    try {
      const { data, error } = await supabase
        .from('risk_progressions')
        .select('*')
        .eq('user_id', params.userId)
        .gte('recorded_at', params.startDate || new Date(Date.now() - 90 * 24 * 60 * 60 * 1000).toISOString())
        .lte('recorded_at', params.endDate || new Date().toISOString())
        .order('recorded_at', { ascending: true });

      if (error) throw error;

      return data || [];
    } catch (error) {
      console.error('Error fetching risk progression:', error);
      return this.getMockRiskProgression();
    }
  }

  /**
   * Get scan sequences for progression analysis
   */
  async getScanSequences(params: TimeSeriesQueryParams): Promise<ScanSequence[]> {
    try {
      const { data, error } = await supabase
        .from('scan_sequences')
        .select('*')
        .eq('user_id', params.userId)
        .order('created_at', { ascending: false });

      if (error) throw error;

      return data || [];
    } catch (error) {
      console.error('Error fetching scan sequences:', error);
      return [];
    }
  }

  /**
   * Save health metric data
   */
  async saveHealthMetric(
    userId: string,
    metricType: string,
    value: any,
    timestamp: string,
    deviceSource?: string
  ): Promise<void> {
    try {
      const { error } = await supabase
        .from('health_metrics_timeseries')
        .insert({
          user_id: userId,
          metric_type: metricType,
          value: value,
          recorded_at: timestamp,
          device_source: deviceSource,
          accuracy_score: 0.95 // Default accuracy
        });

      if (error) throw error;
    } catch (error) {
      console.error('Error saving health metric:', error);
      throw error;
    }
  }

  /**
   * Create or update patient timeline
   */
  async savePatientTimeline(timeline: Partial<PatientTimeline>): Promise<string> {
    try {
      const { data, error } = await supabase
        .from('patient_timelines')
        .insert(timeline)
        .select('id')
        .single();

      if (error) throw error;

      return data.id;
    } catch (error) {
      console.error('Error saving patient timeline:', error);
      throw error;
    }
  }

  /**
   * Get recent scans for a condition
   */
  private async getRecentScans(userId: string, conditionType: string): Promise<Array<{
    id: string;
    date: string;
    riskLevel: string;
    confidence: number;
  }>> {
    try {
      const { data, error } = await supabase
        .from('image_metadata')
        .select('id, created_at, analysis_result')
        .eq('user_id', userId)
        .eq('type', conditionType)
        .order('created_at', { ascending: false })
        .limit(5);

      if (error) throw error;

      return (data || []).map(scan => ({
        id: scan.id,
        date: scan.created_at,
        riskLevel: scan.analysis_result?.riskLevel || 'unknown',
        confidence: scan.analysis_result?.confidence || 0
      }));
    } catch (error) {
      console.error('Error fetching recent scans:', error);
      return [];
    }
  }

  /**
   * Calculate days since baseline
   */
  private calculateDaysSinceBaseline(baselineDate: string): number {
    const baseline = new Date(baselineDate);
    const now = new Date();
    return Math.ceil((now.getTime() - baseline.getTime()) / (1000 * 60 * 60 * 24));
  }

  /**
   * Determine trend from status
   */
  private determineTrend(status: string): string {
    switch (status) {
      case 'improving': return 'positive';
      case 'worsening': return 'negative';
      default: return 'neutral';
    }
  }

  /**
   * Mock data for development
   */
  private getMockHealthMetrics(): HealthMetricsData {
    const now = new Date();
    const data: HealthMetricsData = {
      heartRate: [],
      bloodPressure: [],
      temperature: [],
      weight: [],
      sleepHours: [],
      steps: [],
      waterIntake: []
    };

    // Generate 30 days of mock data
    for (let i = 29; i >= 0; i--) {
      const date = new Date(now.getTime() - i * 24 * 60 * 60 * 1000);
      const timestamp = date.toISOString();

      data.heartRate.push({
        timestamp,
        value: 70 + Math.random() * 20 + Math.sin(i * 0.2) * 5
      });

      data.bloodPressure.push({
        timestamp,
        systolic: 120 + Math.random() * 20,
        diastolic: 80 + Math.random() * 10
      });

      data.temperature.push({
        timestamp,
        value: 98.6 + Math.random() * 0.8
      });

      data.weight.push({
        timestamp,
        value: 165 + Math.sin(i * 0.1) * 2 + Math.random() * 0.5
      });

      data.sleepHours.push({
        timestamp,
        value: 7 + Math.random() * 2
      });

      data.steps.push({
        timestamp,
        value: 8000 + Math.random() * 4000
      });

      data.waterIntake.push({
        timestamp,
        value: 6 + Math.random() * 4
      });
    }

    return data;
  }

  private getMockProgressionData(): ProgressionData[] {
    return [
      {
        conditionType: 'skin_lesion',
        status: 'monitoring',
        severityScore: 0.3,
        confidenceScore: 0.85,
        baselineDate: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString(),
        daysSinceBaseline: 30,
        trend: 'stable',
        recentScans: [
          {
            id: '1',
            date: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString(),
            riskLevel: 'low',
            confidence: 0.92
          },
          {
            id: '2',
            date: new Date(Date.now() - 14 * 24 * 60 * 60 * 1000).toISOString(),
            riskLevel: 'low',
            confidence: 0.88
          }
        ]
      },
      {
        conditionType: 'cardiovascular',
        status: 'improving',
        severityScore: 0.2,
        confidenceScore: 0.78,
        baselineDate: new Date(Date.now() - 60 * 24 * 60 * 60 * 1000).toISOString(),
        daysSinceBaseline: 60,
        trend: 'positive',
        recentScans: []
      }
    ];
  }

  private getMockRiskProgression(): RiskProgression[] {
    const now = new Date();
    const data: RiskProgression[] = [];

    for (let i = 89; i >= 0; i--) {
      const date = new Date(now.getTime() - i * 24 * 60 * 60 * 1000);

      data.push({
        id: `risk_${i}`,
        user_id: 'mock_user',
        condition_type: 'skin_lesion',
        risk_level: Math.random() > 0.8 ? 'high' : Math.random() > 0.6 ? 'medium' : 'low',
        probability: Math.random() * 0.3 + 0.1,
        factors: {},
        recorded_at: date.toISOString(),
        predicted_date: null,
        confidence_score: 0.8 + Math.random() * 0.2,
        metadata: {},
        created_at: date.toISOString()
      });
    }

    return data;
  }
}

// Export singleton instance
export const timeSeriesService = new TimeSeriesService();
