/**
 * Extended Health Metrics Service
 * Handles comprehensive health monitoring across multiple domains
 */

import { supabase } from '@/integrations/supabase/client';
import type {
  CardiovascularMetrics,
  RespiratoryMetrics,
  MetabolicMetrics,
  SleepMetrics,
  FitnessMetrics,
  MentalHealthMetrics,
  HormonalMetrics,
  CardiovascularTrends,
  SleepQualitySummary,
  MetabolicHealthSummary
} from '@/integrations/supabase/types';

export interface ExtendedMetricsQueryParams {
  userId: string;
  startDate?: string;
  endDate?: string;
  limit?: number;
}

export interface HealthDomainSummary {
  cardiovascular: {
    avgHeartRate: number;
    avgBloodPressure: { systolic: number; diastolic: number };
    hrvTrend: 'improving' | 'stable' | 'declining';
    riskLevel: 'low' | 'medium' | 'high';
  };
  respiratory: {
    avgOxygenSaturation: number;
    avgRespiratoryRate: number;
    lungFunction: 'excellent' | 'good' | 'fair' | 'poor';
  };
  metabolic: {
    avgGlucose: number;
    avgHbA1c: number;
    cholesterolStatus: 'optimal' | 'borderline' | 'high';
    insulinSensitivity: 'excellent' | 'good' | 'fair' | 'poor';
  };
  sleep: {
    avgDuration: number;
    avgEfficiency: number;
    qualityTrend: 'improving' | 'stable' | 'declining';
  };
  fitness: {
    avgSteps: number;
    avgActiveMinutes: number;
    vo2Max: number;
    fitnessLevel: 'excellent' | 'good' | 'fair' | 'poor';
  };
  mentalHealth: {
    avgMoodScore: number;
    avgStressLevel: number;
    cognitivePerformance: 'excellent' | 'good' | 'fair' | 'poor';
  };
  hormonal: {
    thyroidStatus: 'normal' | 'hypothyroid' | 'hyperthyroid';
    cortisolPattern: 'normal' | 'elevated' | 'low';
    hormonalBalance: 'optimal' | 'suboptimal' | 'imbalanced';
  };
}

export class ExtendedHealthMetricsService {
  /**
   * Get comprehensive health summary across all domains
   */
  async getHealthSummary(params: ExtendedMetricsQueryParams): Promise<HealthDomainSummary> {
    try {
      const [cardiovascular, respiratory, metabolic, sleep, fitness, mentalHealth, hormonal] = await Promise.all([
        this.getCardiovascularSummary(params),
        this.getRespiratorySummary(params),
        this.getMetabolicSummary(params),
        this.getSleepSummary(params),
        this.getFitnessSummary(params),
        this.getMentalHealthSummary(params),
        this.getHormonalSummary(params)
      ]);

      return {
        cardiovascular,
        respiratory,
        metabolic,
        sleep,
        fitness,
        mentalHealth,
        hormonal
      };
    } catch (error) {
      console.error('Error fetching health summary:', error);
      return this.getMockHealthSummary();
    }
  }

  /**
   * Cardiovascular Health Metrics
   */
  async getCardiovascularMetrics(params: ExtendedMetricsQueryParams): Promise<CardiovascularMetrics[]> {
    try {
      const { data, error } = await supabase
        .from('cardiovascular_metrics')
        .select('*')
        .eq('user_id', params.userId)
        .gte('recorded_at', params.startDate || new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString())
        .lte('recorded_at', params.endDate || new Date().toISOString())
        .order('recorded_at', { ascending: true });

      if (error) throw error;
      return data || [];
    } catch (error) {
      console.error('Error fetching cardiovascular metrics:', error);
      return this.getMockCardiovascularMetrics();
    }
  }

  async getCardiovascularTrends(params: ExtendedMetricsQueryParams): Promise<CardiovascularTrends[]> {
    try {
      const { data, error } = await supabase
        .rpc('get_cardiovascular_trends', {
          p_user_id: params.userId,
          p_days: 30
        });

      if (error) throw error;
      return data || [];
    } catch (error) {
      console.error('Error fetching cardiovascular trends:', error);
      return this.getMockCardiovascularTrends();
    }
  }

  /**
   * Respiratory Health Metrics
   */
  async getRespiratoryMetrics(params: ExtendedMetricsQueryParams): Promise<RespiratoryMetrics[]> {
    try {
      const { data, error } = await supabase
        .from('respiratory_metrics')
        .select('*')
        .eq('user_id', params.userId)
        .gte('recorded_at', params.startDate || new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString())
        .lte('recorded_at', params.endDate || new Date().toISOString())
        .order('recorded_at', { ascending: true });

      if (error) throw error;
      return data || [];
    } catch (error) {
      console.error('Error fetching respiratory metrics:', error);
      return this.getMockRespiratoryMetrics();
    }
  }

  /**
   * Metabolic Health Metrics
   */
  async getMetabolicMetrics(params: ExtendedMetricsQueryParams): Promise<MetabolicMetrics[]> {
    try {
      const { data, error } = await supabase
        .from('metabolic_metrics')
        .select('*')
        .eq('user_id', params.userId)
        .gte('recorded_at', params.startDate || new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString())
        .lte('recorded_at', params.endDate || new Date().toISOString())
        .order('recorded_at', { ascending: true });

      if (error) throw error;
      return data || [];
    } catch (error) {
      console.error('Error fetching metabolic metrics:', error);
      return this.getMockMetabolicMetrics();
    }
  }

  async getMetabolicHealthSummary(params: ExtendedMetricsQueryParams): Promise<MetabolicHealthSummary | null> {
    try {
      const { data, error } = await supabase
        .rpc('get_metabolic_health_summary', {
          p_user_id: params.userId,
          p_days: 30
        });

      if (error) throw error;
      return data?.[0] || null;
    } catch (error) {
      console.error('Error fetching metabolic health summary:', error);
      return this.getMockMetabolicHealthSummary();
    }
  }

  /**
   * Sleep Quality Metrics
   */
  async getSleepMetrics(params: ExtendedMetricsQueryParams): Promise<SleepMetrics[]> {
    try {
      const { data, error } = await supabase
        .from('sleep_metrics')
        .select('*')
        .eq('user_id', params.userId)
        .gte('sleep_date', params.startDate || new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString().split('T')[0])
        .lte('sleep_date', params.endDate || new Date().toISOString().split('T')[0])
        .order('sleep_date', { ascending: true });

      if (error) throw error;
      return data || [];
    } catch (error) {
      console.error('Error fetching sleep metrics:', error);
      return this.getMockSleepMetrics();
    }
  }

  async getSleepQualitySummary(params: ExtendedMetricsQueryParams): Promise<SleepQualitySummary | null> {
    try {
      const { data, error } = await supabase
        .rpc('get_sleep_quality_summary', {
          p_user_id: params.userId,
          p_days: 30
        });

      if (error) throw error;
      return data?.[0] || null;
    } catch (error) {
      console.error('Error fetching sleep quality summary:', error);
      return this.getMockSleepQualitySummary();
    }
  }

  /**
   * Fitness & Activity Metrics
   */
  async getFitnessMetrics(params: ExtendedMetricsQueryParams): Promise<FitnessMetrics[]> {
    try {
      const { data, error } = await supabase
        .from('fitness_metrics')
        .select('*')
        .eq('user_id', params.userId)
        .gte('activity_date', params.startDate || new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString().split('T')[0])
        .lte('activity_date', params.endDate || new Date().toISOString().split('T')[0])
        .order('activity_date', { ascending: true });

      if (error) throw error;
      return data || [];
    } catch (error) {
      console.error('Error fetching fitness metrics:', error);
      return this.getMockFitnessMetrics();
    }
  }

  /**
   * Mental Health Metrics
   */
  async getMentalHealthMetrics(params: ExtendedMetricsQueryParams): Promise<MentalHealthMetrics[]> {
    try {
      const { data, error } = await supabase
        .from('mental_health_metrics')
        .select('*')
        .eq('user_id', params.userId)
        .gte('recorded_at', params.startDate || new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString())
        .lte('recorded_at', params.endDate || new Date().toISOString())
        .order('recorded_at', { ascending: true });

      if (error) throw error;
      return data || [];
    } catch (error) {
      console.error('Error fetching mental health metrics:', error);
      return this.getMockMentalHealthMetrics();
    }
  }

  /**
   * Hormonal Health Metrics
   */
  async getHormonalMetrics(params: ExtendedMetricsQueryParams): Promise<HormonalMetrics[]> {
    try {
      const { data, error } = await supabase
        .from('hormonal_metrics')
        .select('*')
        .eq('user_id', params.userId)
        .gte('recorded_at', params.startDate || new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString())
        .lte('recorded_at', params.endDate || new Date().toISOString())
        .order('recorded_at', { ascending: true });

      if (error) throw error;
      return data || [];
    } catch (error) {
      console.error('Error fetching hormonal metrics:', error);
      return this.getMockHormonalMetrics();
    }
  }

  /**
   * Save new health metrics
   */
  async saveCardiovascularMetrics(userId: string, metrics: Partial<CardiovascularMetrics>): Promise<void> {
    try {
      const { error } = await supabase
        .from('cardiovascular_metrics')
        .insert({
          user_id: userId,
          ...metrics,
          recorded_at: metrics.recorded_at || new Date().toISOString()
        });

      if (error) throw error;
    } catch (error) {
      console.error('Error saving cardiovascular metrics:', error);
      throw error;
    }
  }

  async saveSleepMetrics(userId: string, metrics: Partial<SleepMetrics>): Promise<void> {
    try {
      const { error } = await supabase
        .from('sleep_metrics')
        .insert({
          user_id: userId,
          ...metrics,
          sleep_date: metrics.sleep_date || new Date().toISOString().split('T')[0]
        });

      if (error) throw error;
    } catch (error) {
      console.error('Error saving sleep metrics:', error);
      throw error;
    }
  }

  async saveFitnessMetrics(userId: string, metrics: Partial<FitnessMetrics>): Promise<void> {
    try {
      const { error } = await supabase
        .from('fitness_metrics')
        .insert({
          user_id: userId,
          ...metrics,
          activity_date: metrics.activity_date || new Date().toISOString().split('T')[0]
        });

      if (error) throw error;
    } catch (error) {
      console.error('Error saving fitness metrics:', error);
      throw error;
    }
  }

  // Private helper methods for summaries
  private async getCardiovascularSummary(params: ExtendedMetricsQueryParams) {
    const metrics = await this.getCardiovascularMetrics(params);
    if (metrics.length === 0) return this.getMockCardiovascularSummary();

    const avgHeartRate = metrics.reduce((sum, m) => sum + (m.heart_rate_resting || 0), 0) / metrics.length;
    const avgSystolic = metrics.reduce((sum, m) => sum + (m.blood_pressure_systolic || 0), 0) / metrics.length;
    const avgDiastolic = metrics.reduce((sum, m) => sum + (m.blood_pressure_diastolic || 0), 0) / metrics.length;

    return {
      avgHeartRate: Math.round(avgHeartRate),
      avgBloodPressure: { systolic: Math.round(avgSystolic), diastolic: Math.round(avgDiastolic) },
      hrvTrend: 'stable' as const,
      riskLevel: avgHeartRate > 100 || avgSystolic > 140 ? 'high' : avgHeartRate > 80 || avgSystolic > 120 ? 'medium' : 'low'
    };
  }

  private async getRespiratorySummary(params: ExtendedMetricsQueryParams) {
    const metrics = await this.getRespiratoryMetrics(params);
    if (metrics.length === 0) return this.getMockRespiratorySummary();

    const avgO2 = metrics.reduce((sum, m) => sum + (m.oxygen_saturation || 0), 0) / metrics.length;
    const avgRR = metrics.reduce((sum, m) => sum + (m.respiratory_rate || 0), 0) / metrics.length;

    return {
      avgOxygenSaturation: Math.round(avgO2 * 100) / 100,
      avgRespiratoryRate: Math.round(avgRR),
      lungFunction: avgO2 > 95 ? 'excellent' : avgO2 > 90 ? 'good' : avgO2 > 85 ? 'fair' : 'poor'
    };
  }

  private async getMetabolicSummary(params: ExtendedMetricsQueryParams) {
    const summary = await this.getMetabolicHealthSummary(params);
    if (!summary) return this.getMockMetabolicSummary();

    return {
      avgGlucose: summary.avg_glucose_fasting,
      avgHbA1c: summary.avg_hba1c,
      cholesterolStatus: summary.cholesterol_ratio < 3.5 ? 'optimal' : summary.cholesterol_ratio < 5.0 ? 'borderline' : 'high',
      insulinSensitivity: summary.insulin_sensitivity_status
    };
  }

  private async getSleepSummary(params: ExtendedMetricsQueryParams) {
    const summary = await this.getSleepQualitySummary(params);
    if (!summary) return this.getMockSleepSummary();

    return {
      avgDuration: summary.avg_duration,
      avgEfficiency: summary.avg_efficiency,
      qualityTrend: summary.quality_trend === 'excellent' || summary.quality_trend === 'good' ? 'improving' : 'stable'
    };
  }

  private async getFitnessSummary(params: ExtendedMetricsQueryParams) {
    const metrics = await this.getFitnessMetrics(params);
    if (metrics.length === 0) return this.getMockFitnessSummary();

    const avgSteps = metrics.reduce((sum, m) => sum + (m.steps_count || 0), 0) / metrics.length;
    const avgActiveMinutes = metrics.reduce((sum, m) => sum + (m.active_minutes || 0), 0) / metrics.length;
    const avgVo2Max = metrics.reduce((sum, m) => sum + (m.vo2_max || 0), 0) / metrics.length;

    return {
      avgSteps: Math.round(avgSteps),
      avgActiveMinutes: Math.round(avgActiveMinutes),
      vo2Max: Math.round(avgVo2Max * 10) / 10,
      fitnessLevel: avgVo2Max > 50 ? 'excellent' : avgVo2Max > 40 ? 'good' : avgVo2Max > 30 ? 'fair' : 'poor'
    };
  }

  private async getMentalHealthSummary(params: ExtendedMetricsQueryParams) {
    const metrics = await this.getMentalHealthMetrics(params);
    if (metrics.length === 0) return this.getMockMentalHealthSummary();

    const avgMood = metrics.reduce((sum, m) => sum + (m.mood_score || 0), 0) / metrics.length;
    const avgStress = metrics.reduce((sum, m) => sum + (m.stress_level || 0), 0) / metrics.length;
    const avgMemory = metrics.reduce((sum, m) => sum + (m.memory_score || 0), 0) / metrics.length;

    return {
      avgMoodScore: Math.round(avgMood * 10) / 10,
      avgStressLevel: Math.round(avgStress * 10) / 10,
      cognitivePerformance: avgMemory > 0.8 ? 'excellent' : avgMemory > 0.6 ? 'good' : avgMemory > 0.4 ? 'fair' : 'poor'
    };
  }

  private async getHormonalSummary(params: ExtendedMetricsQueryParams) {
    const metrics = await this.getHormonalMetrics(params);
    if (metrics.length === 0) return this.getMockHormonalSummary();

    const avgTSH = metrics.reduce((sum, m) => sum + (m.tsh || 0), 0) / metrics.length;
    const avgCortisol = metrics.reduce((sum, m) => sum + (m.cortisol_morning || 0), 0) / metrics.length;

    return {
      thyroidStatus: avgTSH > 4.5 ? 'hypothyroid' : avgTSH < 0.4 ? 'hyperthyroid' : 'normal',
      cortisolPattern: avgCortisol > 25 ? 'elevated' : avgCortisol < 10 ? 'low' : 'normal',
      hormonalBalance: 'optimal' as const
    };
  }

  // Mock data methods for development
  private getMockHealthSummary(): HealthDomainSummary {
    return {
      cardiovascular: { avgHeartRate: 72, avgBloodPressure: { systolic: 120, diastolic: 80 }, hrvTrend: 'stable', riskLevel: 'low' },
      respiratory: { avgOxygenSaturation: 98.5, avgRespiratoryRate: 16, lungFunction: 'excellent' },
      metabolic: { avgGlucose: 95, avgHbA1c: 5.2, cholesterolStatus: 'optimal', insulinSensitivity: 'excellent' },
      sleep: { avgDuration: 7.5, avgEfficiency: 85, qualityTrend: 'improving' },
      fitness: { avgSteps: 8500, avgActiveMinutes: 45, vo2Max: 42.5, fitnessLevel: 'good' },
      mentalHealth: { avgMoodScore: 7.5, avgStressLevel: 3.2, cognitivePerformance: 'good' },
      hormonal: { thyroidStatus: 'normal', cortisolPattern: 'normal', hormonalBalance: 'optimal' }
    };
  }

  private getMockCardiovascularMetrics(): CardiovascularMetrics[] {
    return [
      {
        id: '1',
        user_id: 'mock_user',
        heart_rate_resting: 72,
        heart_rate_variability: 45.2,
        blood_pressure_systolic: 120,
        blood_pressure_diastolic: 80,
        recorded_at: new Date().toISOString(),
        created_at: new Date().toISOString()
      }
    ];
  }

  private getMockCardiovascularTrends(): CardiovascularTrends[] {
    return [
      {
        recorded_at: new Date().toISOString(),
        heart_rate_resting: 72,
        heart_rate_variability: 45.2,
        blood_pressure_systolic: 120,
        blood_pressure_diastolic: 80,
        trend_direction: 'stable'
      }
    ];
  }

  private getMockRespiratoryMetrics(): RespiratoryMetrics[] {
    return [
      {
        id: '1',
        user_id: 'mock_user',
        respiratory_rate: 16,
        oxygen_saturation: 98.5,
        recorded_at: new Date().toISOString(),
        created_at: new Date().toISOString()
      }
    ];
  }

  private getMockMetabolicMetrics(): MetabolicMetrics[] {
    return [
      {
        id: '1',
        user_id: 'mock_user',
        blood_glucose_fasting: 95,
        hba1c: 5.2,
        cholesterol_total: 180,
        cholesterol_hdl: 60,
        cholesterol_ldl: 100,
        recorded_at: new Date().toISOString(),
        created_at: new Date().toISOString()
      }
    ];
  }

  private getMockSleepMetrics(): SleepMetrics[] {
    return [
      {
        id: '1',
        user_id: 'mock_user',
        sleep_date: new Date().toISOString().split('T')[0],
        total_duration: 7.5,
        sleep_efficiency: 85,
        created_at: new Date().toISOString()
      }
    ];
  }

  private getMockFitnessMetrics(): FitnessMetrics[] {
    return [
      {
        id: '1',
        user_id: 'mock_user',
        activity_date: new Date().toISOString().split('T')[0],
        steps_count: 8500,
        active_minutes: 45,
        vo2_max: 42.5,
        created_at: new Date().toISOString()
      }
    ];
  }

  private getMockMentalHealthMetrics(): MentalHealthMetrics[] {
    return [
      {
        id: '1',
        user_id: 'mock_user',
        mood_score: 7,
        stress_level: 3,
        memory_score: 0.85,
        recorded_at: new Date().toISOString(),
        created_at: new Date().toISOString()
      }
    ];
  }

  private getMockHormonalMetrics(): HormonalMetrics[] {
    return [
      {
        id: '1',
        user_id: 'mock_user',
        tsh: 2.5,
        cortisol_morning: 15.2,
        recorded_at: new Date().toISOString(),
        created_at: new Date().toISOString()
      }
    ];
  }

  private getMockCardiovascularSummary() {
    return { avgHeartRate: 72, avgBloodPressure: { systolic: 120, diastolic: 80 }, hrvTrend: 'stable' as const, riskLevel: 'low' as const };
  }

  private getMockRespiratorySummary() {
    return { avgOxygenSaturation: 98.5, avgRespiratoryRate: 16, lungFunction: 'excellent' as const };
  }

  private getMockMetabolicSummary() {
    return { avgGlucose: 95, avgHbA1c: 5.2, cholesterolStatus: 'optimal' as const, insulinSensitivity: 'excellent' as const };
  }

  private getMockSleepSummary() {
    return { avgDuration: 7.5, avgEfficiency: 85, qualityTrend: 'improving' as const };
  }

  private getMockFitnessSummary() {
    return { avgSteps: 8500, avgActiveMinutes: 45, vo2Max: 42.5, fitnessLevel: 'good' as const };
  }

  private getMockMentalHealthSummary() {
    return { avgMoodScore: 7.5, avgStressLevel: 3.2, cognitivePerformance: 'good' as const };
  }

  private getMockHormonalSummary() {
    return { thyroidStatus: 'normal' as const, cortisolPattern: 'normal' as const, hormonalBalance: 'optimal' as const };
  }

  private getMockCardiovascularTrends(): CardiovascularTrends[] {
    return [
      {
        recorded_at: new Date().toISOString(),
        heart_rate_resting: 72,
        heart_rate_variability: 45.2,
        blood_pressure_systolic: 120,
        blood_pressure_diastolic: 80,
        trend_direction: 'stable'
      }
    ];
  }

  private getMockSleepQualitySummary(): SleepQualitySummary {
    return {
      avg_duration: 7.5,
      avg_efficiency: 85,
      avg_latency: 15,
      quality_trend: 'good',
      deep_sleep_percentage: 20,
      rem_sleep_percentage: 25
    };
  }

  private getMockMetabolicHealthSummary(): MetabolicHealthSummary {
    return {
      avg_glucose_fasting: 95,
      avg_hba1c: 5.2,
      cholesterol_ratio: 3.0,
      metabolic_risk: 'low',
      insulin_sensitivity_status: 'excellent'
    };
  }
}

export const extendedHealthMetricsService = new ExtendedHealthMetricsService();
