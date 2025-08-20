import { GoogleFitService, GoogleFitConfig } from './googleFitService';
import { FitbitService, FitbitConfig } from './fitbitService';
import { supabase } from '@/integrations/supabase/client';

export interface FitnessDevice {
  id: string;
  name: string;
  type: 'google_fit' | 'fitbit' | 'apple_health' | 'samsung_health';
  isConnected: boolean;
  lastSync: string | null;
  metrics: string[];
  userId: string;
}

export interface SyncResult {
  success: boolean;
  deviceId: string;
  metricsSynced: number;
  errors: string[];
  timestamp: string;
}

export interface HealthMetricsSummary {
  heartRate: {
    current: number;
    average: number;
    min: number;
    max: number;
    trend: 'increasing' | 'decreasing' | 'stable';
  };
  steps: {
    today: number;
    weekly: number;
    goal: number;
    progress: number;
  };
  sleep: {
    lastNight: number;
    average: number;
    quality: number;
    efficiency: number;
  };
  calories: {
    burned: number;
    goal: number;
    remaining: number;
  };
  weight: {
    current: number;
    trend: 'losing' | 'gaining' | 'stable';
    change: number;
  };
}

export class FitnessIntegrationService {
  private googleFitService: GoogleFitService | null = null;
  private fitbitService: FitbitService | null = null;
  private devices: Map<string, FitnessDevice> = new Map();

  constructor() {
    this.loadConnectedDevices();
  }

  /**
   * Initialize Google Fit integration
   */
  async initializeGoogleFit(config: GoogleFitConfig): Promise<string> {
    this.googleFitService = new GoogleFitService(config);
    return await this.googleFitService.initialize();
  }

  /**
   * Initialize Fitbit integration
   */
  async initializeFitbit(config: FitbitConfig): Promise<string> {
    this.fitbitService = new FitbitService(config);
    return await this.fitbitService.initialize();
  }

  /**
   * Connect Google Fit device
   */
  async connectGoogleFit(userId: string, authCode: string): Promise<FitnessDevice> {
    if (!this.googleFitService) {
      throw new Error('Google Fit service not initialized');
    }

    try {
      const authResponse = await this.googleFitService.exchangeCodeForToken(authCode);

      const device: FitnessDevice = {
        id: `google_fit_${userId}`,
        name: 'Google Fit',
        type: 'google_fit',
        isConnected: true,
        lastSync: null,
        metrics: ['heart_rate', 'steps', 'sleep', 'calories', 'distance'],
        userId,
      };

      await this.saveDevice(device);
      this.devices.set(device.id, device);

      return device;
    } catch (error) {
      console.error('Error connecting Google Fit:', error);
      throw error;
    }
  }

  /**
   * Connect Fitbit device
   */
  async connectFitbit(userId: string, authCode: string): Promise<FitnessDevice> {
    if (!this.fitbitService) {
      throw new Error('Fitbit service not initialized');
    }

    try {
      const authResponse = await this.fitbitService.exchangeCodeForToken(authCode);

      const device: FitnessDevice = {
        id: `fitbit_${userId}`,
        name: 'Fitbit',
        type: 'fitbit',
        isConnected: true,
        lastSync: null,
        metrics: ['heart_rate', 'steps', 'sleep', 'calories', 'weight', 'blood_pressure'],
        userId,
      };

      await this.saveDevice(device);
      this.devices.set(device.id, device);

      return device;
    } catch (error) {
      console.error('Error connecting Fitbit:', error);
      throw error;
    }
  }

  /**
   * Sync data from all connected devices
   */
  async syncAllDevices(userId: string, dateRange: { start: string; end: string }): Promise<SyncResult[]> {
    const results: SyncResult[] = [];
    const userDevices = Array.from(this.devices.values()).filter(d => d.userId === userId);

    for (const device of userDevices) {
      try {
        const result = await this.syncDevice(device, dateRange);
        results.push(result);
      } catch (error) {
        results.push({
          success: false,
          deviceId: device.id,
          metricsSynced: 0,
          errors: [error instanceof Error ? error.message : 'Unknown error'],
          timestamp: new Date().toISOString(),
        });
      }
    }

    return results;
  }

  /**
   * Sync data from a specific device
   */
  async syncDevice(device: FitnessDevice, dateRange: { start: string; end: string }): Promise<SyncResult> {
    const errors: string[] = [];
    let metricsSynced = 0;

    try {
      switch (device.type) {
        case 'google_fit':
          if (this.googleFitService) {
            await this.googleFitService.syncToSupabase(device.userId, dateRange.start, dateRange.end);
            metricsSynced = 5; // heart_rate, steps, sleep, calories, distance
          }
          break;

        case 'fitbit':
          if (this.fitbitService) {
            await this.fitbitService.syncToSupabase(device.userId, dateRange.start);
            metricsSynced = 6; // heart_rate, steps, sleep, weight, blood_pressure, calories
          }
          break;

        default:
          throw new Error(`Unsupported device type: ${device.type}`);
      }

      // Update last sync time
      device.lastSync = new Date().toISOString();
      await this.updateDevice(device);

      return {
        success: true,
        deviceId: device.id,
        metricsSynced,
        errors,
        timestamp: device.lastSync,
      };
    } catch (error) {
      errors.push(error instanceof Error ? error.message : 'Unknown error');
      return {
        success: false,
        deviceId: device.id,
        metricsSynced: 0,
        errors,
        timestamp: new Date().toISOString(),
      };
    }
  }

  /**
   * Get health metrics summary for a user
   */
  async getHealthMetricsSummary(userId: string): Promise<HealthMetricsSummary> {
    try {
      const today = new Date().toISOString().split('T')[0];
      const weekAgo = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString().split('T')[0];

      // Get heart rate data
      const heartRateData = await this.getHeartRateData(userId, weekAgo, today);
      const heartRate = this.calculateHeartRateSummary(heartRateData);

      // Get steps data
      const stepsData = await this.getStepsData(userId, today, today);
      const steps = this.calculateStepsSummary(stepsData);

      // Get sleep data
      const sleepData = await this.getSleepData(userId, today, today);
      const sleep = this.calculateSleepSummary(sleepData);

      // Get calories data
      const caloriesData = await this.getCaloriesData(userId, today, today);
      const calories = this.calculateCaloriesSummary(caloriesData);

      // Get weight data
      const weightData = await this.getWeightData(userId, weekAgo, today);
      const weight = this.calculateWeightSummary(weightData);

      return {
        heartRate,
        steps,
        sleep,
        calories,
        weight,
      };
    } catch (error) {
      console.error('Error getting health metrics summary:', error);
      return this.getDefaultHealthMetricsSummary();
    }
  }

  /**
   * Get connected devices for a user
   */
  getConnectedDevices(userId: string): FitnessDevice[] {
    return Array.from(this.devices.values()).filter(d => d.userId === userId);
  }

  /**
   * Disconnect a device
   */
  async disconnectDevice(deviceId: string): Promise<void> {
    const device = this.devices.get(deviceId);
    if (device) {
      device.isConnected = false;
      await this.updateDevice(device);
      this.devices.delete(deviceId);
    }
  }

  /**
   * Load connected devices from database
   */
  private async loadConnectedDevices(): Promise<void> {
    try {
      const { data, error } = await supabase
        .from('fitness_devices')
        .select('*')
        .eq('is_connected', true);

      if (error) throw error;

      if (data) {
        data.forEach(device => {
          this.devices.set(device.id, {
            id: device.id,
            name: device.name,
            type: device.type,
            isConnected: device.is_connected,
            lastSync: device.last_sync,
            metrics: device.metrics || [],
            userId: device.user_id,
          });
        });
      }
    } catch (error) {
      console.error('Error loading connected devices:', error);
    }
  }

  /**
   * Save device to database
   */
  private async saveDevice(device: FitnessDevice): Promise<void> {
    try {
      const { error } = await supabase
        .from('fitness_devices')
        .upsert({
          id: device.id,
          name: device.name,
          type: device.type,
          is_connected: device.isConnected,
          last_sync: device.lastSync,
          metrics: device.metrics,
          user_id: device.userId,
        });

      if (error) throw error;
    } catch (error) {
      console.error('Error saving device:', error);
      throw error;
    }
  }

  /**
   * Update device in database
   */
  private async updateDevice(device: FitnessDevice): Promise<void> {
    try {
      const { error } = await supabase
        .from('fitness_devices')
        .update({
          is_connected: device.isConnected,
          last_sync: device.lastSync,
          metrics: device.metrics,
        })
        .eq('id', device.id);

      if (error) throw error;
    } catch (error) {
      console.error('Error updating device:', error);
      throw error;
    }
  }

  // Helper methods for data retrieval and calculation
  private async getHeartRateData(userId: string, startDate: string, endDate: string) {
    const { data } = await supabase
      .from('health_metrics_timeseries')
      .select('*')
      .eq('user_id', userId)
      .eq('metric_type', 'heart_rate')
      .gte('recorded_at', startDate)
      .lte('recorded_at', endDate);

    return data || [];
  }

  private async getStepsData(userId: string, startDate: string, endDate: string) {
    const { data } = await supabase
      .from('health_metrics_timeseries')
      .select('*')
      .eq('user_id', userId)
      .eq('metric_type', 'steps')
      .gte('recorded_at', startDate)
      .lte('recorded_at', endDate);

    return data || [];
  }

  private async getSleepData(userId: string, startDate: string, endDate: string) {
    const { data } = await supabase
      .from('sleep_metrics')
      .select('*')
      .eq('user_id', userId)
      .gte('sleep_date', startDate)
      .lte('sleep_date', endDate);

    return data || [];
  }

  private async getCaloriesData(userId: string, startDate: string, endDate: string) {
    const { data } = await supabase
      .from('health_metrics_timeseries')
      .select('*')
      .eq('user_id', userId)
      .eq('metric_type', 'calories')
      .gte('recorded_at', startDate)
      .lte('recorded_at', endDate);

    return data || [];
  }

  private async getWeightData(userId: string, startDate: string, endDate: string) {
    const { data } = await supabase
      .from('health_metrics_timeseries')
      .select('*')
      .eq('user_id', userId)
      .eq('metric_type', 'weight')
      .gte('recorded_at', startDate)
      .lte('recorded_at', endDate);

    return data || [];
  }

  // Calculation helper methods
  private calculateHeartRateSummary(data: any[]): HealthMetricsSummary['heartRate'] {
    if (data.length === 0) {
      return { current: 0, average: 0, min: 0, max: 0, trend: 'stable' };
    }

    const values = data.map(d => d.value?.value || 0).filter(v => v > 0);
    const current = values[values.length - 1] || 0;
    const average = values.reduce((sum, v) => sum + v, 0) / values.length;
    const min = Math.min(...values);
    const max = Math.max(...values);

    // Simple trend calculation
    const recent = values.slice(-5);
    const older = values.slice(-10, -5);
    const recentAvg = recent.reduce((sum, v) => sum + v, 0) / recent.length;
    const olderAvg = older.reduce((sum, v) => sum + v, 0) / older.length;

    let trend: 'increasing' | 'decreasing' | 'stable' = 'stable';
    if (recentAvg > olderAvg + 5) trend = 'increasing';
    else if (recentAvg < olderAvg - 5) trend = 'decreasing';

    return { current, average, min, max, trend };
  }

  private calculateStepsSummary(data: any[]): HealthMetricsSummary['steps'] {
    const today = data.reduce((sum, d) => sum + (d.value?.count || 0), 0);
    const goal = 10000; // Default goal
    const progress = Math.min((today / goal) * 100, 100);

    return {
      today,
      weekly: today * 7, // Simplified calculation
      goal,
      progress,
    };
  }

  private calculateSleepSummary(data: any[]): HealthMetricsSummary['sleep'] {
    if (data.length === 0) {
      return { lastNight: 0, average: 0, quality: 0, efficiency: 0 };
    }

    const lastNight = data[data.length - 1]?.total_duration || 0;
    const average = data.reduce((sum, d) => sum + (d.total_duration || 0), 0) / data.length;
    const quality = data[data.length - 1]?.sleep_quality_score || 0;
    const efficiency = data[data.length - 1]?.sleep_efficiency || 0;

    return { lastNight, average, quality, efficiency };
  }

  private calculateCaloriesSummary(data: any[]): HealthMetricsSummary['calories'] {
    const burned = data.reduce((sum, d) => sum + (d.value?.value || 0), 0);
    const goal = 2000; // Default goal
    const remaining = Math.max(goal - burned, 0);

    return { burned, goal, remaining };
  }

  private calculateWeightSummary(data: any[]): HealthMetricsSummary['weight'] {
    if (data.length === 0) {
      return { current: 0, trend: 'stable', change: 0 };
    }

    const current = data[data.length - 1]?.value?.value || 0;
    const previous = data[data.length - 2]?.value?.value || current;
    const change = current - previous;

    let trend: 'losing' | 'gaining' | 'stable' = 'stable';
    if (change < -0.5) trend = 'losing';
    else if (change > 0.5) trend = 'gaining';

    return { current, trend, change };
  }

  private getDefaultHealthMetricsSummary(): HealthMetricsSummary {
    return {
      heartRate: { current: 0, average: 0, min: 0, max: 0, trend: 'stable' },
      steps: { today: 0, weekly: 0, goal: 10000, progress: 0 },
      sleep: { lastNight: 0, average: 0, quality: 0, efficiency: 0 },
      calories: { burned: 0, goal: 2000, remaining: 2000 },
      weight: { current: 0, trend: 'stable', change: 0 },
    };
  }
}
