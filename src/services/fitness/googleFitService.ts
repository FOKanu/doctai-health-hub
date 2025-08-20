import { supabase } from '@/integrations/supabase/client';

export interface GoogleFitConfig {
  clientId: string;
  clientSecret: string;
  redirectUri: string;
  scopes: string[];
}

export interface GoogleFitMetrics {
  heartRate: Array<{ timestamp: string; value: number }>;
  steps: Array<{ timestamp: string; value: number }>;
  calories: Array<{ timestamp: string; value: number }>;
  distance: Array<{ timestamp: string; value: number }>;
  sleep: Array<{ timestamp: string; duration: number; quality: string }>;
  weight: Array<{ timestamp: string; value: number }>;
  bloodPressure: Array<{ timestamp: string; systolic: number; diastolic: number }>;
  oxygenSaturation: Array<{ timestamp: string; value: number }>;
}

export interface GoogleFitAuthResponse {
  access_token: string;
  refresh_token: string;
  expires_in: number;
  token_type: string;
}

export class GoogleFitService {
  private config: GoogleFitConfig;
  private accessToken: string | null = null;
  private refreshToken: string | null = null;

  constructor(config: GoogleFitConfig) {
    this.config = config;
  }

  /**
   * Initialize Google Fit authentication
   */
  async initialize(): Promise<string> {
    const authUrl = `https://accounts.google.com/o/oauth2/v2/auth?` +
      `client_id=${this.config.clientId}&` +
      `redirect_uri=${encodeURIComponent(this.config.redirectUri)}&` +
      `scope=${encodeURIComponent(this.config.scopes.join(' '))}&` +
      `response_type=code&` +
      `access_type=offline&` +
      `prompt=consent`;

    return authUrl;
  }

  /**
   * Exchange authorization code for access token
   */
  async exchangeCodeForToken(code: string): Promise<GoogleFitAuthResponse> {
    const response = await fetch('https://oauth2.googleapis.com/token', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: new URLSearchParams({
        client_id: this.config.clientId,
        client_secret: this.config.clientSecret,
        code,
        grant_type: 'authorization_code',
        redirect_uri: this.config.redirectUri,
      }),
    });

    if (!response.ok) {
      throw new Error(`Failed to exchange code for token: ${response.statusText}`);
    }

    const data = await response.json();
    this.accessToken = data.access_token;
    this.refreshToken = data.refresh_token;

    return data;
  }

  /**
   * Refresh access token
   */
  async refreshAccessToken(): Promise<string> {
    if (!this.refreshToken) {
      throw new Error('No refresh token available');
    }

    const response = await fetch('https://oauth2.googleapis.com/token', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: new URLSearchParams({
        client_id: this.config.clientId,
        client_secret: this.config.clientSecret,
        refresh_token: this.refreshToken,
        grant_type: 'refresh_token',
      }),
    });

    if (!response.ok) {
      throw new Error(`Failed to refresh token: ${response.statusText}`);
    }

    const data = await response.json();
    this.accessToken = data.access_token;

    return data.access_token;
  }

  /**
   * Get heart rate data from Google Fit
   */
  async getHeartRateData(startTime: string, endTime: string): Promise<Array<{ timestamp: string; value: number }>> {
    const token = await this.getValidToken();

    const response = await fetch(
      `https://www.googleapis.com/fitness/v1/users/me/dataset:aggregate`,
      {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          aggregateBy: [{
            dataTypeName: 'com.google.heart_rate.bpm',
            dataSourceId: 'derived:com.google.heart_rate.bpm:com.google.android.gms:merge_heart_rate_bpm'
          }],
          bucketByTime: { durationMillis: 86400000 }, // 24 hours
          startTimeMillis: new Date(startTime).getTime(),
          endTimeMillis: new Date(endTime).getTime(),
        }),
      }
    );

    if (!response.ok) {
      throw new Error(`Failed to fetch heart rate data: ${response.statusText}`);
    }

    const data = await response.json();
    return this.parseHeartRateData(data);
  }

  /**
   * Get steps data from Google Fit
   */
  async getStepsData(startTime: string, endTime: string): Promise<Array<{ timestamp: string; value: number }>> {
    const token = await this.getValidToken();

    const response = await fetch(
      `https://www.googleapis.com/fitness/v1/users/me/dataset:aggregate`,
      {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          aggregateBy: [{
            dataTypeName: 'com.google.step_count.delta',
            dataSourceId: 'derived:com.google.step_count.delta:com.google.android.gms:estimated_steps'
          }],
          bucketByTime: { durationMillis: 86400000 }, // 24 hours
          startTimeMillis: new Date(startTime).getTime(),
          endTimeMillis: new Date(endTime).getTime(),
        }),
      }
    );

    if (!response.ok) {
      throw new Error(`Failed to fetch steps data: ${response.statusText}`);
    }

    const data = await response.json();
    return this.parseStepsData(data);
  }

  /**
   * Get sleep data from Google Fit
   */
  async getSleepData(startTime: string, endTime: string): Promise<Array<{ timestamp: string; duration: number; quality: string }>> {
    const token = await this.getValidToken();

    const response = await fetch(
      `https://www.googleapis.com/fitness/v1/users/me/sessions`,
      {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${token}`,
        },
        body: JSON.stringify({
          startTime: new Date(startTime).toISOString(),
          endTime: new Date(endTime).toISOString(),
          activityTypes: ['sleep'],
        }),
      }
    );

    if (!response.ok) {
      throw new Error(`Failed to fetch sleep data: ${response.statusText}`);
    }

    const data = await response.json();
    return this.parseSleepData(data);
  }

  /**
   * Sync all Google Fit data to Supabase
   */
  async syncToSupabase(userId: string, startTime: string, endTime: string): Promise<void> {
    try {
      // Fetch all metrics
      const [heartRate, steps, sleep] = await Promise.all([
        this.getHeartRateData(startTime, endTime),
        this.getStepsData(startTime, endTime),
        this.getSleepData(startTime, endTime),
      ]);

      // Sync heart rate data
      for (const record of heartRate) {
        await supabase.from('health_metrics_timeseries').upsert({
          user_id: userId,
          metric_type: 'heart_rate',
          value: { value: record.value },
          recorded_at: record.timestamp,
          device_source: 'google_fit',
          accuracy_score: 0.95,
        });
      }

      // Sync steps data
      for (const record of steps) {
        await supabase.from('health_metrics_timeseries').upsert({
          user_id: userId,
          metric_type: 'steps',
          value: { count: record.value },
          recorded_at: record.timestamp,
          device_source: 'google_fit',
          accuracy_score: 0.90,
        });
      }

      // Sync sleep data
      for (const record of sleep) {
        await supabase.from('sleep_metrics').upsert({
          user_id: userId,
          sleep_date: new Date(record.timestamp).toISOString().split('T')[0],
          total_duration: record.duration,
          sleep_efficiency: this.calculateSleepEfficiency(record.quality),
          device_source: 'google_fit',
        });
      }

      console.log('Google Fit data synced successfully');
    } catch (error) {
      console.error('Error syncing Google Fit data:', error);
      throw error;
    }
  }

  /**
   * Get valid access token (refresh if needed)
   */
  private async getValidToken(): Promise<string> {
    if (!this.accessToken) {
      throw new Error('No access token available. Please authenticate first.');
    }
    return this.accessToken;
  }

  /**
   * Parse heart rate data from Google Fit response
   */
  private parseHeartRateData(data: any): Array<{ timestamp: string; value: number }> {
    const result: Array<{ timestamp: string; value: number }> = [];

    if (data.bucket) {
      data.bucket.forEach((bucket: any) => {
        if (bucket.dataset && bucket.dataset[0].point) {
          bucket.dataset[0].point.forEach((point: any) => {
            result.push({
              timestamp: new Date(parseInt(point.startTimeNanos) / 1000000).toISOString(),
              value: point.value[0].fpVal || point.value[0].intVal,
            });
          });
        }
      });
    }

    return result;
  }

  /**
   * Parse steps data from Google Fit response
   */
  private parseStepsData(data: any): Array<{ timestamp: string; value: number }> {
    const result: Array<{ timestamp: string; value: number }> = [];

    if (data.bucket) {
      data.bucket.forEach((bucket: any) => {
        if (bucket.dataset && bucket.dataset[0].point) {
          bucket.dataset[0].point.forEach((point: any) => {
            result.push({
              timestamp: new Date(parseInt(point.startTimeNanos) / 1000000).toISOString(),
              value: point.value[0].intVal,
            });
          });
        }
      });
    }

    return result;
  }

  /**
   * Parse sleep data from Google Fit response
   */
  private parseSleepData(data: any): Array<{ timestamp: string; duration: number; quality: string }> {
    const result: Array<{ timestamp: string; duration: number; quality: string }> = [];

    if (data.session) {
      data.session.forEach((session: any) => {
        const duration = (parseInt(session.endTimeMillis) - parseInt(session.startTimeMillis)) / (1000 * 60 * 60); // hours
        result.push({
          timestamp: new Date(parseInt(session.startTimeMillis)).toISOString(),
          duration,
          quality: session.activityType || 'unknown',
        });
      });
    }

    return result;
  }

  /**
   * Calculate sleep efficiency based on quality
   */
  private calculateSleepEfficiency(quality: string): number {
    const qualityMap: Record<string, number> = {
      'deep': 0.9,
      'light': 0.7,
      'rem': 0.8,
      'awake': 0.3,
      'unknown': 0.6,
    };

    return qualityMap[quality] || 0.6;
  }
}
