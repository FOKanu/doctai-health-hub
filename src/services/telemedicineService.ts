/**
 * Telemedicine Service
 * Handles remote health monitoring, virtual consultations, and healthcare provider management
 */

import { supabase } from '@/integrations/supabase/client';
import type {
  HealthcareProvider,
  TelemedicineConsultation,
  RemoteMonitoring,
  HealthAlert,
  Appointment,
  ProviderAvailability,
  PatientProviderRelationship,
  ConsultationRecording,
  AvailableAppointmentSlot,
  PatientHealthSummary,
  ConsultationStats
} from '@/integrations/supabase/types';

export interface TelemedicineQueryParams {
  userId: string;
  providerId?: string;
  startDate?: string;
  endDate?: string;
  status?: string;
  limit?: number;
}

export interface AppointmentBookingParams {
  patientId: string;
  providerId: string;
  appointmentType: string;
  scheduledAt: string;
  durationMinutes?: number;
  reason?: string;
  symptoms?: string[];
  isUrgent?: boolean;
}

export interface RemoteMonitoringParams {
  patientId: string;
  providerId?: string;
  monitoringType: string;
  deviceId?: string;
  deviceType?: string;
  monitoringFrequency?: string;
  alertThresholds?: any;
}

export interface HealthAlertParams {
  patientId: string;
  providerId?: string;
  alertType: string;
  alertLevel: 'low' | 'medium' | 'high' | 'critical';
  title: string;
  message: string;
  metricName?: string;
  metricValue?: any;
  thresholdValue?: any;
}

export class TelemedicineService {
  /**
   * Healthcare Provider Management
   */
  async getHealthcareProviders(specialty?: string): Promise<HealthcareProvider[]> {
    try {
      let query = supabase
        .from('healthcare_providers')
        .select('*')
        .eq('is_available', true)
        .eq('is_verified', true);

      if (specialty) {
        query = query.eq('specialty', specialty);
      }

      const { data, error } = await query.order('rating', { ascending: false });

      if (error) throw error;
      return data || [];
    } catch (error) {
      console.error('Error fetching healthcare providers:', error);
      return this.getMockHealthcareProviders();
    }
  }

  async getProviderById(providerId: string): Promise<HealthcareProvider | null> {
    try {
      const { data, error } = await supabase
        .from('healthcare_providers')
        .select('*')
        .eq('id', providerId)
        .single();

      if (error) throw error;
      return data;
    } catch (error) {
      console.error('Error fetching provider:', error);
      return null;
    }
  }

  async getProviderAvailability(providerId: string, date: string): Promise<AvailableAppointmentSlot[]> {
    try {
      const { data, error } = await supabase
        .rpc('get_available_appointment_slots', {
          p_provider_id: providerId,
          p_date: date,
          p_duration_minutes: 30
        });

      if (error) throw error;
      return data || [];
    } catch (error) {
      console.error('Error fetching provider availability:', error);
      return this.getMockAvailableSlots();
    }
  }

  /**
   * Appointment Management
   */
  async bookAppointment(params: AppointmentBookingParams): Promise<Appointment | null> {
    try {
      const { data, error } = await supabase
        .from('appointments')
        .insert({
          patient_id: params.patientId,
          provider_id: params.providerId,
          appointment_type: params.appointmentType,
          scheduled_at: params.scheduledAt,
          duration_minutes: params.durationMinutes || 30,
          reason: params.reason,
          symptoms: params.symptoms,
          is_urgent: params.isUrgent || false
        })
        .select()
        .single();

      if (error) throw error;
      return data;
    } catch (error) {
      console.error('Error booking appointment:', error);
      throw error;
    }
  }

  async getAppointments(params: TelemedicineQueryParams): Promise<Appointment[]> {
    try {
      let query = supabase
        .from('appointments')
        .select('*');

      if (params.userId) {
        query = query.eq('patient_id', params.userId);
      }

      if (params.providerId) {
        query = query.eq('provider_id', params.providerId);
      }

      if (params.status) {
        query = query.eq('status', params.status);
      }

      if (params.startDate) {
        query = query.gte('scheduled_at', params.startDate);
      }

      if (params.endDate) {
        query = query.lte('scheduled_at', params.endDate);
      }

      const { data, error } = await query.order('scheduled_at', { ascending: true });

      if (error) throw error;
      return data || [];
    } catch (error) {
      console.error('Error fetching appointments:', error);
      return this.getMockAppointments();
    }
  }

  async updateAppointmentStatus(appointmentId: string, status: string): Promise<void> {
    try {
      const { error } = await supabase
        .from('appointments')
        .update({ status })
        .eq('id', appointmentId);

      if (error) throw error;
    } catch (error) {
      console.error('Error updating appointment status:', error);
      throw error;
    }
  }

  /**
   * Telemedicine Consultations
   */
  async createConsultation(consultation: Partial<TelemedicineConsultation>): Promise<TelemedicineConsultation | null> {
    try {
      const { data, error } = await supabase
        .from('telemedicine_consultations')
        .insert(consultation)
        .select()
        .single();

      if (error) throw error;
      return data;
    } catch (error) {
      console.error('Error creating consultation:', error);
      throw error;
    }
  }

  async getConsultations(params: TelemedicineQueryParams): Promise<TelemedicineConsultation[]> {
    try {
      let query = supabase
        .from('telemedicine_consultations')
        .select('*');

      if (params.userId) {
        query = query.eq('patient_id', params.userId);
      }

      if (params.providerId) {
        query = query.eq('provider_id', params.providerId);
      }

      if (params.status) {
        query = query.eq('status', params.status);
      }

      const { data, error } = await query.order('scheduled_at', { ascending: false });

      if (error) throw error;
      return data || [];
    } catch (error) {
      console.error('Error fetching consultations:', error);
      return this.getMockConsultations();
    }
  }

  async updateConsultation(consultationId: string, updates: Partial<TelemedicineConsultation>): Promise<void> {
    try {
      const { error } = await supabase
        .from('telemedicine_consultations')
        .update(updates)
        .eq('id', consultationId);

      if (error) throw error;
    } catch (error) {
      console.error('Error updating consultation:', error);
      throw error;
    }
  }

  /**
   * Remote Health Monitoring
   */
  async startRemoteMonitoring(params: RemoteMonitoringParams): Promise<RemoteMonitoring | null> {
    try {
      const { data, error } = await supabase
        .from('remote_monitoring')
        .insert({
          patient_id: params.patientId,
          provider_id: params.providerId,
          monitoring_type: params.monitoringType,
          device_id: params.deviceId,
          device_type: params.deviceType,
          monitoring_frequency: params.monitoringFrequency,
          alert_thresholds: params.alertThresholds,
          start_date: new Date().toISOString()
        })
        .select()
        .single();

      if (error) throw error;
      return data;
    } catch (error) {
      console.error('Error starting remote monitoring:', error);
      throw error;
    }
  }

  async getRemoteMonitoring(params: TelemedicineQueryParams): Promise<RemoteMonitoring[]> {
    try {
      let query = supabase
        .from('remote_monitoring')
        .select('*');

      if (params.userId) {
        query = query.eq('patient_id', params.userId);
      }

      if (params.providerId) {
        query = query.eq('provider_id', params.providerId);
      }

      const { data, error } = await query.eq('is_active', true).order('created_at', { ascending: false });

      if (error) throw error;
      return data || [];
    } catch (error) {
      console.error('Error fetching remote monitoring:', error);
      return this.getMockRemoteMonitoring();
    }
  }

  async updateMonitoringReading(monitoringId: string, reading: any): Promise<void> {
    try {
      const { error } = await supabase
        .from('remote_monitoring')
        .update({
          last_reading_at: new Date().toISOString(),
          last_reading_value: reading
        })
        .eq('id', monitoringId);

      if (error) throw error;
    } catch (error) {
      console.error('Error updating monitoring reading:', error);
      throw error;
    }
  }

  async stopRemoteMonitoring(monitoringId: string): Promise<void> {
    try {
      const { error } = await supabase
        .from('remote_monitoring')
        .update({
          is_active: false,
          end_date: new Date().toISOString()
        })
        .eq('id', monitoringId);

      if (error) throw error;
    } catch (error) {
      console.error('Error stopping remote monitoring:', error);
      throw error;
    }
  }

  /**
   * Health Alerts
   */
  async createHealthAlert(params: HealthAlertParams): Promise<string | null> {
    try {
      const { data, error } = await supabase
        .rpc('create_health_alert', {
          p_patient_id: params.patientId,
          p_alert_type: params.alertType,
          p_alert_level: params.alertLevel,
          p_title: params.title,
          p_message: params.message,
          p_metric_name: params.metricName,
          p_metric_value: params.metricValue,
          p_threshold_value: params.thresholdValue
        });

      if (error) throw error;
      return data;
    } catch (error) {
      console.error('Error creating health alert:', error);
      throw error;
    }
  }

  async getHealthAlerts(params: TelemedicineQueryParams): Promise<HealthAlert[]> {
    try {
      let query = supabase
        .from('health_alerts')
        .select('*');

      if (params.userId) {
        query = query.eq('patient_id', params.userId);
      }

      if (params.providerId) {
        query = query.eq('provider_id', params.providerId);
      }

      const { data, error } = await query.order('created_at', { ascending: false });

      if (error) throw error;
      return data || [];
    } catch (error) {
      console.error('Error fetching health alerts:', error);
      return this.getMockHealthAlerts();
    }
  }

  async markAlertAsRead(alertId: string): Promise<void> {
    try {
      const { error } = await supabase
        .from('health_alerts')
        .update({
          is_read: true,
          acknowledged_at: new Date().toISOString()
        })
        .eq('id', alertId);

      if (error) throw error;
    } catch (error) {
      console.error('Error marking alert as read:', error);
      throw error;
    }
  }

  async getUnreadAlertsCount(userId: string): Promise<number> {
    try {
      const { count, error } = await supabase
        .from('health_alerts')
        .select('*', { count: 'exact', head: true })
        .eq('patient_id', userId)
        .eq('is_read', false);

      if (error) throw error;
      return count || 0;
    } catch (error) {
      console.error('Error getting unread alerts count:', error);
      return 0;
    }
  }

  /**
   * Patient-Provider Relationships
   */
  async getPatientProviders(patientId: string): Promise<PatientProviderRelationship[]> {
    try {
      const { data, error } = await supabase
        .from('patient_provider_relationships')
        .select('*')
        .eq('patient_id', patientId)
        .eq('is_active', true);

      if (error) throw error;
      return data || [];
    } catch (error) {
      console.error('Error fetching patient providers:', error);
      return [];
    }
  }

  async getProviderPatients(providerId: string): Promise<PatientProviderRelationship[]> {
    try {
      const { data, error } = await supabase
        .from('patient_provider_relationships')
        .select('*')
        .eq('provider_id', providerId)
        .eq('is_active', true);

      if (error) throw error;
      return data || [];
    } catch (error) {
      console.error('Error fetching provider patients:', error);
      return [];
    }
  }

  /**
   * Analytics and Statistics
   */
  async getPatientHealthSummary(patientId: string, days: number = 30): Promise<PatientHealthSummary | null> {
    try {
      const { data, error } = await supabase
        .rpc('get_patient_health_summary', {
          p_patient_id: patientId,
          p_days: days
        });

      if (error) throw error;
      return data?.[0] || null;
    } catch (error) {
      console.error('Error fetching patient health summary:', error);
      return this.getMockPatientHealthSummary();
    }
  }

  async getConsultationStats(providerId: string, startDate: string, endDate: string): Promise<ConsultationStats | null> {
    try {
      const { data, error } = await supabase
        .rpc('get_consultation_stats', {
          p_provider_id: providerId,
          p_start_date: startDate,
          p_end_date: endDate
        });

      if (error) throw error;
      return data?.[0] || null;
    } catch (error) {
      console.error('Error fetching consultation stats:', error);
      return this.getMockConsultationStats();
    }
  }

  /**
   * Video Consultation Integration
   */
  async generateMeetingUrl(consultationId: string): Promise<string> {
    // In a real implementation, this would integrate with video platforms like Zoom, Teams, etc.
    // For now, return a mock URL
    return `https://meet.example.com/${consultationId}`;
  }

  async joinConsultation(consultationId: string, userId: string): Promise<{ meetingUrl: string; meetingId: string }> {
    try {
      const meetingUrl = await this.generateMeetingUrl(consultationId);
      const meetingId = `meeting_${consultationId}_${Date.now()}`;

      await this.updateConsultation(consultationId, {
        meeting_url: meetingUrl,
        meeting_id: meetingId,
        status: 'in_progress',
        started_at: new Date().toISOString()
      });

      return { meetingUrl, meetingId };
    } catch (error) {
      console.error('Error joining consultation:', error);
      throw error;
    }
  }

  async endConsultation(consultationId: string, durationMinutes: number): Promise<void> {
    try {
      await this.updateConsultation(consultationId, {
        status: 'completed',
        ended_at: new Date().toISOString(),
        duration_minutes: durationMinutes
      });
    } catch (error) {
      console.error('Error ending consultation:', error);
      throw error;
    }
  }

  // Mock data methods for development
  private getMockHealthcareProviders(): HealthcareProvider[] {
    return [
      {
        id: '1',
        user_id: 'provider1',
        provider_name: 'Dr. Sarah Johnson',
        specialty: 'cardiology',
        experience_years: 15,
        consultation_fee: 150.00,
        rating: 4.8,
        total_consultations: 1250,
        is_verified: true,
        is_available: true,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString()
      },
      {
        id: '2',
        user_id: 'provider2',
        provider_name: 'Dr. Michael Chen',
        specialty: 'dermatology',
        experience_years: 12,
        consultation_fee: 120.00,
        rating: 4.9,
        total_consultations: 980,
        is_verified: true,
        is_available: true,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString()
      }
    ];
  }

  private getMockAvailableSlots(): AvailableAppointmentSlot[] {
    return [
      { start_time: '09:00:00', end_time: '09:30:00', is_available: true },
      { start_time: '09:30:00', end_time: '10:00:00', is_available: false },
      { start_time: '10:00:00', end_time: '10:30:00', is_available: true },
      { start_time: '10:30:00', end_time: '11:00:00', is_available: true }
    ];
  }

  private getMockAppointments(): Appointment[] {
    return [
      {
        id: '1',
        patient_id: 'patient1',
        provider_id: 'provider1',
        appointment_type: 'consultation',
        scheduled_at: new Date(Date.now() + 24 * 60 * 60 * 1000).toISOString(),
        duration_minutes: 30,
        status: 'scheduled',
        is_urgent: false,
        reminder_sent: false,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString()
      }
    ];
  }

  private getMockConsultations(): TelemedicineConsultation[] {
    return [
      {
        id: '1',
        patient_id: 'patient1',
        provider_id: 'provider1',
        consultation_type: 'video',
        status: 'completed',
        scheduled_at: new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString(),
        started_at: new Date(Date.now() - 24 * 60 * 60 * 1000 + 5 * 60 * 1000).toISOString(),
        ended_at: new Date(Date.now() - 24 * 60 * 60 * 1000 + 35 * 60 * 1000).toISOString(),
        duration_minutes: 30,
        follow_up_required: false,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString()
      }
    ];
  }

  private getMockRemoteMonitoring(): RemoteMonitoring[] {
    return [
      {
        id: '1',
        patient_id: 'patient1',
        provider_id: 'provider1',
        monitoring_type: 'heart_rate',
        device_type: 'smartwatch',
        is_active: true,
        start_date: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString(),
        monitoring_frequency: 'continuous',
        last_reading_at: new Date().toISOString(),
        last_reading_value: { heart_rate: 72, timestamp: new Date().toISOString() },
        alert_level: 'low',
        is_alert_active: false,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString()
      }
    ];
  }

  private getMockHealthAlerts(): HealthAlert[] {
    return [
      {
        id: '1',
        patient_id: 'patient1',
        alert_type: 'heart_rate',
        alert_level: 'medium',
        title: 'Elevated Heart Rate',
        message: 'Your heart rate has been elevated for the past hour.',
        metric_name: 'heart_rate',
        metric_value: { value: 95, unit: 'bpm' },
        threshold_value: { value: 90, unit: 'bpm' },
        is_read: false,
        is_acknowledged: false,
        follow_up_required: false,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString()
      }
    ];
  }

  private getMockPatientHealthSummary(): PatientHealthSummary {
    return {
      recent_consultations: 3,
      active_monitoring_count: 2,
      unread_alerts: 1,
      last_consultation_date: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString(),
      health_score: 78.5,
      risk_level: 'low'
    };
  }

  private getMockConsultationStats(): ConsultationStats {
    return {
      total_consultations: 45,
      completed_consultations: 42,
      cancelled_consultations: 3,
      average_duration_minutes: 28.5,
      total_revenue: 6300.00
    };
  }
}

export const telemedicineService = new TelemedicineService();
