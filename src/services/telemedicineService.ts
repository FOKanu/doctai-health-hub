/**
 * Telemedicine Service - Mock Implementation
 * Provides basic telemedicine functionality for development
 */

// Define types locally
export interface HealthcareProvider {
  id: string;
  name: string;
  specialty: string;
  credentials?: string;
  isAvailable: boolean;
  rating: number;
  profile_image_url?: string;
  provider_name?: string;
  total_consultations?: number;
  experience_years?: number;
  bio?: string;
  consultation_fee?: number;
  is_available?: boolean;
}

export interface TelemedicineConsultation {
  id: string;
  patientId: string;
  providerId: string;
  scheduledAt: string;
  status: 'scheduled' | 'in-progress' | 'completed' | 'cancelled';
  duration: number;
  notes?: string;
  consultation_type?: string;
  scheduled_at?: string;
  duration_minutes?: number;
  diagnosis?: string;
  recommendations?: string[];
  meeting_url?: string;
}

export interface Appointment {
  id: string;
  patientId: string;
  providerId: string;
  scheduledAt: string;
  status: string;
  type: string;
  reason?: string;
  scheduled_at?: string;
  appointment_type?: string;
  symptoms?: string[];
}

export class TelemedicineService {
  async getHealthcareProviders(params?: any): Promise<HealthcareProvider[]> {
    return [
      {
        id: '1',
        name: 'Dr. Sarah Johnson',
        provider_name: 'Dr. Sarah Johnson',
        specialty: 'Cardiology',
        experience_years: 15,
        consultation_fee: 150,
        rating: 4.8,
        total_consultations: 1250,
        isAvailable: true,
        is_available: true,
        bio: 'Experienced cardiologist specializing in preventive care.',
        profile_image_url: '/api/placeholder/100/100'
      },
      {
        id: '2',
        name: 'Dr. Michael Chen',
        provider_name: 'Dr. Michael Chen',
        specialty: 'Dermatology',
        experience_years: 12,
        consultation_fee: 120,
        rating: 4.9,
        total_consultations: 980,
        isAvailable: true,
        is_available: true,
        bio: 'Expert dermatologist with focus on skin cancer prevention.',
        profile_image_url: '/api/placeholder/100/100'
      }
    ];
  }

  async getConsultations(params?: any): Promise<TelemedicineConsultation[]> {
    return [
      {
        id: '1',
        patientId: 'patient1',
        providerId: 'provider1',
        scheduledAt: new Date().toISOString(),
        scheduled_at: new Date().toISOString(),
        status: 'completed',
        duration: 30,
        duration_minutes: 30,
        consultation_type: 'video',
        diagnosis: 'Routine checkup completed',
        recommendations: ['Continue current medication', 'Schedule follow-up in 3 months'],
        meeting_url: 'https://meet.example.com/consultation1'
      }
    ];
  }

  async getAppointments(params?: any): Promise<Appointment[]> {
    return [
      {
        id: '1',
        patientId: 'patient1',
        providerId: 'provider1',
        scheduledAt: new Date(Date.now() + 24 * 60 * 60 * 1000).toISOString(),
        scheduled_at: new Date(Date.now() + 24 * 60 * 60 * 1000).toISOString(),
        status: 'scheduled',
        type: 'consultation',
        appointment_type: 'consultation',
        reason: 'Regular checkup',
        symptoms: ['No symptoms']
      }
    ];
  }

  async getProviderAvailability(providerId: string, date: string): Promise<any[]> {
    return [{ 
      providerId: '1', 
      availableSlots: ['10:00', '14:00', '16:00'] 
    }];
  }

  async bookAppointment(appointmentData: any): Promise<any> {
    return { 
      id: 'new-appointment', 
      status: 'booked', 
      ...appointmentData 
    };
  }

  async joinConsultation(consultationId: string, userId: string): Promise<any> {
    return { 
      id: consultationId, 
      meetingUrl: 'https://meet.example.com/' + consultationId 
    };
  }
}

export const telemedicineService = new TelemedicineService();