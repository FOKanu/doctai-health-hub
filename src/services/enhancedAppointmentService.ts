import { supabase } from './supabaseClient';

export interface EnhancedAppointment {
  id: string;
  patientId: string;
  providerId: string;
  patientName: string;
  providerName: string;
  appointmentType: 'consultation' | 'follow-up' | 'procedure' | 'emergency' | 'telemedicine';
  scheduledAt: string;
  duration: number; // in minutes
  status: 'scheduled' | 'confirmed' | 'in-progress' | 'completed' | 'cancelled' | 'no-show';
  reason: string;
  symptoms?: string[];
  notes?: string;
  location?: string;
  meetingUrl?: string;
  insurance?: string;
  followUpRequired?: boolean;
  followUpDate?: string;
  outcome?: string;
  createdAt: string;
  updatedAt: string;
  metadata?: any;
}

export interface ProviderAvailability {
  id: string;
  providerId: string;
  date: string;
  startTime: string;
  endTime: string;
  slotDuration: number; // in minutes
  isAvailable: boolean;
  appointmentType: 'consultation' | 'follow-up' | 'procedure' | 'emergency' | 'telemedicine';
  maxBookings?: number;
  currentBookings?: number;
}

export interface TimeSlot {
  time: string;
  isAvailable: boolean;
  appointmentType: string;
  duration: number;
}

export interface AppointmentReminder {
  id: string;
  appointmentId: string;
  patientId: string;
  reminderType: 'email' | 'sms' | 'push';
  scheduledFor: string;
  message: string;
  isSent: boolean;
  sentAt?: string;
}

/**
 * Get available time slots for a provider on a specific date
 */
export const getProviderAvailability = async (
  providerId: string,
  date: string
): Promise<TimeSlot[]> => {
  try {
    // For MVP, we'll use mock data with realistic availability patterns
    const mockAvailability: TimeSlot[] = [
      { time: '09:00', isAvailable: true, appointmentType: 'consultation', duration: 30 },
      { time: '09:30', isAvailable: true, appointmentType: 'consultation', duration: 30 },
      { time: '10:00', isAvailable: false, appointmentType: 'consultation', duration: 30 },
      { time: '10:30', isAvailable: true, appointmentType: 'consultation', duration: 30 },
      { time: '11:00', isAvailable: true, appointmentType: 'consultation', duration: 30 },
      { time: '11:30', isAvailable: true, appointmentType: 'consultation', duration: 30 },
      { time: '14:00', isAvailable: true, appointmentType: 'consultation', duration: 30 },
      { time: '14:30', isAvailable: true, appointmentType: 'consultation', duration: 30 },
      { time: '15:00', isAvailable: false, appointmentType: 'consultation', duration: 30 },
      { time: '15:30', isAvailable: true, appointmentType: 'consultation', duration: 30 },
      { time: '16:00', isAvailable: true, appointmentType: 'consultation', duration: 30 },
      { time: '16:30', isAvailable: true, appointmentType: 'consultation', duration: 30 }
    ];

    // Filter based on current time (don't show past slots for today)
    const today = new Date().toISOString().split('T')[0];
    const currentTime = new Date().getHours() * 60 + new Date().getMinutes();

    if (date === today) {
      return mockAvailability.filter(slot => {
        const [hours, minutes] = slot.time.split(':').map(Number);
        const slotTime = hours * 60 + minutes;
        return slotTime > currentTime;
      });
    }

    return mockAvailability;
  } catch (error) {
    console.error('Error fetching provider availability:', error);
    return [];
  }
};

/**
 * Book an appointment
 */
export const bookAppointment = async (appointmentData: Omit<EnhancedAppointment, 'id' | 'createdAt' | 'updatedAt'>): Promise<EnhancedAppointment | null> => {
  try {
    console.log('Booking appointment:', appointmentData);

    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 1000));

    // Create appointment object
    const appointment: EnhancedAppointment = {
      ...appointmentData,
      id: `apt_${Date.now()}`,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString()
    };

    // In a real implementation, this would:
    // 1. Save to database
    // 2. Send confirmation to patient
    // 3. Send notification to provider
    // 4. Create appointment reminders
    // 5. Update provider availability

    return appointment;
  } catch (error) {
    console.error('Error booking appointment:', error);
    return null;
  }
};

/**
 * Get appointments for a patient
 */
export const getPatientAppointments = async (patientId: string, status?: string): Promise<EnhancedAppointment[]> => {
  try {
    // Mock appointments data
    const mockAppointments: EnhancedAppointment[] = [
      {
        id: 'apt_001',
        patientId: patientId,
        providerId: 'provider_001',
        patientName: 'Sarah Johnson',
        providerName: 'Dr. Sarah Weber',
        appointmentType: 'consultation',
        scheduledAt: new Date(Date.now() + 2 * 24 * 60 * 60 * 1000).toISOString(), // 2 days from now
        duration: 30,
        status: 'confirmed',
        reason: 'Follow-up for AI analysis results',
        symptoms: ['Skin lesion concern'],
        notes: 'Patient uploaded skin lesion image, AI analysis suggests follow-up needed',
        location: 'Berlin Medical Center',
        insurance: 'TK - Techniker Krankenkasse',
        createdAt: new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString(),
        updatedAt: new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString()
      },
      {
        id: 'apt_002',
        patientId: patientId,
        providerId: 'provider_002',
        patientName: 'Sarah Johnson',
        providerName: 'Dr. Michael Brown',
        appointmentType: 'telemedicine',
        scheduledAt: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000).toISOString(), // 1 week from now
        duration: 45,
        status: 'scheduled',
        reason: 'Initial consultation',
        symptoms: ['General health check'],
        meetingUrl: 'https://meet.doctai.com/room/123',
        insurance: 'TK - Techniker Krankenkasse',
        createdAt: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000).toISOString(),
        updatedAt: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000).toISOString()
      },
      {
        id: 'apt_003',
        patientId: patientId,
        providerId: 'provider_001',
        patientName: 'Sarah Johnson',
        providerName: 'Dr. Sarah Weber',
        appointmentType: 'consultation',
        scheduledAt: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString(), // 1 week ago
        duration: 30,
        status: 'completed',
        reason: 'Initial skin examination',
        symptoms: ['Mole examination'],
        outcome: 'Benign findings, 6-month follow-up recommended',
        followUpRequired: true,
        followUpDate: new Date(Date.now() + 5 * 30 * 24 * 60 * 60 * 1000).toISOString(), // 5 months from now
        insurance: 'TK - Techniker Krankenkasse',
        createdAt: new Date(Date.now() - 14 * 24 * 60 * 60 * 1000).toISOString(),
        updatedAt: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString()
      }
    ];

    // Filter by status if provided
    if (status) {
      return mockAppointments.filter(apt => apt.status === status);
    }

    return mockAppointments;
  } catch (error) {
    console.error('Error fetching patient appointments:', error);
    return [];
  }
};

/**
 * Get appointments for a provider
 */
export const getProviderAppointments = async (providerId: string, date?: string): Promise<EnhancedAppointment[]> => {
  try {
    // Mock provider appointments
    const mockAppointments: EnhancedAppointment[] = [
      {
        id: 'apt_001',
        patientId: 'patient_001',
        providerId: providerId,
        patientName: 'Sarah Johnson',
        providerName: 'Dr. Sarah Weber',
        appointmentType: 'consultation',
        scheduledAt: new Date(Date.now() + 2 * 24 * 60 * 60 * 1000).toISOString(),
        duration: 30,
        status: 'confirmed',
        reason: 'Follow-up for AI analysis results',
        symptoms: ['Skin lesion concern'],
        notes: 'Patient uploaded skin lesion image, AI analysis suggests follow-up needed',
        location: 'Berlin Medical Center',
        insurance: 'TK - Techniker Krankenkasse',
        createdAt: new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString(),
        updatedAt: new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString()
      },
      {
        id: 'apt_004',
        patientId: 'patient_002',
        providerId: providerId,
        patientName: 'Michael Chen',
        providerName: 'Dr. Sarah Weber',
        appointmentType: 'telemedicine',
        scheduledAt: new Date(Date.now() + 1 * 24 * 60 * 60 * 1000).toISOString(),
        duration: 45,
        status: 'scheduled',
        reason: 'X-ray results review',
        symptoms: ['Chest pain'],
        meetingUrl: 'https://meet.doctai.com/room/456',
        insurance: 'AOK - Allgemeine Ortskrankenkasse',
        createdAt: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000).toISOString(),
        updatedAt: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000).toISOString()
      }
    ];

    // Filter by date if provided
    if (date) {
      const targetDate = new Date(date).toISOString().split('T')[0];
      return mockAppointments.filter(apt =>
        apt.scheduledAt.split('T')[0] === targetDate
      );
    }

    return mockAppointments;
  } catch (error) {
    console.error('Error fetching provider appointments:', error);
    return [];
  }
};

/**
 * Update appointment status
 */
export const updateAppointmentStatus = async (
  appointmentId: string,
  status: EnhancedAppointment['status'],
  notes?: string
): Promise<boolean> => {
  try {
    console.log('Updating appointment status:', { appointmentId, status, notes });

    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 500));

    // In a real implementation, this would:
    // 1. Update the appointment in the database
    // 2. Send notification to patient if status changed
    // 3. Log the status change for audit purposes

    return true;
  } catch (error) {
    console.error('Error updating appointment status:', error);
    return false;
  }
};

/**
 * Cancel an appointment
 */
export const cancelAppointment = async (appointmentId: string, reason?: string): Promise<boolean> => {
  try {
    console.log('Cancelling appointment:', { appointmentId, reason });

    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 500));

    // In a real implementation, this would:
    // 1. Update appointment status to 'cancelled'
    // 2. Free up the time slot
    // 3. Send cancellation notification to patient
    // 4. Send notification to provider
    // 5. Log the cancellation reason

    return true;
  } catch (error) {
    console.error('Error cancelling appointment:', error);
    return false;
  }
};

/**
 * Reschedule an appointment
 */
export const rescheduleAppointment = async (
  appointmentId: string,
  newDateTime: string,
  reason?: string
): Promise<boolean> => {
  try {
    console.log('Rescheduling appointment:', { appointmentId, newDateTime, reason });

    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 500));

    // In a real implementation, this would:
    // 1. Check if new time slot is available
    // 2. Update appointment with new time
    // 3. Send reschedule notification to patient
    // 4. Send notification to provider
    // 5. Update any existing reminders

    return true;
  } catch (error) {
    console.error('Error rescheduling appointment:', error);
    return false;
  }
};

/**
 * Set provider availability
 */
export const setProviderAvailability = async (
  providerId: string,
  availability: Omit<ProviderAvailability, 'id' | 'providerId'>
): Promise<boolean> => {
  try {
    console.log('Setting provider availability:', { providerId, availability });

    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 500));

    // In a real implementation, this would:
    // 1. Save availability to database
    // 2. Update available time slots
    // 3. Notify patients of new availability

    return true;
  } catch (error) {
    console.error('Error setting provider availability:', error);
    return false;
  }
};
