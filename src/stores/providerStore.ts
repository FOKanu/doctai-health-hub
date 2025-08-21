import { create } from 'zustand';
import { devtools } from 'zustand/middleware';

// Types
export interface Patient {
  id: string;
  name: string;
  age: number;
  sex: 'M' | 'F' | 'Other';
  mrn: string;
  phone: string;
  email: string;
  primarySpecialty: string;
  risk: 'Low' | 'Medium' | 'High';
  status: 'New Patient' | 'Follow-up' | 'Routine';
  lastVisit: string;
  avatar?: string;
  conditions: string[];
  medications: string[];
  vitals: {
    bloodPressure: string;
    heartRate: number;
    temperature: number;
    weight: number;
  };
}

export interface Appointment {
  id: string;
  patientId: string;
  patientName: string;
  date: string;
  time: string;
  type: 'Consultation' | 'Follow-up' | 'Procedure' | 'Emergency';
  status: 'Scheduled' | 'In Progress' | 'Completed' | 'Cancelled';
  specialty: string;
  notes?: string;
}

export interface Message {
  id: string;
  threadId: string;
  senderId: string;
  senderName: string;
  recipientId: string;
  recipientName: string;
  content: string;
  timestamp: string;
  isRead: boolean;
  attachments?: string[];
}

export interface AIInsight {
  id: string;
  patientId: string;
  patientName: string;
  type: 'Risk Assessment' | 'Diagnostic Support' | 'Treatment Recommendation';
  insight: string;
  confidence: number;
  priority: 'High' | 'Medium' | 'Low';
  status: 'Pending' | 'Approved' | 'Rejected' | 'Needs Clarification';
  timestamp: string;
  evidence?: string[];
}

export interface AuditLog {
  id: string;
  action: string;
  userId: string;
  userName: string;
  resource: string;
  resourceId: string;
  details: string;
  timestamp: string;
  ipAddress: string;
}

export interface Order {
  id: string;
  patientId: string;
  patientName: string;
  type: 'Lab' | 'Imaging' | 'Medication' | 'Procedure';
  status: 'Pending' | 'In Progress' | 'Completed' | 'Cancelled';
  orderedBy: string;
  orderedDate: string;
  completedDate?: string;
  notes?: string;
}

// Deterministic seed data
const generatePatients = (): Patient[] => {
  const names = [
    'Sarah Johnson', 'Michael Chen', 'Emily Rodriguez', 'David Thompson', 'Lisa Wang',
    'James Wilson', 'Maria Garcia', 'Robert Brown', 'Jennifer Lee', 'Christopher Davis',
    'Amanda Martinez', 'Daniel Anderson', 'Jessica Taylor', 'Matthew White', 'Ashley Clark',
    'Joshua Hall', 'Stephanie Lewis', 'Andrew Walker', 'Nicole Green', 'Kevin Baker',
    'Rachel Adams', 'Ryan Nelson', 'Lauren Carter', 'Brandon Mitchell', 'Megan Perez'
  ];

  return names.map((name, index) => ({
    id: `P${String(index + 1).padStart(3, '0')}`,
    name,
    age: 25 + (index % 50),
    sex: index % 3 === 0 ? 'M' : index % 3 === 1 ? 'F' : 'Other',
    mrn: `MRN${String(index + 1).padStart(6, '0')}`,
    phone: `555-${String(100 + index).padStart(3, '0')}-${String(1000 + index).padStart(4, '0')}`,
    email: `${name.toLowerCase().replace(' ', '.')}@email.com`,
    primarySpecialty: ['Cardiology', 'Neurology', 'Ophthalmology', 'Orthopedics'][index % 4],
    risk: index % 10 < 3 ? 'High' : index % 10 < 7 ? 'Medium' : 'Low',
    status: index % 6 < 2 ? 'New Patient' : index % 6 < 4 ? 'Follow-up' : 'Routine',
    lastVisit: `${index % 30 + 1} days ago`,
    conditions: ['Hypertension', 'Diabetes', 'Asthma', 'Arthritis'].slice(0, (index % 3) + 1),
    medications: ['Lisinopril', 'Metformin', 'Albuterol', 'Ibuprofen'].slice(0, (index % 2) + 1),
    vitals: {
      bloodPressure: `${120 + (index % 20)}/${80 + (index % 15)}`,
      heartRate: 70 + (index % 20),
      temperature: 98.6 + (index % 2),
      weight: 150 + (index % 50)
    }
  }));
};

const generateAppointments = (): Appointment[] => {
  const today = new Date();
  const appointments = [];
  
  // Generate 18 appointments for today (matching dashboard)
  for (let i = 0; i < 18; i++) {
    const hour = 8 + (i % 8); // 8 AM to 4 PM
    const minute = (i % 4) * 15; // 0, 15, 30, 45
    const date = new Date(today);
    date.setHours(hour, minute, 0, 0);
    
    appointments.push({
      id: `A${String(i + 1).padStart(3, '0')}`,
      patientId: `P${String((i % 25) + 1).padStart(3, '0')}`,
      patientName: generatePatients()[i % 25].name,
      date: date.toISOString().split('T')[0],
      time: date.toTimeString().slice(0, 5),
      type: ['Consultation', 'Follow-up', 'Procedure', 'Emergency'][i % 4],
      status: 'Scheduled',
      specialty: ['Cardiology', 'Neurology', 'Ophthalmology', 'Orthopedics'][i % 4],
      notes: i % 3 === 0 ? 'Patient requested early appointment' : undefined
    });
  }
  
  return appointments;
};

const generateAIInsights = (): AIInsight[] => {
  const patients = generatePatients();
  const insights = [];
  
  // Generate insights to match 89% AI diagnoses rate
  for (let i = 0; i < 50; i++) {
    const patient = patients[i % patients.length];
    insights.push({
      id: `AI${String(i + 1).padStart(3, '0')}`,
      patientId: patient.id,
      patientName: patient.name,
      type: ['Risk Assessment', 'Diagnostic Support', 'Treatment Recommendation'][i % 3],
      insight: [
        'Cardiovascular risk increased by 15%',
        'X-ray analysis suggests early-stage pneumonia',
        'Consider adjusting medication dosage',
        'Blood pressure trending upward',
        'Cholesterol levels require monitoring'
      ][i % 5],
      confidence: 85 + (i % 15),
      priority: i % 10 < 3 ? 'High' : i % 10 < 7 ? 'Medium' : 'Low',
      status: i % 10 < 8 ? 'Pending' : 'Approved', // 80% pending, 20% approved
      timestamp: new Date(Date.now() - (i * 3600000)).toISOString(),
      evidence: ['Lab results', 'Imaging', 'Vital signs', 'Patient history'].slice(0, (i % 3) + 1)
    });
  }
  
  return insights;
};

// Store
interface ProviderStore {
  // State
  patients: Patient[];
  appointments: Appointment[];
  messages: Message[];
  aiInsights: AIInsight[];
  auditLogs: AuditLog[];
  orders: Order[];
  
  // Actions
  addPatient: (patient: Omit<Patient, 'id'>) => void;
  updatePatient: (id: string, updates: Partial<Patient>) => void;
  deletePatient: (id: string) => void;
  
  addAppointment: (appointment: Omit<Appointment, 'id'>) => void;
  updateAppointment: (id: string, updates: Partial<Appointment>) => void;
  deleteAppointment: (id: string) => void;
  
  addMessage: (message: Omit<Message, 'id'>) => void;
  markMessageAsRead: (id: string) => void;
  
  updateAIInsight: (id: string, updates: Partial<AIInsight>) => void;
  addAuditLog: (log: Omit<AuditLog, 'id'>) => void;
  
  // Selectors
  getActivePatients: () => Patient[];
  getTodaysAppointments: () => Appointment[];
  getPendingAlerts: () => AIInsight[];
  getAIDiagnosesRate: () => number;
  getPatientById: (id: string) => Patient | undefined;
  getAppointmentsByPatient: (patientId: string) => Appointment[];
  getAIInsightsByPatient: (patientId: string) => AIInsight[];
}

export const useProviderStore = create<ProviderStore>()(
  devtools(
    (set, get) => ({
      // Initial state with deterministic data
      patients: generatePatients(),
      appointments: generateAppointments(),
      messages: [],
      aiInsights: generateAIInsights(),
      auditLogs: [],
      orders: [],
      
      // Actions
      addPatient: (patientData) => {
        const newPatient: Patient = {
          ...patientData,
          id: `P${String(get().patients.length + 1).padStart(3, '0')}`
        };
        set((state) => ({ patients: [...state.patients, newPatient] }));
      },
      
      updatePatient: (id, updates) => {
        set((state) => ({
          patients: state.patients.map(patient =>
            patient.id === id ? { ...patient, ...updates } : patient
          )
        }));
      },
      
      deletePatient: (id) => {
        set((state) => ({
          patients: state.patients.filter(patient => patient.id !== id)
        }));
      },
      
      addAppointment: (appointmentData) => {
        const newAppointment: Appointment = {
          ...appointmentData,
          id: `A${String(get().appointments.length + 1).padStart(3, '0')}`
        };
        set((state) => ({ appointments: [...state.appointments, newAppointment] }));
      },
      
      updateAppointment: (id, updates) => {
        set((state) => ({
          appointments: state.appointments.map(appointment =>
            appointment.id === id ? { ...appointment, ...updates } : appointment
          )
        }));
      },
      
      deleteAppointment: (id) => {
        set((state) => ({
          appointments: state.appointments.filter(appointment => appointment.id !== id)
        }));
      },
      
      addMessage: (messageData) => {
        const newMessage: Message = {
          ...messageData,
          id: `M${String(get().messages.length + 1).padStart(3, '0')}`
        };
        set((state) => ({ messages: [...state.messages, newMessage] }));
      },
      
      markMessageAsRead: (id) => {
        set((state) => ({
          messages: state.messages.map(message =>
            message.id === id ? { ...message, isRead: true } : message
          )
        }));
      },
      
      updateAIInsight: (id, updates) => {
        set((state) => ({
          aiInsights: state.aiInsights.map(insight =>
            insight.id === id ? { ...insight, ...updates } : insight
          )
        }));
      },
      
      addAuditLog: (logData) => {
        const newLog: AuditLog = {
          ...logData,
          id: `AL${String(get().auditLogs.length + 1).padStart(3, '0')}`
        };
        set((state) => ({ auditLogs: [...state.auditLogs, newLog] }));
      },
      
      // Selectors
      getActivePatients: () => {
        return get().patients.filter(patient => 
          patient.status !== 'Routine' || 
          new Date(patient.lastVisit).getTime() > Date.now() - (30 * 24 * 60 * 60 * 1000)
        );
      },
      
      getTodaysAppointments: () => {
        const today = new Date().toISOString().split('T')[0];
        return get().appointments.filter(appointment => 
          appointment.date === today && appointment.status === 'Scheduled'
        );
      },
      
      getPendingAlerts: () => {
        return get().aiInsights.filter(insight => 
          insight.status === 'Pending' && insight.priority === 'High'
        );
      },
      
      getAIDiagnosesRate: () => {
        const insights = get().aiInsights;
        const approved = insights.filter(insight => insight.status === 'Approved').length;
        return Math.round((approved / insights.length) * 100);
      },
      
      getPatientById: (id) => {
        return get().patients.find(patient => patient.id === id);
      },
      
      getAppointmentsByPatient: (patientId) => {
        return get().appointments.filter(appointment => appointment.patientId === patientId);
      },
      
      getAIInsightsByPatient: (patientId) => {
        return get().aiInsights.filter(insight => insight.patientId === patientId);
      }
    }),
    {
      name: 'provider-store'
    }
  )
);
