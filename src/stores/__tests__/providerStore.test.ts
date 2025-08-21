import { describe, it, expect, beforeEach } from '@jest/globals';
import { useProviderStore } from '../providerStore';

// Import the generator functions from the store file
const generatePatients = () => {
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

const generateAppointments = () => {
  const today = new Date();
  const appointments = [];
  
  for (let i = 0; i < 18; i++) {
    const hour = 8 + (i % 8);
    const minute = (i % 4) * 15;
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

const generateAIInsights = () => {
  const patients = generatePatients();
  const insights = [];
  
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
      status: i % 10 < 8 ? 'Pending' : 'Approved',
      timestamp: new Date(Date.now() - (i * 3600000)).toISOString(),
      evidence: ['Lab results', 'Imaging', 'Vital signs', 'Patient history'].slice(0, (i % 3) + 1)
    });
  }
  
  return insights;
};

describe('ProviderStore', () => {
  beforeEach(() => {
    // Reset store to initial state before each test
    useProviderStore.setState({
      patients: generatePatients(),
      appointments: generateAppointments(),
      messages: [],
      aiInsights: generateAIInsights(),
      auditLogs: [],
      orders: []
    });
  });

  describe('Initial State', () => {
    it('should have correct initial data', () => {
      const store = useProviderStore.getState();
      
      // Check that we have the expected number of patients
      expect(store.patients.length).toBe(25);
      
      // Check that we have the expected number of appointments
      expect(store.appointments.length).toBe(18);
      
      // Check that we have the expected number of AI insights
      expect(store.aiInsights.length).toBe(50);
    });

    it('should have deterministic dashboard numbers', () => {
      const store = useProviderStore.getState();
      
      // Test dashboard numbers match requirements
      const activePatients = store.getActivePatients();
      const todaysAppointments = store.getTodaysAppointments();
      const pendingAlerts = store.getPendingAlerts();
      const aiDiagnosesRate = store.getAIDiagnosesRate();
      
      expect(activePatients.length).toBeGreaterThan(0);
      expect(todaysAppointments.length).toBe(18);
      expect(pendingAlerts.length).toBeGreaterThan(0);
      expect(aiDiagnosesRate).toBe(20); // 20% approved from our seed data
    });
  });

  describe('Patient Management', () => {
    it('should add a new patient', () => {
      const store = useProviderStore.getState();
      const initialCount = store.patients.length;
      
      const newPatient = {
        name: 'John Doe',
        age: 35,
        sex: 'M' as const,
        mrn: 'MRN999999',
        phone: '555-123-4567',
        email: 'john.doe@email.com',
        primarySpecialty: 'Cardiology',
        risk: 'Low' as const,
        status: 'New Patient' as const,
        lastVisit: '1 day ago',
        conditions: ['Hypertension'],
        medications: ['Lisinopril'],
        vitals: {
          bloodPressure: '120/80',
          heartRate: 72,
          temperature: 98.6,
          weight: 175
        }
      };
      
      store.addPatient(newPatient);
      
      expect(store.patients.length).toBe(initialCount + 1);
      expect(store.patients[store.patients.length - 1].name).toBe('John Doe');
    });

    it('should update a patient', () => {
      const store = useProviderStore.getState();
      const patientId = store.patients[0].id;
      
      store.updatePatient(patientId, { age: 50 });
      
      const updatedPatient = store.patients.find(p => p.id === patientId);
      expect(updatedPatient?.age).toBe(50);
    });

    it('should delete a patient', () => {
      const store = useProviderStore.getState();
      const initialCount = store.patients.length;
      const patientId = store.patients[0].id;
      
      store.deletePatient(patientId);
      
      expect(store.patients.length).toBe(initialCount - 1);
      expect(store.patients.find(p => p.id === patientId)).toBeUndefined();
    });

    it('should get patient by ID', () => {
      const store = useProviderStore.getState();
      const patientId = store.patients[0].id;
      
      const patient = store.getPatientById(patientId);
      
      expect(patient).toBeDefined();
      expect(patient?.id).toBe(patientId);
    });
  });

  describe('Appointment Management', () => {
    it('should add a new appointment', () => {
      const store = useProviderStore.getState();
      const initialCount = store.appointments.length;
      
      const newAppointment = {
        patientId: 'P001',
        patientName: 'John Doe',
        date: '2024-01-15',
        time: '10:00',
        type: 'Consultation' as const,
        status: 'Scheduled' as const,
        specialty: 'Cardiology'
      };
      
      store.addAppointment(newAppointment);
      
      expect(store.appointments.length).toBe(initialCount + 1);
      expect(store.appointments[store.appointments.length - 1].patientName).toBe('John Doe');
    });

    it('should get appointments by patient', () => {
      const store = useProviderStore.getState();
      const patientId = store.patients[0].id;
      
      const appointments = store.getAppointmentsByPatient(patientId);
      
      expect(Array.isArray(appointments)).toBe(true);
      appointments.forEach(appointment => {
        expect(appointment.patientId).toBe(patientId);
      });
    });
  });

  describe('AI Insights', () => {
    it('should update AI insight status', () => {
      const store = useProviderStore.getState();
      const insightId = store.aiInsights[0].id;
      
      store.updateAIInsight(insightId, { status: 'Approved' });
      
      const updatedInsight = store.aiInsights.find(i => i.id === insightId);
      expect(updatedInsight?.status).toBe('Approved');
    });

    it('should get AI insights by patient', () => {
      const store = useProviderStore.getState();
      const patientId = store.patients[0].id;
      
      const insights = store.getAIInsightsByPatient(patientId);
      
      expect(Array.isArray(insights)).toBe(true);
      insights.forEach(insight => {
        expect(insight.patientId).toBe(patientId);
      });
    });
  });

  describe('Audit Logging', () => {
    it('should add audit log', () => {
      const store = useProviderStore.getState();
      const initialCount = store.auditLogs.length;
      
      const newLog = {
        action: 'AI_INSIGHT_APPROVED',
        userId: 'U001',
        userName: 'Dr. Sarah Johnson',
        resource: 'AIInsight',
        resourceId: 'AI001',
        details: 'Approved cardiovascular risk assessment',
        timestamp: new Date().toISOString(),
        ipAddress: '192.168.1.1'
      };
      
      store.addAuditLog(newLog);
      
      expect(store.auditLogs.length).toBe(initialCount + 1);
      expect(store.auditLogs[store.auditLogs.length - 1].action).toBe('AI_INSIGHT_APPROVED');
    });
  });

  describe('Dashboard Selectors', () => {
    it('should calculate active patients correctly', () => {
      const store = useProviderStore.getState();
      const activePatients = store.getActivePatients();
      
      expect(activePatients.length).toBeGreaterThan(0);
      activePatients.forEach(patient => {
        expect(patient.status).not.toBe('Routine');
      });
    });

    it('should calculate today\'s appointments correctly', () => {
      const store = useProviderStore.getState();
      const todaysAppointments = store.getTodaysAppointments();
      
      expect(todaysAppointments.length).toBe(18);
      todaysAppointments.forEach(appointment => {
        expect(appointment.status).toBe('Scheduled');
        expect(appointment.date).toBe(new Date().toISOString().split('T')[0]);
      });
    });

    it('should calculate pending alerts correctly', () => {
      const store = useProviderStore.getState();
      const pendingAlerts = store.getPendingAlerts();
      
      pendingAlerts.forEach(alert => {
        expect(alert.status).toBe('Pending');
        expect(alert.priority).toBe('High');
      });
    });

    it('should calculate AI diagnoses rate correctly', () => {
      const store = useProviderStore.getState();
      const aiDiagnosesRate = store.getAIDiagnosesRate();
      
      expect(aiDiagnosesRate).toBe(20); // 20% approved from seed data
      expect(aiDiagnosesRate).toBeGreaterThanOrEqual(0);
      expect(aiDiagnosesRate).toBeLessThanOrEqual(100);
    });
  });
});
