// Medical types for the healthcare application

export interface Medication {
  id: number;
  name: string;
  dosage: string;
  frequency: string;
  timing: string;
  reminderEnabled: boolean;
  treatmentPlan: string;
  refillDate: string;
  nextRefill: string;
  pillsRemaining: number;
}

export interface Treatment {
  id: number;
  name: string;
  type: string;
  doctor: string;
  startDate: string;
  duration: string;
  description: string;
  priority: string;
  status: string;
  createdAt: string;
  patientId?: string; // Make optional to match common.ts
  prescribedBy?: string; // Add to match common.ts
  updatedAt?: string; // Add to match common.ts
}

export interface Appointment {
  id: string;
  date: string;
  time: string;
  doctor: string;
  type: string;
  status: string;
  location: string;
  duration: string;
  insurance: string;
  address?: string;
  notes?: string;
  outcome?: string;
  followUpRequired?: boolean;
  followUpDate?: string;
}

export interface TimeSlot {
  id: string;
  time: string;
  start_time: string;
  end_time: string;
  is_available: boolean;
  duration: number;
}

export interface Notification {
  id: string;
  message: string;
  status: string;
  provider: string;
  userId?: string; // Add to match common.ts
  type?: string; // Add to match common.ts
  title?: string; // Add to match common.ts
  priority?: string; // Add to match common.ts
  isRead?: boolean; // Add to match common.ts
  actionRequired?: boolean; // Add to match common.ts
  actionUrl?: string; // Add to match common.ts
  scheduledFor?: Date; // Add to match common.ts
}

export interface DietPlanFormData {
  age: number;
  weight: number;
  height: number;
  activityLevel: string;
  goal: string;
  allergies: string[];
  preferences: string[];
}

export interface HealthAnalysisResponse {
  message: string;
  possibleConditions: string[];
  severity: 'low' | 'medium' | 'high';
  urgency: 'low' | 'medium' | 'high';
  recommendations: string[];
  explanation: string;
  disclaimer: string;
  simplified: string;
}

export interface LabTest {
  id: string;
  patientId: string;
  patientName: string;
  testType: string;
  testName: string;
  orderedBy: string;
  orderedDate: string;
  completedDate?: string;
  status: 'Pending' | 'In Progress' | 'Completed' | 'Cancelled';
  results?: string;
  normalRange?: string;
  notes?: string;
  priority: 'Low' | 'Medium' | 'High' | 'Urgent';
}

export interface Prescription {
  id: string;
  patientId: string;
  patientName: string;
  medicationName: string;
  dosage: string;
  frequency: string;
  quantity: number;
  refillsRemaining: number;
  prescribedDate: string;
  expirationDate: string;
  renewalDate: string;
  status: 'Active' | 'Expired' | 'Cancelled' | 'Pending Renewal';
  notes?: string;
  prescribedBy: string;
}

export interface VitalRecord {
  id: string;
  patientId: string;
  patientName: string;
  recordedDate: string;
  recordedBy: string;
  bloodPressureSystolic?: number;
  bloodPressureDiastolic?: number;
  heartRate?: number;
  temperature?: number;
  respiratoryRate?: number;
  oxygenSaturation?: number;
  weight?: number;
  height?: number;
  bmi?: number;
  notes?: string;
}