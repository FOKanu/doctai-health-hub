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