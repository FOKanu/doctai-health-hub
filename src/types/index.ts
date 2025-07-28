// Core application types
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
  id: number;
  title: string;
  date: string;
  time: string;
  doctor: string;
  type: string;
  status: string;
  location?: string;
  notes?: string;
  duration?: string;
  address?: string;
  insurance?: string;
}

export interface Notification {
  id: string;
  title: string;
  message: string;
  type: 'info' | 'warning' | 'error' | 'success';
  timestamp: string;
  read: boolean;
  provider?: string;
  status?: string;
}

export interface TimeSlot {
  id: string;
  startTime: string;
  endTime: string;
  available: boolean;
  providerId: string;
}

export interface ApiResponse<T = any> {
  success: boolean;
  data: T;
  message?: string;
  error?: string;
  possibleConditions?: string[];
  severity?: 'low' | 'medium' | 'high';
  urgency?: 'low' | 'medium' | 'high';
  recommendations?: string[];
  explanation?: string;
  disclaimer?: string;
  simplified?: string;
}

export interface HealthAssistantResponse {
  message: string;
  possibleConditions?: string[];
  severity?: 'low' | 'medium' | 'high';
  urgency?: 'low' | 'medium' | 'high';
  recommendations?: string[];
  explanation?: string;
  disclaimer?: string;
  simplified?: string;
}

export interface RiskProgression {
  date: string;
  low: number;
  medium: number;
  high: number;
}

export interface UserProfile {
  age: number;
  weight: number;
  height: number;
  activityLevel: string;
  goal: string;
  allergies: string[];
  preferences: string[];
}