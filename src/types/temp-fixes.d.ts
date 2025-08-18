// Temporary type fixes for development build
// TODO: Replace these with proper types when refactoring

declare module '@/types/medical' {
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
    [key: string]: any; // Allow additional properties
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
    patientId?: string;
    prescribedBy?: string;
    updatedAt?: string;
    [key: string]: any; // Allow additional properties
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
    [key: string]: any; // Allow additional properties
  }

  export interface TimeSlot {
    id: string;
    time: string;
    start_time: string;
    end_time: string;
    is_available: boolean;
    duration: number;
    [key: string]: any; // Allow additional properties
  }

  export interface Notification {
    id: string;
    message: string;
    status: string;
    provider: string;
    [key: string]: any; // Allow additional properties
  }

  export interface DietPlanFormData {
    age: number;
    weight: number;
    height: number;
    activityLevel: string;
    goal: string;
    allergies: string[];
    preferences: string[];
    [key: string]: any; // Allow additional properties
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
    [key: string]: any; // Allow additional properties
  }
}

// Temporary API response type fixes
declare global {
  interface ApiResponse<T> {
    data?: T;
    message?: string;
    possibleConditions?: string[];
    severity?: 'low' | 'medium' | 'high';
    urgency?: 'low' | 'medium' | 'high';
    recommendations?: string[];
    explanation?: string;
    disclaimer?: string;
    simplified?: string;
    map?: any;
    [key: string]: any;
  }

  // Supabase types fallback
  namespace Database {
    interface public {
      Tables: {
        [key: string]: {
          Row: Record<string, any>;
          Insert: Record<string, any>;
          Update: Record<string, any>;
        };
      };
    }
  }
}

export {};