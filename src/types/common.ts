// =============================================================================
// COMMON TYPE DEFINITIONS FOR DOCTAI HEALTH HUB
// =============================================================================

// =============================================================================
// BASIC TYPES
// =============================================================================

export interface BaseEntity {
  id: string;
  createdAt: Date;
  updatedAt: Date;
}

export interface User {
  id: string;
  name: string;
  email: string;
  phone?: string;
  dateOfBirth?: Date;
  gender?: 'male' | 'female' | 'other';
  address?: Address;
}

export interface Address {
  street: string;
  city: string;
  state: string;
  zipCode: string;
  country: string;
}

// =============================================================================
// MEDICAL TYPES
// =============================================================================

export interface Patient extends BaseEntity {
  userId: string;
  medicalRecordNumber: string;
  primaryCarePhysician?: string;
  emergencyContact?: EmergencyContact;
  insurance?: InsuranceInfo;
  allergies: string[];
  conditions: MedicalCondition[];
}

export interface MedicalCondition {
  id: string;
  name: string;
  diagnosisDate: Date;
  severity: 'mild' | 'moderate' | 'severe';
  status: 'active' | 'resolved' | 'chronic';
  notes?: string;
}

export interface EmergencyContact {
  name: string;
  relationship: string;
  phone: string;
  email?: string;
}

export interface InsuranceInfo {
  provider: string;
  policyNumber: string;
  groupNumber?: string;
  expirationDate: Date;
}

// =============================================================================
// TREATMENT TYPES
// =============================================================================

export interface Treatment extends BaseEntity {
  patientId: string;
  type: 'medication' | 'procedure' | 'therapy' | 'surgery';
  name: string;
  description: string;
  startDate: Date;
  endDate?: Date;
  status: 'active' | 'completed' | 'discontinued';
  prescribedBy: string;
  dosage?: string;
  frequency?: string;
  instructions?: string;
  sideEffects?: string[];
}

export interface Medication extends Treatment {
  type: 'medication';
  dosage: string;
  frequency: string;
  route: 'oral' | 'injection' | 'topical' | 'inhalation';
  quantity: number;
  refillDate?: Date;
  takenToday: boolean;
  reminderEnabled: boolean;
}

export interface Procedure extends Treatment {
  type: 'procedure';
  procedureCode?: string;
  facility: string;
  surgeon?: string;
  anesthesia?: string;
  recoveryTime?: string;
}

// =============================================================================
// APPOINTMENT TYPES
// =============================================================================

export interface Appointment extends BaseEntity {
  patientId: string;
  providerId: string;
  type: 'consultation' | 'follow-up' | 'procedure' | 'emergency';
  dateTime: Date;
  duration: number; // minutes
  status: 'scheduled' | 'confirmed' | 'completed' | 'cancelled' | 'no-show';
  location: string;
  notes?: string;
  symptoms?: string[];
  diagnosis?: string;
  treatment?: string;
}

export interface Provider {
  id: string;
  name: string;
  specialty: string;
  license: string;
  phone: string;
  email: string;
  location: string;
  availability: Availability[];
}

export interface Availability {
  dayOfWeek: number; // 0-6 (Sunday-Saturday)
  startTime: string; // HH:MM format
  endTime: string; // HH:MM format
  isAvailable: boolean;
}

// =============================================================================
// DIAGNOSTIC TYPES
// =============================================================================

export interface ScanRecord extends BaseEntity {
  patientId: string;
  type: 'skin' | 'xray' | 'mri' | 'ct' | 'ultrasound' | 'eeg';
  imageUrl: string;
  analysisResult: AnalysisResult;
  riskLevel: 'low' | 'medium' | 'high';
  confidence: number; // 0-100
  followUpRequired: boolean;
  recommendations: string[];
  date: Date;
}

export interface AnalysisResult {
  diagnosis: string;
  confidence: number;
  keyFindings: string[];
  recommendations: string[];
  riskFactors: string[];
  followUpActions: string[];
}

// =============================================================================
// ANALYTICS TYPES
// =============================================================================

export interface HealthMetrics {
  patientId: string;
  date: Date;
  vitalSigns: VitalSigns;
  symptoms: string[];
  medications: string[];
  activities: string[];
  notes?: string;
}

export interface VitalSigns {
  bloodPressure: {
    systolic: number;
    diastolic: number;
  };
  heartRate: number;
  temperature: number;
  oxygenSaturation: number;
  weight: number;
  height: number;
}

export interface HealthScore {
  overall: number; // 0-100
  categories: {
    cardiovascular: number;
    respiratory: number;
    metabolic: number;
    mental: number;
    physical: number;
  };
  trends: {
    lastWeek: number;
    lastMonth: number;
    lastQuarter: number;
  };
  recommendations: string[];
}

// =============================================================================
// NOTIFICATION TYPES
// =============================================================================

export interface Notification extends BaseEntity {
  userId: string;
  type: 'appointment' | 'medication' | 'test_result' | 'general';
  title: string;
  message: string;
  priority: 'low' | 'medium' | 'high' | 'urgent';
  isRead: boolean;
  actionRequired: boolean;
  actionUrl?: string;
  scheduledFor?: Date;
}

// =============================================================================
// API RESPONSE TYPES
// =============================================================================

export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
  timestamp: Date;
}

export interface PaginatedResponse<T> {
  data: T[];
  pagination: {
    page: number;
    limit: number;
    total: number;
    totalPages: number;
  };
}

// =============================================================================
// EVENT HANDLER TYPES
// =============================================================================

export interface EventHandler<T = Event> {
  (event: T): void;
}

export interface FormEventHandler<T = HTMLFormElement> {
  (event: React.FormEvent<T>): void;
}

export interface ChangeEventHandler<T = HTMLInputElement> {
  (event: React.ChangeEvent<T>): void;
}

export interface ClickEventHandler<T = HTMLButtonElement> {
  (event: React.MouseEvent<T>): void;
}

// =============================================================================
// UTILITY TYPES
// =============================================================================

export type Status = 'loading' | 'success' | 'error' | 'idle';

export type SortDirection = 'asc' | 'desc';

export interface SortConfig {
  field: string;
  direction: SortDirection;
}

export interface FilterConfig {
  field: string;
  value: string | number | boolean;
  operator: 'equals' | 'contains' | 'greater' | 'less' | 'between';
}

// =============================================================================
// AI/ML TYPES
// =============================================================================

export interface PredictionRequest {
  imageUrl: string;
  type: 'skin' | 'xray' | 'mri' | 'ct' | 'ultrasound' | 'eeg';
  patientId?: string;
  metadata?: Record<string, unknown>;
}

export interface PredictionResponse {
  diagnosis: string;
  confidence: number;
  riskLevel: 'low' | 'medium' | 'high';
  findings: string[];
  recommendations: string[];
  followUpRequired: boolean;
  processingTime: number;
}

// =============================================================================
// CLOUD HEALTHCARE TYPES
// =============================================================================

export interface CloudHealthcareConfig {
  provider: 'google' | 'azure' | 'ibm';
  projectId: string;
  location: string;
  datasetId: string;
  apiKey: string;
  timeout: number;
  maxRetries: number;
}

export interface HealthcareDataset {
  id: string;
  name: string;
  location: string;
  projectId: string;
  labels: Record<string, string>;
}

// =============================================================================
// EXPORT ALL TYPES
// =============================================================================

export type {
  BaseEntity,
  User,
  Address,
  Patient,
  MedicalCondition,
  EmergencyContact,
  InsuranceInfo,
  Treatment,
  Medication,
  Procedure,
  Appointment,
  Provider,
  Availability,
  ScanRecord,
  AnalysisResult,
  HealthMetrics,
  VitalSigns,
  HealthScore,
  Notification,
  ApiResponse,
  PaginatedResponse,
  EventHandler,
  FormEventHandler,
  ChangeEventHandler,
  ClickEventHandler,
  Status,
  SortDirection,
  SortConfig,
  FilterConfig,
  PredictionRequest,
  PredictionResponse,
  CloudHealthcareConfig,
  HealthcareDataset,
};
