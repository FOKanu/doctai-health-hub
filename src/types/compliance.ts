// HIPAA Compliance Types

export interface HIPAAComplianceConfig {
  encryption: {
    algorithm: string;
    keyRotationDays: number;
    keyStorage: 'aws-kms' | 'azure-keyvault' | 'gcp-kms' | 'local';
  };
  audit: {
    retentionDays: number;
    logLevel: 'basic' | 'detailed' | 'comprehensive';
    realTimeAlerts: boolean;
  };
  access: {
    sessionTimeoutMinutes: number;
    maxFailedLogins: number;
    requireMFA: boolean;
    businessHoursOnly: boolean;
  };
  retention: {
    medicalRecords: number; // days
    appointmentRecords: number;
    billingRecords: number;
    auditLogs: number;
    patientConsent: number;
  };
}

export interface ComplianceViolation {
  id: string;
  timestamp: Date;
  userId: string;
  violationType: 'unauthorized_access' | 'data_breach' | 'retention_violation' | 'audit_failure';
  severity: 'low' | 'medium' | 'high' | 'critical';
  description: string;
  resource: string;
  resourceId: string;
  ipAddress: string;
  userAgent: string;
  resolved: boolean;
  resolutionNotes?: string;
  resolvedBy?: string;
  resolvedAt?: Date;
}

export interface ComplianceReport {
  id: string;
  generatedAt: Date;
  period: {
    start: Date;
    end: Date;
  };
  metrics: {
    totalViolations: number;
    criticalViolations: number;
    resolvedViolations: number;
    complianceScore: number;
    auditLogsGenerated: number;
    dataDisposalEvents: number;
  };
  violations: ComplianceViolation[];
  recommendations: string[];
  nextReviewDate: Date;
}

export interface DataClassification {
  level: 'public' | 'internal' | 'confidential' | 'restricted' | 'phi';
  description: string;
  encryptionRequired: boolean;
  accessControls: string[];
  retentionPolicy: string;
  disposalMethod: 'secure-delete' | 'archive' | 'anonymize';
}

export interface ConsentRecord {
  id: string;
  patientId: string;
  consentType: 'treatment' | 'billing' | 'research' | 'marketing';
  granted: boolean;
  grantedAt: Date;
  expiresAt?: Date;
  revokedAt?: Date;
  revokedBy?: string;
  consentText: string;
  version: string;
  ipAddress: string;
  userAgent: string;
}

export interface BreachNotification {
  id: string;
  breachId: string;
  notificationType: 'internal' | 'patient' | 'regulatory' | 'media';
  recipient: string;
  sentAt: Date;
  content: string;
  status: 'pending' | 'sent' | 'delivered' | 'failed';
  deliveryConfirmation?: string;
}

export interface SecurityIncident {
  id: string;
  timestamp: Date;
  incidentType: 'data_breach' | 'unauthorized_access' | 'system_compromise' | 'physical_security';
  severity: 'low' | 'medium' | 'high' | 'critical';
  description: string;
  affectedUsers: number;
  affectedRecords: number;
  containmentStatus: 'active' | 'contained' | 'resolved';
  investigationStatus: 'open' | 'in_progress' | 'closed';
  assignedTo?: string;
  resolutionNotes?: string;
  lessonsLearned?: string[];
}

export interface ComplianceTraining {
  id: string;
  userId: string;
  trainingType: 'hipaa_basics' | 'security_awareness' | 'data_handling' | 'incident_response';
  completedAt: Date;
  score: number;
  expiresAt: Date;
  certificateId?: string;
  instructor?: string;
}

export interface RiskAssessment {
  id: string;
  assessmentDate: Date;
  assessor: string;
  riskAreas: {
    dataSecurity: number; // 1-10 scale
    accessControl: number;
    auditTrail: number;
    physicalSecurity: number;
    businessContinuity: number;
    vendorManagement: number;
  };
  overallRiskScore: number;
  riskLevel: 'low' | 'medium' | 'high' | 'critical';
  recommendations: string[];
  nextAssessmentDate: Date;
}

export interface VendorCompliance {
  vendorId: string;
  vendorName: string;
  serviceType: 'cloud_storage' | 'email' | 'analytics' | 'billing' | 'telemedicine';
  hipaaCompliant: boolean;
  baaSigned: boolean;
  baaExpiryDate?: Date;
  lastAssessmentDate: Date;
  riskLevel: 'low' | 'medium' | 'high';
  complianceNotes: string[];
}

export interface ComplianceMetrics {
  // Real-time metrics
  activeUsers: number;
  activeSessions: number;
  failedLoginAttempts: number;
  suspiciousActivities: number;

  // Compliance scores
  overallComplianceScore: number;
  securityScore: number;
  privacyScore: number;
  auditScore: number;

  // Violations
  openViolations: number;
  criticalViolations: number;
  resolvedViolations: number;

  // Data management
  recordsForDisposal: number;
  retentionCompliance: number; // percentage
  encryptionCoverage: number; // percentage

  // Training
  usersWithExpiredTraining: number;
  trainingCompletionRate: number;

  // Vendors
  nonCompliantVendors: number;
  expiringBAAs: number;
}

export interface ComplianceAlert {
  id: string;
  timestamp: Date;
  alertType: 'security' | 'privacy' | 'audit' | 'retention' | 'training';
  severity: 'info' | 'warning' | 'error' | 'critical';
  title: string;
  message: string;
  affectedUsers?: string[];
  affectedResources?: string[];
  actionRequired: boolean;
  actionDescription?: string;
  acknowledged: boolean;
  acknowledgedBy?: string;
  acknowledgedAt?: Date;
}

export interface ComplianceSchedule {
  id: string;
  taskType: 'audit' | 'training' | 'assessment' | 'review' | 'disposal';
  frequency: 'daily' | 'weekly' | 'monthly' | 'quarterly' | 'yearly';
  lastRun: Date;
  nextRun: Date;
  assignedTo?: string;
  status: 'active' | 'paused' | 'completed';
  description: string;
  priority: 'low' | 'medium' | 'high' | 'critical';
}

// Utility types for compliance operations
export type ComplianceAction =
  | 'view_patient_record'
  | 'edit_patient_record'
  | 'delete_patient_record'
  | 'export_patient_data'
  | 'access_audit_logs'
  | 'manage_users'
  | 'configure_system'
  | 'view_reports'
  | 'manage_consent'
  | 'dispose_data';

export type ComplianceResource =
  | 'patient_record'
  | 'medical_record'
  | 'appointment'
  | 'billing'
  | 'audit_log'
  | 'user_account'
  | 'system_config'
  | 'consent_record'
  | 'training_record'
  | 'vendor_record';

export type ComplianceStatus =
  | 'compliant'
  | 'non_compliant'
  | 'under_review'
  | 'pending_assessment'
  | 'requires_action';

// Constants for compliance
export const COMPLIANCE_CONSTANTS = {
  // HIPAA Requirements
  MINIMUM_RETENTION_PERIODS: {
    MEDICAL_RECORDS: 2555, // 7 years
    APPOINTMENT_RECORDS: 1095, // 3 years
    BILLING_RECORDS: 1825, // 5 years
    AUDIT_LOGS: 7300, // 20 years
    PATIENT_CONSENT: 3650, // 10 years
  },

  // Security thresholds
  MAX_FAILED_LOGINS: 5,
  SESSION_TIMEOUT_MINUTES: 480, // 8 hours
  PASSWORD_EXPIRY_DAYS: 90,
  MFA_REQUIRED_FOR: ['admin', 'doctor', 'nurse'],

  // Audit requirements
  AUDIT_LOG_RETENTION_DAYS: 7300, // 20 years
  REAL_TIME_ALERT_THRESHOLDS: {
    MULTIPLE_FAILED_LOGINS: 3,
    UNAUTHORIZED_ACCESS_ATTEMPTS: 1,
    BULK_DATA_EXPORT: 100,
    ACCESS_OUTSIDE_HOURS: 1,
  },

  // Business hours (for sensitive operations)
  BUSINESS_HOURS: {
    START: 8, // 8 AM
    END: 18, // 6 PM
    DAYS: [1, 2, 3, 4, 5], // Monday-Friday
  },
} as const;
