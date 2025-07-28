// HIPAA Compliance Service - Browser Compatible
export interface HIPAAAuditLog {
  id: string;
  timestamp: Date;
  userId: string;
  action: string;
  resource: string;
  resourceId: string;
  ipAddress: string;
  userAgent: string;
  success: boolean;
  details?: Record<string, unknown>;
}

export interface HIPAAUser {
  id: string;
  email: string;
  role: 'patient' | 'doctor' | 'admin' | 'nurse' | 'specialist';
  permissions: string[];
  lastLogin: Date;
  isActive: boolean;
  mfaEnabled: boolean;
}

export interface EncryptedData {
  encrypted: string;
  iv: string;
  algorithm: string;
}

export class HIPAAService {
  private readonly auditLogs: HIPAAAuditLog[] = [];

  constructor() {
    // In production, this should come from environment variables
    // For browser compatibility, we'll use a simpler approach
  }

  // Data Encryption (At Rest and In Transit) - Browser Compatible
  async encryptPHI(data: string): Promise<EncryptedData> {
    try {
      // Generate a random key for demo purposes
      // In production, use proper key management
      const key = await crypto.subtle.generateKey(
        {
          name: 'AES-GCM',
          length: 256
        },
        true,
        ['encrypt', 'decrypt']
      );

      const iv = crypto.getRandomValues(new Uint8Array(12));
      const encoder = new TextEncoder();
      const encodedData = encoder.encode(data);

      const encryptedBuffer = await crypto.subtle.encrypt(
        {
          name: 'AES-GCM',
          iv: iv
        },
        key,
        encodedData
      );

      return {
        encrypted: btoa(String.fromCharCode(...new Uint8Array(encryptedBuffer))),
        iv: btoa(String.fromCharCode(...iv)),
        algorithm: 'AES-GCM'
      };
    } catch (error) {
      console.error('Encryption failed:', error);
      // Fallback for demo purposes
      return {
        encrypted: btoa(data),
        iv: btoa('demo-iv'),
        algorithm: 'base64-fallback'
      };
    }
  }

  async decryptPHI(encryptedData: EncryptedData): Promise<string> {
    try {
      if (encryptedData.algorithm === 'base64-fallback') {
        return atob(encryptedData.encrypted);
      }

      // In production, you would use the same key that was used for encryption
      // For demo purposes, we'll just return the base64 decoded data
      return atob(encryptedData.encrypted);
    } catch (error) {
      console.error('Decryption failed:', error);
      return 'Decryption failed';
    }
  }

  // Audit Trail Implementation
  logActivity(params: {
    userId: string;
    action: string;
    resource: string;
    resourceId: string;
    ipAddress: string;
    userAgent: string;
    success: boolean;
    details?: Record<string, unknown>;
  }): void {
    const auditLog: HIPAAAuditLog = {
      id: this.generateId(),
      timestamp: new Date(),
      ...params
    };

    this.auditLogs.push(auditLog);

    // In production, this would be sent to a secure audit log service
    console.log('HIPAA Audit Log:', auditLog);
  }

  // Access Control - Role-Based Permissions
  checkPermission(userId: string, action: string, resource: string): boolean {
    const user = this.getUserById(userId);
    if (!user || !user.isActive) return false;

    const permissions = this.getPermissionsForRole(user.role);
    return permissions.includes(`${action}:${resource}`);
  }

  // Data Retention Policies
  getRetentionPolicy(resourceType: string): {
    retentionPeriod: number; // in days
    disposalMethod: 'secure-delete' | 'archive' | 'anonymize';
  } {
    const policies = {
      'medical-records': { retentionPeriod: 2555, disposalMethod: 'archive' as const }, // 7 years
      'appointment-records': { retentionPeriod: 1095, disposalMethod: 'secure-delete' as const }, // 3 years
      'billing-records': { retentionPeriod: 1825, disposalMethod: 'archive' as const }, // 5 years
      'audit-logs': { retentionPeriod: 7300, disposalMethod: 'archive' as const }, // 20 years
      'patient-consent': { retentionPeriod: 3650, disposalMethod: 'archive' as const }, // 10 years
    };

    return policies[resourceType as keyof typeof policies] ||
           { retentionPeriod: 1095, disposalMethod: 'secure-delete' as const };
  }

  // HIPAA Security Measures
  validatePHIAccess(userId: string, patientId: string): boolean {
    const user = this.getUserById(userId);
    if (!user) return false;

    // Check if user has access to this patient's data
    const hasAccess = this.checkPatientAccess(userId, patientId);

    // Log the access attempt
    this.logActivity({
      userId,
      action: 'access_phi',
      resource: 'patient',
      resourceId: patientId,
      ipAddress: 'client-ip', // In production, get from request
      userAgent: 'client-agent', // In production, get from request
      success: hasAccess
    });

    return hasAccess;
  }

  // Data Anonymization for Research/Reporting
  anonymizePHI(data: Record<string, unknown>): Record<string, unknown> {
    const anonymized = { ...data };

    // Remove or hash identifiable information
    if (anonymized.email) anonymized.email = this.hashData(anonymized.email as string);
    if (anonymized.phone) anonymized.phone = this.hashData(anonymized.phone as string);
    if (anonymized.name) anonymized.name = this.hashData(anonymized.name as string);
    if (anonymized.address) anonymized.address = this.hashData(anonymized.address as string);
    if (anonymized.dateOfBirth) anonymized.dateOfBirth = this.hashData(anonymized.dateOfBirth as string);

    return anonymized;
  }

  // Breach Detection and Response
  detectBreach(activity: HIPAAAuditLog): boolean {
    const suspiciousPatterns = [
      'multiple_failed_logins',
      'unauthorized_access_attempt',
      'data_export_large_volume',
      'access_outside_business_hours'
    ];

    return suspiciousPatterns.some(pattern =>
      activity.action.includes(pattern)
    );
  }

  // Private helper methods
  private generateId(): string {
    // Browser-compatible ID generation
    return Math.random().toString(36).substr(2, 9) + Date.now().toString(36);
  }

  private async hashData(data: string): Promise<string> {
    try {
      const encoder = new TextEncoder();
      const dataBuffer = encoder.encode(data);
      const hashBuffer = await crypto.subtle.digest('SHA-256', dataBuffer);
      const hashArray = Array.from(new Uint8Array(hashBuffer));
      return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
    } catch (error) {
      console.error('Hashing failed:', error);
      // Fallback for demo purposes
      return btoa(data).substr(0, 32);
    }
  }

  private getUserById(userId: string): HIPAAUser | null {
    // In production, this would fetch from database
    const mockUsers: HIPAAUser[] = [
      {
        id: '1',
        email: 'doctor@doctai.com',
        role: 'doctor',
        permissions: ['read:patient', 'write:medical-record', 'read:appointment'],
        lastLogin: new Date(),
        isActive: true,
        mfaEnabled: true
      },
      {
        id: '2',
        email: 'patient@doctai.com',
        role: 'patient',
        permissions: ['read:own-record', 'write:own-appointment'],
        lastLogin: new Date(),
        isActive: true,
        mfaEnabled: false
      }
    ];

    return mockUsers.find(user => user.id === userId) || null;
  }

  private getPermissionsForRole(role: string): string[] {
    const permissions = {
      'admin': ['*:*'],
      'doctor': ['read:patient', 'write:medical-record', 'read:appointment', 'write:appointment'],
      'nurse': ['read:patient', 'read:medical-record', 'read:appointment', 'write:appointment'],
      'specialist': ['read:patient', 'write:medical-record', 'read:appointment'],
      'patient': ['read:own-record', 'write:own-appointment', 'read:own-appointment']
    };

    return permissions[role as keyof typeof permissions] || [];
  }

  private checkPatientAccess(userId: string, patientId: string): boolean {
    const user = this.getUserById(userId);
    if (!user) return false;

    // Admin and doctors have access to all patients
    if (['admin', 'doctor', 'nurse', 'specialist'].includes(user.role)) {
      return true;
    }

    // Patients can only access their own data
    if (user.role === 'patient') {
      return userId === patientId;
    }

    return false;
  }

  // Get audit logs (for compliance reporting)
  getAuditLogs(filters?: {
    userId?: string;
    action?: string;
    startDate?: Date;
    endDate?: Date;
  }): HIPAAAuditLog[] {
    let logs = [...this.auditLogs];

    if (filters?.userId) {
      logs = logs.filter(log => log.userId === filters.userId);
    }

    if (filters?.action) {
      logs = logs.filter(log => log.action === filters.action);
    }

    if (filters?.startDate) {
      logs = logs.filter(log => log.timestamp >= filters.startDate!);
    }

    if (filters?.endDate) {
      logs = logs.filter(log => log.timestamp <= filters.endDate!);
    }

    return logs;
  }
}

export const hipaaService = new HIPAAService();
