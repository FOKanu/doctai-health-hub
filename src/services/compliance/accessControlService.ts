import { hipaaService } from './hipaaService';

export interface UserRole {
  id: string;
  name: string;
  permissions: Permission[];
  description: string;
  hipaaCompliant: boolean;
}

export interface Permission {
  id: string;
  resource: string;
  action: string;
  conditions?: string[];
}

export interface AccessRequest {
  id: string;
  userId: string;
  resource: string;
  resourceId: string;
  reason: string;
  requestedAt: Date;
  approvedBy?: string;
  approvedAt?: Date;
  status: 'pending' | 'approved' | 'denied' | 'expired';
  expiresAt: Date;
}

export interface SessionInfo {
  userId: string;
  sessionId: string;
  loginTime: Date;
  lastActivity: Date;
  ipAddress: string;
  userAgent: string;
  mfaVerified: boolean;
}

export class AccessControlService {
  private sessions: Map<string, SessionInfo> = new Map();
  private accessRequests: AccessRequest[] = [];

  // Role definitions with HIPAA compliance
  private readonly roles: Record<string, UserRole> = {
    'admin': {
      id: 'admin',
      name: 'System Administrator',
      permissions: [
        { id: 'admin-all', resource: '*', action: '*' }
      ],
      description: 'Full system access for administrative purposes',
      hipaaCompliant: true
    },
    'doctor': {
      id: 'doctor',
      name: 'Physician',
      permissions: [
        { id: 'doctor-read-patient', resource: 'patient', action: 'read' },
        { id: 'doctor-write-medical-record', resource: 'medical-record', action: 'write' },
        { id: 'doctor-read-appointment', resource: 'appointment', action: 'read' },
        { id: 'doctor-write-appointment', resource: 'appointment', action: 'write' },
        { id: 'doctor-read-lab-results', resource: 'lab-results', action: 'read' },
        { id: 'doctor-write-prescription', resource: 'prescription', action: 'write' }
      ],
      description: 'Full access to patient care and medical records',
      hipaaCompliant: true
    },
    'nurse': {
      id: 'nurse',
      name: 'Registered Nurse',
      permissions: [
        { id: 'nurse-read-patient', resource: 'patient', action: 'read' },
        { id: 'nurse-read-medical-record', resource: 'medical-record', action: 'read' },
        { id: 'nurse-write-appointment', resource: 'appointment', action: 'write' },
        { id: 'nurse-read-lab-results', resource: 'lab-results', action: 'read' }
      ],
      description: 'Access to patient care and appointment management',
      hipaaCompliant: true
    },
    'specialist': {
      id: 'specialist',
      name: 'Medical Specialist',
      permissions: [
        { id: 'specialist-read-patient', resource: 'patient', action: 'read' },
        { id: 'specialist-write-medical-record', resource: 'medical-record', action: 'write' },
        { id: 'specialist-read-appointment', resource: 'appointment', action: 'read' }
      ],
      description: 'Specialized medical care access',
      hipaaCompliant: true
    },
    'patient': {
      id: 'patient',
      name: 'Patient',
      permissions: [
        { id: 'patient-read-own-record', resource: 'own-record', action: 'read' },
        { id: 'patient-write-own-appointment', resource: 'own-appointment', action: 'write' },
        { id: 'patient-read-own-appointment', resource: 'own-appointment', action: 'read' }
      ],
      description: 'Access to own medical records and appointments',
      hipaaCompliant: true
    },
    'receptionist': {
      id: 'receptionist',
      name: 'Receptionist',
      permissions: [
        { id: 'receptionist-read-appointment', resource: 'appointment', action: 'read' },
        { id: 'receptionist-write-appointment', resource: 'appointment', action: 'write' },
        { id: 'receptionist-read-patient-basic', resource: 'patient-basic', action: 'read' }
      ],
      description: 'Appointment scheduling and basic patient information',
      hipaaCompliant: true
    }
  };

  // Check if user has permission for specific action
  hasPermission(userId: string, action: string, resource: string, resourceId?: string): boolean {
    const user = this.getUserById(userId);
    if (!user) return false;

    const role = this.roles[user.role];
    if (!role) return false;

    // Check for wildcard permissions
    if (role.permissions.some(p => p.resource === '*' && p.action === '*')) {
      return true;
    }

    // Check specific permissions
    const hasPermission = role.permissions.some(p =>
      (p.resource === resource || p.resource === '*') &&
      (p.action === action || p.action === '*')
    );

    // Log the permission check
    hipaaService.logActivity({
      userId,
      action: `check_permission:${action}`,
      resource,
      resourceId: resourceId || 'unknown',
      ipAddress: 'client-ip',
      userAgent: 'client-agent',
      success: hasPermission
    });

    return hasPermission;
  }

  // Request access to restricted resources
  requestAccess(userId: string, resource: string, resourceId: string, reason: string): AccessRequest {
    const accessRequest: AccessRequest = {
      id: this.generateId(),
      userId,
      resource,
      resourceId,
      reason,
      requestedAt: new Date(),
      status: 'pending',
      expiresAt: new Date(Date.now() + 24 * 60 * 60 * 1000) // 24 hours
    };

    this.accessRequests.push(accessRequest);

    // Log the access request
    hipaaService.logActivity({
      userId,
      action: 'request_access',
      resource,
      resourceId,
      ipAddress: 'client-ip',
      userAgent: 'client-agent',
      success: true,
      details: { reason, requestId: accessRequest.id }
    });

    return accessRequest;
  }

  // Approve or deny access request
  processAccessRequest(requestId: string, approvedBy: string, approved: boolean, reason?: string): AccessRequest | null {
    const request = this.accessRequests.find(r => r.id === requestId);
    if (!request) return null;

    request.status = approved ? 'approved' : 'denied';
    request.approvedBy = approvedBy;
    request.approvedAt = new Date();

    // Log the decision
    hipaaService.logActivity({
      userId: approvedBy,
      action: approved ? 'approve_access' : 'deny_access',
      resource: request.resource,
      resourceId: request.resourceId,
      ipAddress: 'system',
      userAgent: 'access-control-service',
      success: true,
      details: { requestId, reason }
    });

    return request;
  }

  // Session management
  createSession(userId: string, ipAddress: string, userAgent: string): SessionInfo {
    const sessionId = this.generateId();
    const session: SessionInfo = {
      userId,
      sessionId,
      loginTime: new Date(),
      lastActivity: new Date(),
      ipAddress,
      userAgent,
      mfaVerified: false
    };

    this.sessions.set(sessionId, session);

    // Log the session creation
    hipaaService.logActivity({
      userId,
      action: 'create_session',
      resource: 'session',
      resourceId: sessionId,
      ipAddress,
      userAgent,
      success: true
    });

    return session;
  }

  // Validate session
  validateSession(sessionId: string): SessionInfo | null {
    const session = this.sessions.get(sessionId);
    if (!session) return null;

    // Check if session is expired (8 hours)
    const sessionAge = Date.now() - session.lastActivity.getTime();
    if (sessionAge > 8 * 60 * 60 * 1000) {
      this.sessions.delete(sessionId);
      return null;
    }

    // Update last activity
    session.lastActivity = new Date();
    return session;
  }

  // End session
  endSession(sessionId: string): boolean {
    const session = this.sessions.get(sessionId);
    if (!session) return false;

    this.sessions.delete(sessionId);

    // Log session termination
    hipaaService.logActivity({
      userId: session.userId,
      action: 'end_session',
      resource: 'session',
      resourceId: sessionId,
      ipAddress: session.ipAddress,
      userAgent: session.userAgent,
      success: true
    });

    return true;
  }

  // Get active sessions for user
  getUserSessions(userId: string): SessionInfo[] {
    return Array.from(this.sessions.values()).filter(session => session.userId === userId);
  }

  // Force logout all sessions for user (security measure)
  forceLogoutUser(userId: string): number {
    let logoutCount = 0;

    for (const [sessionId, session] of this.sessions.entries()) {
      if (session.userId === userId) {
        this.sessions.delete(sessionId);
        logoutCount++;
      }
    }

    // Log the force logout
    hipaaService.logActivity({
      userId,
      action: 'force_logout',
      resource: 'session',
      resourceId: 'all',
      ipAddress: 'system',
      userAgent: 'access-control-service',
      success: true,
      details: { logoutCount }
    });

    return logoutCount;
  }

  // Get access control report
  generateAccessReport(): {
    totalSessions: number;
    activeSessions: number;
    pendingRequests: number;
    roles: UserRole[];
  } {
    const activeSessions = Array.from(this.sessions.values()).filter(
      session => Date.now() - session.lastActivity.getTime() < 8 * 60 * 60 * 1000
    );

    const pendingRequests = this.accessRequests.filter(
      request => request.status === 'pending' && request.expiresAt > new Date()
    );

    return {
      totalSessions: this.sessions.size,
      activeSessions: activeSessions.length,
      pendingRequests: pendingRequests.length,
      roles: Object.values(this.roles)
    };
  }

  // Private helper methods
  private generateId(): string {
    return Math.random().toString(36).substr(2, 9);
  }

  private getUserById(userId: string): { id: string; role: string } | null {
    // In production, this would fetch from database
    const mockUsers = [
      { id: '1', role: 'doctor' },
      { id: '2', role: 'patient' },
      { id: '3', role: 'admin' },
      { id: '4', role: 'nurse' }
    ];

    return mockUsers.find(user => user.id === userId) || null;
  }
}

export const accessControlService = new AccessControlService();
