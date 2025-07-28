import { hipaaService } from './hipaaService';
import { accessControlService } from './accessControlService';

export interface SecurityContext {
  userId: string;
  sessionId: string;
  ipAddress: string;
  userAgent: string;
  timestamp: Date;
}

export interface SecurityCheck {
  passed: boolean;
  reason?: string;
  action: 'allow' | 'deny' | 'require_approval';
}

export class SecurityMiddleware {
  private readonly rateLimitMap = new Map<string, { count: number; resetTime: number }>();
  private readonly suspiciousIPs = new Set<string>();

  // Main security check for API requests
  async checkSecurity(
    context: SecurityContext,
    resource: string,
    action: string,
    resourceId?: string
  ): Promise<SecurityCheck> {
    const checks = await Promise.all([
      this.checkSession(context),
      this.checkRateLimit(context),
      this.checkSuspiciousActivity(context),
      this.checkPermissions(context, resource, action, resourceId),
      this.checkBusinessHours(context),
      this.checkDataAccess(context, resource, resourceId)
    ]);

    // If any check fails, deny access
    const failedCheck = checks.find(check => !check.passed);
    if (failedCheck) {
      return {
        passed: false,
        reason: failedCheck.reason,
        action: 'deny'
      };
    }

    // Log successful access
    hipaaService.logActivity({
      userId: context.userId,
      action: `api_${action}`,
      resource,
      resourceId: resourceId || 'unknown',
      ipAddress: context.ipAddress,
      userAgent: context.userAgent,
      success: true
    });

    return {
      passed: true,
      action: 'allow'
    };
  }

  // Check if user session is valid
  private async checkSession(context: SecurityContext): Promise<SecurityCheck> {
    const session = accessControlService.validateSession(context.sessionId);

    if (!session) {
      return {
        passed: false,
        reason: 'Invalid or expired session',
        action: 'deny'
      };
    }

    if (session.userId !== context.userId) {
      return {
        passed: false,
        reason: 'Session user mismatch',
        action: 'deny'
      };
    }

    return { passed: true, action: 'allow' };
  }

  // Rate limiting to prevent abuse
  private async checkRateLimit(context: SecurityContext): Promise<SecurityCheck> {
    const key = `${context.userId}:${context.ipAddress}`;
    const now = Date.now();
    const window = 60 * 1000; // 1 minute window
    const maxRequests = 100; // Max requests per minute

    const current = this.rateLimitMap.get(key);

    if (!current || now > current.resetTime) {
      this.rateLimitMap.set(key, { count: 1, resetTime: now + window });
      return { passed: true, action: 'allow' };
    }

    if (current.count >= maxRequests) {
      return {
        passed: false,
        reason: 'Rate limit exceeded',
        action: 'deny'
      };
    }

    current.count++;
    return { passed: true, action: 'allow' };
  }

  // Detect suspicious activity patterns
  private async checkSuspiciousActivity(context: SecurityContext): Promise<SecurityCheck> {
    const suspiciousPatterns = [
      // Multiple failed logins
      context.userId.includes('failed_login'),
      // Access from suspicious IP
      this.suspiciousIPs.has(context.ipAddress),
      // Unusual user agent
      !context.userAgent || context.userAgent.length < 10,
      // Access outside normal hours (simplified check)
      new Date().getHours() < 6 || new Date().getHours() > 22
    ];

    if (suspiciousPatterns.some(pattern => pattern)) {
      // Log suspicious activity
      hipaaService.logActivity({
        userId: context.userId,
        action: 'suspicious_activity_detected',
        resource: 'security',
        resourceId: 'middleware',
        ipAddress: context.ipAddress,
        userAgent: context.userAgent,
        success: false,
        details: { patterns: suspiciousPatterns }
      });

      return {
        passed: false,
        reason: 'Suspicious activity detected',
        action: 'require_approval'
      };
    }

    return { passed: true, action: 'allow' };
  }

  // Check user permissions for the requested action
  private async checkPermissions(
    context: SecurityContext,
    resource: string,
    action: string,
    resourceId?: string
  ): Promise<SecurityCheck> {
    const hasPermission = accessControlService.hasPermission(
      context.userId,
      action,
      resource,
      resourceId
    );

    if (!hasPermission) {
      return {
        passed: false,
        reason: 'Insufficient permissions',
        action: 'deny'
      };
    }

    return { passed: true, action: 'allow' };
  }

  // Check if access is during business hours (for certain resources)
  private async checkBusinessHours(context: SecurityContext): Promise<SecurityCheck> {
    const now = new Date();
    const hour = now.getHours();
    const day = now.getDay();

    // Business hours: Monday-Friday, 8 AM - 6 PM
    const isBusinessHours = day >= 1 && day <= 5 && hour >= 8 && hour < 18;

    // For sensitive operations, require business hours
    const sensitiveOperations = ['delete_patient', 'export_data', 'admin_override'];
    const isSensitiveOperation = sensitiveOperations.some(op =>
      context.userAgent.includes(op)
    );

    if (isSensitiveOperation && !isBusinessHours) {
      return {
        passed: false,
        reason: 'Sensitive operations only allowed during business hours',
        action: 'require_approval'
      };
    }

    return { passed: true, action: 'allow' };
  }

  // Check data access patterns
  private async checkDataAccess(
    context: SecurityContext,
    resource: string,
    resourceId?: string
  ): Promise<SecurityCheck> {
    // Check for unusual data access patterns
    const recentLogs = hipaaService.getAuditLogs({
      userId: context.userId,
      startDate: new Date(Date.now() - 24 * 60 * 60 * 1000) // Last 24 hours
    });

    // Check for bulk data access
    const dataAccessCount = recentLogs.filter(log =>
      log.action.includes('read') && log.resource === resource
    ).length;

    if (dataAccessCount > 100) {
      return {
        passed: false,
        reason: 'Excessive data access detected',
        action: 'require_approval'
      };
    }

    // Check for access to multiple patients in short time
    const uniquePatients = new Set(
      recentLogs
        .filter(log => log.resource === 'patient')
        .map(log => log.resourceId)
    );

    if (uniquePatients.size > 50) {
      return {
        passed: false,
        reason: 'Accessing too many patient records',
        action: 'require_approval'
      };
    }

    return { passed: true, action: 'allow' };
  }

  // Add IP to suspicious list
  addSuspiciousIP(ipAddress: string): void {
    this.suspiciousIPs.add(ipAddress);
  }

  // Remove IP from suspicious list
  removeSuspiciousIP(ipAddress: string): void {
    this.suspiciousIPs.delete(ipAddress);
  }

  // Get security statistics
  getSecurityStats(): {
    totalRequests: number;
    blockedRequests: number;
    suspiciousIPs: number;
    rateLimitBlocks: number;
  } {
    const totalRequests = Array.from(this.rateLimitMap.values())
      .reduce((sum, entry) => sum + entry.count, 0);

    return {
      totalRequests,
      blockedRequests: 0, // In production, track this
      suspiciousIPs: this.suspiciousIPs.size,
      rateLimitBlocks: 0 // In production, track this
    };
  }

  // Clear rate limit for testing
  clearRateLimit(userId: string, ipAddress: string): void {
    const key = `${userId}:${ipAddress}`;
    this.rateLimitMap.delete(key);
  }
}

export const securityMiddleware = new SecurityMiddleware();
