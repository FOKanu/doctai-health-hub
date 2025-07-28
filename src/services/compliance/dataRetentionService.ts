import { hipaaService } from './hipaaService';

export interface RetentionPolicy {
  resourceType: string;
  retentionPeriod: number; // in days
  disposalMethod: 'secure-delete' | 'archive' | 'anonymize';
  legalBasis: string;
  description: string;
}

export interface DataDisposalRecord {
  id: string;
  resourceType: string;
  resourceId: string;
  disposalDate: Date;
  disposalMethod: string;
  disposedBy: string;
  verificationHash: string;
}

export class DataRetentionService {
  private disposalRecords: DataDisposalRecord[] = [];

  // Medical Record Retention Policies
  private readonly retentionPolicies: Record<string, RetentionPolicy> = {
    'medical-records': {
      resourceType: 'medical-records',
      retentionPeriod: 2555, // 7 years
      disposalMethod: 'archive',
      legalBasis: 'HIPAA §164.316(b)(1)',
      description: 'Medical records must be retained for 7 years from last treatment'
    },
    'appointment-records': {
      resourceType: 'appointment-records',
      retentionPeriod: 1095, // 3 years
      disposalMethod: 'secure-delete',
      legalBasis: 'State medical board requirements',
      description: 'Appointment records retained for 3 years'
    },
    'billing-records': {
      resourceType: 'billing-records',
      retentionPeriod: 1825, // 5 years
      disposalMethod: 'archive',
      legalBasis: 'IRS requirements',
      description: 'Billing records retained for 5 years for tax purposes'
    },
    'audit-logs': {
      resourceType: 'audit-logs',
      retentionPeriod: 7300, // 20 years
      disposalMethod: 'archive',
      legalBasis: 'HIPAA §164.316(b)(1)',
      description: 'Audit logs retained for 20 years for compliance'
    },
    'patient-consent': {
      resourceType: 'patient-consent',
      retentionPeriod: 3650, // 10 years
      disposalMethod: 'archive',
      legalBasis: 'HIPAA §164.508',
      description: 'Patient consent forms retained for 10 years'
    },
    'imaging-records': {
      resourceType: 'imaging-records',
      retentionPeriod: 3650, // 10 years
      disposalMethod: 'archive',
      legalBasis: 'State medical board requirements',
      description: 'Medical imaging records retained for 10 years'
    },
    'lab-results': {
      resourceType: 'lab-results',
      retentionPeriod: 2555, // 7 years
      disposalMethod: 'archive',
      legalBasis: 'CLIA regulations',
      description: 'Laboratory results retained for 7 years'
    }
  };

  // Check if data should be disposed
  shouldDisposeData(resourceType: string, creationDate: Date): boolean {
    const policy = this.retentionPolicies[resourceType];
    if (!policy) return false;

    const ageInDays = (Date.now() - creationDate.getTime()) / (1000 * 60 * 60 * 24);
    return ageInDays > policy.retentionPeriod;
  }

  // Get data that needs disposal
  getDataForDisposal(): Array<{
    resourceType: string;
    resourceId: string;
    creationDate: Date;
    policy: RetentionPolicy;
  }> {
    // In production, this would query the database
    const mockData = [
      {
        resourceType: 'appointment-records',
        resourceId: 'apt-001',
        creationDate: new Date('2021-01-01')
      },
      {
        resourceType: 'medical-records',
        resourceId: 'med-001',
        creationDate: new Date('2017-01-01')
      }
    ];

    return mockData
      .filter(item => this.shouldDisposeData(item.resourceType, item.creationDate))
      .map(item => ({
        ...item,
        policy: this.retentionPolicies[item.resourceType]
      }));
  }

  // Secure data disposal
  disposeData(resourceType: string, resourceId: string, disposedBy: string): DataDisposalRecord {
    const policy = this.retentionPolicies[resourceType];
    if (!policy) {
      throw new Error(`No retention policy found for resource type: ${resourceType}`);
    }

    // Perform secure disposal based on method
    switch (policy.disposalMethod) {
      case 'secure-delete':
        this.secureDelete(resourceId);
        break;
      case 'archive':
        this.archiveData(resourceId);
        break;
      case 'anonymize':
        this.anonymizeData(resourceId);
        break;
    }

    // Create disposal record
    const disposalRecord: DataDisposalRecord = {
      id: this.generateId(),
      resourceType,
      resourceId,
      disposalDate: new Date(),
      disposalMethod: policy.disposalMethod,
      disposedBy,
      verificationHash: this.generateVerificationHash(resourceId, policy.disposalMethod)
    };

    this.disposalRecords.push(disposalRecord);

    // Log the disposal activity
    hipaaService.logActivity({
      userId: disposedBy,
      action: 'dispose_data',
      resource: resourceType,
      resourceId,
      ipAddress: 'system',
      userAgent: 'data-retention-service',
      success: true,
      details: {
        disposalMethod: policy.disposalMethod,
        retentionPolicy: policy
      }
    });

    return disposalRecord;
  }

  // Get disposal records for compliance reporting
  getDisposalRecords(filters?: {
    resourceType?: string;
    startDate?: Date;
    endDate?: Date;
    disposedBy?: string;
  }): DataDisposalRecord[] {
    let records = [...this.disposalRecords];

    if (filters?.resourceType) {
      records = records.filter(record => record.resourceType === filters.resourceType);
    }

    if (filters?.startDate) {
      records = records.filter(record => record.disposalDate >= filters.startDate!);
    }

    if (filters?.endDate) {
      records = records.filter(record => record.disposalDate <= filters.endDate!);
    }

    if (filters?.disposedBy) {
      records = records.filter(record => record.disposedBy === filters.disposedBy);
    }

    return records;
  }

  // Compliance reporting
  generateRetentionReport(): {
    totalRecords: number;
    recordsForDisposal: number;
    retentionPolicies: RetentionPolicy[];
    disposalRecords: DataDisposalRecord[];
  } {
    const dataForDisposal = this.getDataForDisposal();

    return {
      totalRecords: 1000, // In production, get from database
      recordsForDisposal: dataForDisposal.length,
      retentionPolicies: Object.values(this.retentionPolicies),
      disposalRecords: this.disposalRecords
    };
  }

  // Private methods for disposal operations
  private secureDelete(resourceId: string): void {
    // In production, this would:
    // 1. Overwrite data with random bytes
    // 2. Delete from database
    // 3. Remove from backups
    // 4. Verify deletion
    console.log(`Securely deleting resource: ${resourceId}`);
  }

  private archiveData(resourceId: string): void {
    // In production, this would:
    // 1. Move to long-term storage
    // 2. Update access controls
    // 3. Create archive index
    console.log(`Archiving resource: ${resourceId}`);
  }

  private anonymizeData(resourceId: string): void {
    // In production, this would:
    // 1. Remove all PII
    // 2. Hash remaining data
    // 3. Update data structure
    console.log(`Anonymizing resource: ${resourceId}`);
  }

  private generateId(): string {
    return Math.random().toString(36).substr(2, 9);
  }

  private generateVerificationHash(resourceId: string, disposalMethod: string): string {
    const data = `${resourceId}-${disposalMethod}-${Date.now()}`;
    return require('crypto').createHash('sha256').update(data).digest('hex');
  }
}

export const dataRetentionService = new DataRetentionService();
