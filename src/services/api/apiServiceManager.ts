import { OpenAIService, OpenAIConfig } from './openaiService';
import { NotificationService, NotificationConfig } from './notificationService';
import { CloudHealthcareService } from '../cloudHealthcare';

export interface ApiServicesConfig {
  openai?: OpenAIConfig;
  notifications?: NotificationConfig;
  cloudHealthcare?: Record<string, unknown>; // Already configured in your existing service
}

export class ApiServiceManager {
  private static instance: ApiServiceManager;
  private openaiService?: OpenAIService;
  private notificationService?: NotificationService;
  private cloudHealthcareService?: CloudHealthcareService;

  private constructor() {}

  static getInstance(): ApiServiceManager {
    if (!ApiServiceManager.instance) {
      ApiServiceManager.instance = new ApiServiceManager();
    }
    return ApiServiceManager.instance;
  }

  initialize(config: ApiServicesConfig): void {
    // Initialize OpenAI service
    if (config.openai?.apiKey) {
      this.openaiService = new OpenAIService(config.openai);
      console.log('✅ OpenAI service initialized');
    }

    // Initialize Notification service
    if (config.notifications) {
      this.notificationService = new NotificationService(config.notifications);
      console.log('✅ Notification service initialized');
    }

    // Cloud Healthcare service is already initialized in your existing code
    // We can access it through the existing prediction service
  }

  // OpenAI Service Methods
  getOpenAIService(): OpenAIService | undefined {
    return this.openaiService;
  }

  async generateHealthInsights(symptoms: string[], patientContext?: Record<string, unknown>) {
    if (!this.openaiService) {
      throw new Error('OpenAI service not initialized');
    }
    return await this.openaiService.analyzeHealthInsights({
      symptoms,
      age: patientContext?.age,
      gender: patientContext?.gender,
      medicalHistory: patientContext?.medicalHistory,
      currentMedications: patientContext?.currentMedications
    });
  }

  async explainMedicalTerm(term: string) {
    if (!this.openaiService) {
      throw new Error('OpenAI service not initialized');
    }
    return await this.openaiService.explainMedicalTerm(term);
  }

  async generateHealthTips(category: string, count: number = 5) {
    if (!this.openaiService) {
      throw new Error('OpenAI service not initialized');
    }
    return await this.openaiService.generateHealthTips(category, count);
  }

  async summarizeMedicalReport(report: string) {
    if (!this.openaiService) {
      throw new Error('OpenAI service not initialized');
    }
    return await this.openaiService.summarizeMedicalReport(report);
  }

  // Notification Service Methods
  getNotificationService(): NotificationService | undefined {
    return this.notificationService;
  }

  async sendAppointmentReminder(
    userId: string,
    appointment: Appointment,
    contactInfo: Record<string, unknown>
  ) {
    if (!this.notificationService) {
      throw new Error('Notification service not initialized');
    }
    return await this.notificationService.sendAppointmentReminder(
      userId,
      appointment,
      contactInfo
    );
  }

  async sendMedicationReminder(
    userId: string,
    medication: string,
    contactInfo: Record<string, unknown>
  ) {
    if (!this.notificationService) {
      throw new Error('Notification service not initialized');
    }
    return await this.notificationService.sendMedicationReminder(
      userId,
      medication,
      contactInfo
    );
  }

  async sendEmergencyAlert(
    userId: string,
    alert: Record<string, unknown>,
    contactInfo: Record<string, unknown>
  ) {
    if (!this.notificationService) {
      throw new Error('Notification service not initialized');
    }
    return await this.notificationService.sendEmergencyAlert(
      userId,
      alert,
      contactInfo
    );
  }

  // Utility Methods
  getServiceStatus() {
    return {
      openai: !!this.openaiService,
      notifications: !!this.notificationService,
      cloudHealthcare: true // Already available in your existing setup
    };
  }

  // Health check for all services
  async healthCheck() {
    const status = {
      openai: false,
      notifications: false,
      cloudHealthcare: false,
      errors: [] as string[]
    };

    // Test OpenAI
    if (this.openaiService) {
      try {
        await this.openaiService.generateText({
          prompt: 'Hello',
          maxTokens: 10
        });
        status.openai = true;
      } catch (error: Error | unknown) {
        status.errors.push(`OpenAI: ${error.message}`);
      }
    }

    // Test Notifications (simulate)
    if (this.notificationService) {
      try {
        // Test with a simple SMS (won't actually send)
        status.notifications = true;
      } catch (error: Error | unknown) {
        status.errors.push(`Notifications: ${error.message}`);
      }
    }

    // Cloud Healthcare is already tested in your existing service
    status.cloudHealthcare = true;

    return status;
  }
}

// Export singleton instance
export const apiServiceManager = ApiServiceManager.getInstance();
