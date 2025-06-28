
import { BaseApiService, ApiResponse } from './baseApiService';

export interface NotificationConfig {
  twilio?: {
    accountSid: string;
    authToken: string;
    phoneNumber: string;
  };
  sendGrid?: {
    apiKey: string;
    fromEmail: string;
  };
}

export interface SMSNotification {
  to: string;
  message: string;
  priority?: 'low' | 'normal' | 'high';
}

export interface EmailNotification {
  to: string;
  subject: string;
  body: string;
  htmlBody?: string;
  attachments?: Array<{
    filename: string;
    content: string;
    type: string;
  }>;
}

export interface PushNotification {
  userId: string;
  title: string;
  body: string;
  data?: Record<string, any>;
  priority?: 'low' | 'normal' | 'high';
}

export interface NotificationResponse {
  id: string;
  status: 'sent' | 'delivered' | 'failed';
  provider: 'twilio' | 'sendgrid' | 'push';
  timestamp: string;
  error?: string;
}

interface TwilioResponse {
  sid: string;
  status: string;
  error_message?: string;
}

interface SendGridEmailData {
  personalizations: Array<{
    to: Array<{ email: string }>;
  }>;
  from: { email: string };
  subject: string;
  content: Array<{
    type: string;
    value: string;
  }>;
  attachments?: Array<{
    filename: string;
    content: string;
    type: string;
  }>;
}

export class NotificationService extends BaseApiService {
  private twilioConfig?: NotificationConfig['twilio'];
  private sendGridConfig?: NotificationConfig['sendGrid'];

  constructor(config: NotificationConfig) {
    super({
      baseURL: '', // Will use specific APIs
      timeout: 10000
    });

    this.twilioConfig = config.twilio;
    this.sendGridConfig = config.sendGrid;
  }

  async sendSMS(notification: SMSNotification): Promise<ApiResponse<NotificationResponse>> {
    if (!this.twilioConfig) {
      return {
        data: null,
        status: 400,
        statusText: 'Twilio not configured',
        headers: {},
        success: false,
        error: 'Twilio SMS service not configured'
      };
    }

    try {
      const response = await this.post(
        `https://api.twilio.com/2010-04-01/Accounts/${this.twilioConfig.accountSid}/Messages.json`,
        {
          To: notification.to,
          From: this.twilioConfig.phoneNumber,
          Body: notification.message
        },
        {
          headers: {
            'Authorization': `Basic ${btoa(`${this.twilioConfig.accountSid}:${this.twilioConfig.authToken}`)}`,
            'Content-Type': 'application/x-www-form-urlencoded'
          }
        }
      );

      if (response.success && response.data) {
        const twilioData = response.data as TwilioResponse;
        return {
          data: {
            id: twilioData.sid,
            status: twilioData.status === 'sent' ? 'sent' : 'failed',
            provider: 'twilio',
            timestamp: new Date().toISOString(),
            error: twilioData.error_message
          },
          status: response.status,
          statusText: response.statusText,
          headers: response.headers,
          success: response.success
        };
      }

      return {
        data: null,
        status: response.status,
        statusText: response.statusText,
        headers: response.headers,
        success: false,
        error: 'Failed to send SMS'
      };
    } catch (error: any) {
      return {
        data: null,
        status: 500,
        statusText: 'SMS sending failed',
        headers: {},
        success: false,
        error: error.message
      };
    }
  }

  async sendEmail(notification: EmailNotification): Promise<ApiResponse<NotificationResponse>> {
    if (!this.sendGridConfig) {
      return {
        data: null,
        status: 400,
        statusText: 'SendGrid not configured',
        headers: {},
        success: false,
        error: 'SendGrid email service not configured'
      };
    }

    try {
      const emailData: SendGridEmailData = {
        personalizations: [
          {
            to: [{ email: notification.to }]
          }
        ],
        from: { email: this.sendGridConfig.fromEmail },
        subject: notification.subject,
        content: [
          {
            type: notification.htmlBody ? 'text/html' : 'text/plain',
            value: notification.htmlBody || notification.body
          }
        ]
      };

      if (notification.attachments) {
        emailData.attachments = notification.attachments;
      }

      const response = await this.post(
        'https://api.sendgrid.com/v3/mail/send',
        emailData,
        {
          headers: {
            'Authorization': `Bearer ${this.sendGridConfig.apiKey}`,
            'Content-Type': 'application/json'
          }
        }
      );

      if (response.success) {
        return {
          data: {
            id: `email_${Date.now()}`,
            status: 'sent',
            provider: 'sendgrid',
            timestamp: new Date().toISOString()
          },
          status: response.status,
          statusText: response.statusText,
          headers: response.headers,
          success: response.success
        };
      }

      return {
        data: null,
        status: response.status,
        statusText: response.statusText,
        headers: response.headers,
        success: false,
        error: 'Failed to send email'
      };
    } catch (error: any) {
      return {
        data: null,
        status: 500,
        statusText: 'Email sending failed',
        headers: {},
        success: false,
        error: error.message
      };
    }
  }

  async sendPushNotification(notification: PushNotification): Promise<ApiResponse<NotificationResponse>> {
    // This would integrate with a push notification service like Firebase Cloud Messaging
    // For now, we'll simulate the response
    try {
      // Simulate push notification sending
      await this.delay(100);

      return {
        data: {
          id: `push_${Date.now()}`,
          status: 'sent',
          provider: 'push',
          timestamp: new Date().toISOString()
        },
        status: 200,
        statusText: 'OK',
        headers: {},
        success: true
      };
    } catch (error: any) {
      return {
        data: null,
        status: 500,
        statusText: 'Push notification failed',
        headers: {},
        success: false,
        error: error.message
      };
    }
  }

  async sendAppointmentReminder(
    userId: string,
    appointment: {
      date: string;
      time: string;
      doctor: string;
      specialty: string;
    },
    contactInfo: {
      phone?: string;
      email?: string;
    }
  ): Promise<ApiResponse<NotificationResponse[]>> {
    const responses: NotificationResponse[] = [];
    const message = `Reminder: You have an appointment with Dr. ${appointment.doctor} (${appointment.specialty}) on ${appointment.date} at ${appointment.time}. Please arrive 15 minutes early.`;

    // Send SMS if phone provided
    if (contactInfo.phone) {
      const smsResponse = await this.sendSMS({
        to: contactInfo.phone,
        message,
        priority: 'normal'
      });
      if (smsResponse.success && smsResponse.data) {
        responses.push(smsResponse.data);
      }
    }

    // Send email if email provided
    if (contactInfo.email) {
      const emailResponse = await this.sendEmail({
        to: contactInfo.email,
        subject: 'Appointment Reminder - DoctAI Health Hub',
        body: message,
        htmlBody: `
          <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
            <h2 style="color: #2563eb;">Appointment Reminder</h2>
            <p>You have an upcoming appointment:</p>
            <div style="background: #f8fafc; padding: 20px; border-radius: 8px; margin: 20px 0;">
              <p><strong>Doctor:</strong> Dr. ${appointment.doctor}</p>
              <p><strong>Specialty:</strong> ${appointment.specialty}</p>
              <p><strong>Date:</strong> ${appointment.date}</p>
              <p><strong>Time:</strong> ${appointment.time}</p>
            </div>
            <p><strong>Please arrive 15 minutes early.</strong></p>
            <p>If you need to reschedule, please contact us as soon as possible.</p>
          </div>
        `
      });
      if (emailResponse.success && emailResponse.data) {
        responses.push(emailResponse.data);
      }
    }

    // Send push notification
    const pushResponse = await this.sendPushNotification({
      userId,
      title: 'Appointment Reminder',
      body: `Appointment with Dr. ${appointment.doctor} on ${appointment.date}`,
      data: {
        type: 'appointment_reminder',
        appointmentId: appointment.date
      }
    });
    if (pushResponse.success && pushResponse.data) {
      responses.push(pushResponse.data);
    }

    return {
      data: responses,
      status: 200,
      statusText: 'OK',
      headers: {},
      success: responses.length > 0
    };
  }

  async sendMedicationReminder(
    userId: string,
    medication: {
      name: string;
      dosage: string;
      frequency: string;
      time: string;
    },
    contactInfo: {
      phone?: string;
      email?: string;
    }
  ): Promise<ApiResponse<NotificationResponse[]>> {
    const responses: NotificationResponse[] = [];
    const message = `Medication Reminder: Take ${medication.dosage} of ${medication.name} ${medication.frequency} at ${medication.time}.`;

    // Send SMS if phone provided
    if (contactInfo.phone) {
      const smsResponse = await this.sendSMS({
        to: contactInfo.phone,
        message,
        priority: 'high'
      });
      if (smsResponse.success && smsResponse.data) {
        responses.push(smsResponse.data);
      }
    }

    // Send push notification
    const pushResponse = await this.sendPushNotification({
      userId,
      title: 'Medication Reminder',
      body: `Time to take ${medication.name}`,
      data: {
        type: 'medication_reminder',
        medication: medication.name
      },
      priority: 'high'
    });
    if (pushResponse.success && pushResponse.data) {
      responses.push(pushResponse.data);
    }

    return {
      data: responses,
      status: 200,
      statusText: 'OK',
      headers: {},
      success: responses.length > 0
    };
  }

  async sendEmergencyAlert(
    userId: string,
    alert: {
      type: 'health_emergency' | 'medication_overdue' | 'appointment_missed';
      severity: 'low' | 'medium' | 'high' | 'critical';
      message: string;
    },
    contactInfo: {
      phone?: string;
      email?: string;
    }
  ): Promise<ApiResponse<NotificationResponse[]>> {
    const responses: NotificationResponse[] = [];
    const priority = alert.severity === 'critical' ? 'high' : 'normal';

    // Send SMS if phone provided
    if (contactInfo.phone) {
      const smsResponse = await this.sendSMS({
        to: contactInfo.phone,
        message: `URGENT: ${alert.message}`,
        priority
      });
      if (smsResponse.success && smsResponse.data) {
        responses.push(smsResponse.data);
      }
    }

    // Send email if email provided
    if (contactInfo.email) {
      const emailResponse = await this.sendEmail({
        to: contactInfo.email,
        subject: `URGENT: ${alert.type.replace('_', ' ').toUpperCase()}`,
        body: alert.message,
        htmlBody: `
          <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
            <h2 style="color: #dc2626;">URGENT ALERT</h2>
            <div style="background: #fef2f2; border-left: 4px solid #dc2626; padding: 20px; margin: 20px 0;">
              <p><strong>Type:</strong> ${alert.type.replace('_', ' ').toUpperCase()}</p>
              <p><strong>Severity:</strong> ${alert.severity.toUpperCase()}</p>
              <p><strong>Message:</strong> ${alert.message}</p>
            </div>
            <p>Please take immediate action as required.</p>
          </div>
        `
      });
      if (emailResponse.success && emailResponse.data) {
        responses.push(emailResponse.data);
      }
    }

    // Send push notification
    const pushResponse = await this.sendPushNotification({
      userId,
      title: 'URGENT ALERT',
      body: alert.message,
      data: {
        type: alert.type,
        severity: alert.severity
      },
      priority: 'high'
    });
    if (pushResponse.success && pushResponse.data) {
      responses.push(pushResponse.data);
    }

    return {
      data: responses,
      status: 200,
      statusText: 'OK',
      headers: {},
      success: responses.length > 0
    };
  }
}
