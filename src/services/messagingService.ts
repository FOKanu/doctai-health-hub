import { supabase } from './supabaseClient';

export interface Message {
  id: string;
  threadId: string;
  senderId: string;
  senderName: string;
  senderType: 'patient' | 'provider' | 'staff';
  recipientId: string;
  recipientName: string;
  subject: string;
  content: string;
  timestamp: string;
  status: 'sent' | 'delivered' | 'read';
  priority: 'normal' | 'urgent' | 'emergency';
  messageType: 'text' | 'analysis_result' | 'appointment' | 'prescription' | 'lab_result';
  attachments?: MessageAttachment[];
  metadata?: any;
}

export interface MessageThread {
  id: string;
  participants: {
    id: string;
    name: string;
    type: 'patient' | 'provider' | 'staff';
  }[];
  subject: string;
  lastMessage: Message;
  unreadCount: number;
  priority: 'normal' | 'urgent' | 'emergency';
  createdAt: string;
  updatedAt: string;
}

export interface MessageAttachment {
  id: string;
  name: string;
  type: string;
  size: number;
  url: string;
}

export interface Notification {
  id: string;
  userId: string;
  type: 'message' | 'analysis_result' | 'appointment' | 'prescription';
  title: string;
  message: string;
  isRead: boolean;
  timestamp: string;
  metadata?: any;
}

/**
 * Get all message threads for a user
 */
export const getMessageThreads = async (userId: string): Promise<MessageThread[]> => {
  try {
    // For MVP, we'll use mock data since the database structure is still being set up
    const mockThreads: MessageThread[] = [
      {
        id: 'thread_001',
        participants: [
          { id: 'patient_001', name: 'Sarah Johnson', type: 'patient' },
          { id: 'provider_001', name: 'Dr. Sarah Weber', type: 'provider' }
        ],
        subject: 'AI Analysis Results - Skin Lesion',
        lastMessage: {
          id: 'msg_001',
          threadId: 'thread_001',
          senderId: 'provider_001',
          senderName: 'Dr. Sarah Weber',
          senderType: 'provider',
          recipientId: 'patient_001',
          recipientName: 'Sarah Johnson',
          subject: 'AI Analysis Results - Skin Lesion',
          content: 'I\'ve reviewed your AI analysis results. The findings suggest we should schedule a follow-up appointment for further evaluation.',
          timestamp: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(),
          status: 'read',
          priority: 'normal',
          messageType: 'analysis_result'
        },
        unreadCount: 0,
        priority: 'normal',
        createdAt: new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString(),
        updatedAt: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString()
      },
      {
        id: 'thread_002',
        participants: [
          { id: 'patient_002', name: 'Michael Chen', type: 'patient' },
          { id: 'provider_001', name: 'Dr. Sarah Weber', type: 'provider' }
        ],
        subject: 'Appointment Scheduling',
        lastMessage: {
          id: 'msg_002',
          threadId: 'thread_002',
          senderId: 'patient_002',
          senderName: 'Michael Chen',
          senderType: 'patient',
          recipientId: 'provider_001',
          recipientName: 'Dr. Sarah Weber',
          subject: 'Appointment Scheduling',
          content: 'Thank you for the quick response. I\'d like to schedule the follow-up appointment for next week.',
          timestamp: new Date(Date.now() - 1 * 60 * 60 * 1000).toISOString(),
          status: 'delivered',
          priority: 'normal',
          messageType: 'text'
        },
        unreadCount: 1,
        priority: 'normal',
        createdAt: new Date(Date.now() - 48 * 60 * 60 * 1000).toISOString(),
        updatedAt: new Date(Date.now() - 1 * 60 * 60 * 1000).toISOString()
      },
      {
        id: 'thread_003',
        participants: [
          { id: 'patient_003', name: 'Emily Rodriguez', type: 'patient' },
          { id: 'provider_002', name: 'Dr. Michael Brown', type: 'provider' }
        ],
        subject: 'Urgent: MRI Results',
        lastMessage: {
          id: 'msg_003',
          threadId: 'thread_003',
          senderId: 'provider_002',
          senderName: 'Dr. Michael Brown',
          senderType: 'provider',
          recipientId: 'patient_003',
          recipientName: 'Emily Rodriguez',
          subject: 'Urgent: MRI Results',
          content: 'Your MRI results require immediate attention. Please call our office to schedule an urgent consultation.',
          timestamp: new Date(Date.now() - 30 * 60 * 1000).toISOString(),
          status: 'sent',
          priority: 'urgent',
          messageType: 'analysis_result'
        },
        unreadCount: 1,
        priority: 'urgent',
        createdAt: new Date(Date.now() - 72 * 60 * 60 * 1000).toISOString(),
        updatedAt: new Date(Date.now() - 30 * 60 * 1000).toISOString()
      }
    ];

    return mockThreads;
  } catch (error) {
    console.error('Error fetching message threads:', error);
    return [];
  }
};

/**
 * Get messages for a specific thread
 */
export const getThreadMessages = async (threadId: string): Promise<Message[]> => {
  try {
    // Mock messages for the thread
    const mockMessages: Message[] = [
      {
        id: 'msg_001',
        threadId: threadId,
        senderId: 'patient_001',
        senderName: 'Sarah Johnson',
        senderType: 'patient',
        recipientId: 'provider_001',
        recipientName: 'Dr. Sarah Weber',
        subject: 'AI Analysis Results - Skin Lesion',
        content: 'I received my AI analysis results for the skin lesion I uploaded. The AI suggested it might be concerning. Could you please review the results?',
        timestamp: new Date(Date.now() - 4 * 60 * 60 * 1000).toISOString(),
        status: 'read',
        priority: 'normal',
        messageType: 'analysis_result'
      },
      {
        id: 'msg_002',
        threadId: threadId,
        senderId: 'provider_001',
        senderName: 'Dr. Sarah Weber',
        senderType: 'provider',
        recipientId: 'patient_001',
        recipientName: 'Sarah Johnson',
        subject: 'AI Analysis Results - Skin Lesion',
        content: 'I\'ve reviewed your AI analysis results. The findings suggest we should schedule a follow-up appointment for further evaluation. The AI detected some concerning features that warrant a closer look.',
        timestamp: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(),
        status: 'read',
        priority: 'normal',
        messageType: 'analysis_result'
      }
    ];

    return mockMessages;
  } catch (error) {
    console.error('Error fetching thread messages:', error);
    return [];
  }
};

/**
 * Send a new message
 */
export const sendMessage = async (message: Omit<Message, 'id' | 'timestamp' | 'status'>): Promise<boolean> => {
  try {
    console.log('Sending message:', message);

    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 1000));

    // In a real implementation, this would:
    // 1. Save the message to the database
    // 2. Send push notification to recipient
    // 3. Update thread metadata
    // 4. Log the message for compliance

    return true;
  } catch (error) {
    console.error('Error sending message:', error);
    return false;
  }
};

/**
 * Mark message as read
 */
export const markMessageAsRead = async (messageId: string): Promise<boolean> => {
  try {
    console.log('Marking message as read:', messageId);

    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 500));

    return true;
  } catch (error) {
    console.error('Error marking message as read:', error);
    return false;
  }
};

/**
 * Get notifications for a user
 */
export const getNotifications = async (userId: string): Promise<Notification[]> => {
  try {
    const mockNotifications: Notification[] = [
      {
        id: 'notif_001',
        userId: userId,
        type: 'message',
        title: 'New Message from Dr. Sarah Weber',
        message: 'You have a new message about your AI analysis results.',
        isRead: false,
        timestamp: new Date(Date.now() - 30 * 60 * 1000).toISOString()
      },
      {
        id: 'notif_002',
        userId: userId,
        type: 'analysis_result',
        title: 'AI Analysis Complete',
        message: 'Your skin lesion analysis is ready for review.',
        isRead: true,
        timestamp: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString()
      },
      {
        id: 'notif_003',
        userId: userId,
        type: 'appointment',
        title: 'Appointment Reminder',
        message: 'Your appointment with Dr. Michael Brown is tomorrow at 2:00 PM.',
        isRead: false,
        timestamp: new Date(Date.now() - 4 * 60 * 60 * 1000).toISOString()
      }
    ];

    return mockNotifications;
  } catch (error) {
    console.error('Error fetching notifications:', error);
    return [];
  }
};

/**
 * Mark notification as read
 */
export const markNotificationAsRead = async (notificationId: string): Promise<boolean> => {
  try {
    console.log('Marking notification as read:', notificationId);

    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 500));

    return true;
  } catch (error) {
    console.error('Error marking notification as read:', error);
    return false;
  }
};

/**
 * Create a new message thread
 */
export const createMessageThread = async (
  participants: { id: string; name: string; type: 'patient' | 'provider' | 'staff' }[],
  subject: string,
  initialMessage: string,
  senderId: string
): Promise<string | null> => {
  try {
    console.log('Creating message thread:', { participants, subject, initialMessage, senderId });

    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 1000));

    // Return mock thread ID
    return `thread_${Date.now()}`;
  } catch (error) {
    console.error('Error creating message thread:', error);
    return null;
  }
};
