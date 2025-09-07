import { supabase } from './supabaseClient';

export interface PendingAnalysis {
  id: string;
  patientId: string;
  patientName: string;
  imageType: string;
  prediction: string;
  confidence: number;
  riskLevel: 'low' | 'medium' | 'high';
  timestamp: string;
  status: 'pending' | 'approved' | 'rejected' | 'needs_review';
  imageUrl?: string;
  metadata?: any;
}

export interface ProviderReview {
  id: string;
  analysisId: string;
  providerId: string;
  status: 'approved' | 'rejected' | 'needs_review';
  clinicalNotes: string;
  finalDiagnosis: string;
  recommendations: string[];
  reviewedAt: string;
}

/**
 * Get all pending AI analyses for provider review
 */
export const getPendingAnalyses = async (providerId?: string): Promise<PendingAnalysis[]> => {
  try {
    // For MVP, we'll use mock data since the database structure is still being set up
    const mockPendingAnalyses: PendingAnalysis[] = [
      {
        id: 'analysis_001',
        patientId: 'patient_001',
        patientName: 'Sarah Johnson',
        imageType: 'skin_lesion',
        prediction: 'malignant',
        confidence: 0.87,
        riskLevel: 'high',
        timestamp: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(), // 2 hours ago
        status: 'pending',
        metadata: {
          findings: 'Irregular border, color variation, diameter >6mm',
          recommendations: ['Immediate dermatology consultation', 'Biopsy recommended']
        }
      },
      {
        id: 'analysis_002',
        patientId: 'patient_002',
        patientName: 'Michael Chen',
        imageType: 'xray',
        prediction: 'benign',
        confidence: 0.92,
        riskLevel: 'low',
        timestamp: new Date(Date.now() - 4 * 60 * 60 * 1000).toISOString(), // 4 hours ago
        status: 'pending',
        metadata: {
          findings: 'Normal lung fields, no acute abnormalities',
          recommendations: ['Routine follow-up in 6 months']
        }
      },
      {
        id: 'analysis_003',
        patientId: 'patient_003',
        patientName: 'Emily Rodriguez',
        imageType: 'mri',
        prediction: 'malignant',
        confidence: 0.78,
        riskLevel: 'high',
        timestamp: new Date(Date.now() - 6 * 60 * 60 * 1000).toISOString(), // 6 hours ago
        status: 'needs_review',
        metadata: {
          findings: 'Suspicious mass in right breast, requires further evaluation',
          recommendations: ['Mammography follow-up', 'Consider biopsy']
        }
      }
    ];

    // Filter by provider if specified (for future use)
    return mockPendingAnalyses;
  } catch (error) {
    console.error('Error fetching pending analyses:', error);
    return [];
  }
};

/**
 * Submit provider review for an AI analysis
 */
export const submitProviderReview = async (review: Omit<ProviderReview, 'id' | 'reviewedAt'>): Promise<boolean> => {
  try {
    // For MVP, we'll simulate the review submission
    console.log('Submitting provider review:', review);

    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 1000));

    // In a real implementation, this would:
    // 1. Save the review to the database
    // 2. Update the analysis status
    // 3. Send notification to patient
    // 4. Log the review for compliance

    return true;
  } catch (error) {
    console.error('Error submitting provider review:', error);
    return false;
  }
};

/**
 * Get analysis details by ID
 */
export const getAnalysisDetails = async (analysisId: string): Promise<PendingAnalysis | null> => {
  try {
    const pendingAnalyses = await getPendingAnalyses();
    return pendingAnalyses.find(analysis => analysis.id === analysisId) || null;
  } catch (error) {
    console.error('Error fetching analysis details:', error);
    return null;
  }
};

/**
 * Get provider review history
 */
export const getProviderReviewHistory = async (providerId: string): Promise<ProviderReview[]> => {
  try {
    // Mock review history
    const mockReviews: ProviderReview[] = [
      {
        id: 'review_001',
        analysisId: 'analysis_001',
        providerId: providerId,
        status: 'approved',
        clinicalNotes: 'AI analysis appears accurate. Patient shows classic signs of melanoma.',
        finalDiagnosis: 'Suspicious lesion - biopsy recommended',
        recommendations: ['Immediate dermatology consultation', 'Biopsy within 1 week'],
        reviewedAt: new Date(Date.now() - 1 * 60 * 60 * 1000).toISOString()
      }
    ];

    return mockReviews;
  } catch (error) {
    console.error('Error fetching review history:', error);
    return [];
  }
};
