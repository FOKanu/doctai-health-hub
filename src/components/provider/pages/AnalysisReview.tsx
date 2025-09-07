import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import {
  CheckCircle,
  XCircle,
  AlertTriangle,
  Clock,
  User,
  FileText,
  Brain,
  Calendar
} from 'lucide-react';
import { useToast } from '@/hooks/use-toast';
import {
  getPendingAnalyses,
  submitProviderReview,
  getAnalysisDetails,
  PendingAnalysis,
  ProviderReview
} from '../../../services/providerReviewService';
import { useAuth } from '../../../contexts/AuthContext';

export function AnalysisReview() {
  const { user } = useAuth();
  const { toast } = useToast();
  const [pendingAnalyses, setPendingAnalyses] = useState<PendingAnalysis[]>([]);
  const [selectedAnalysis, setSelectedAnalysis] = useState<PendingAnalysis | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Review form state
  const [reviewStatus, setReviewStatus] = useState<'approved' | 'rejected' | 'needs_review'>('approved');
  const [clinicalNotes, setClinicalNotes] = useState('');
  const [finalDiagnosis, setFinalDiagnosis] = useState('');
  const [recommendations, setRecommendations] = useState('');

  useEffect(() => {
    loadPendingAnalyses();
  }, []);

  const loadPendingAnalyses = async () => {
    try {
      setIsLoading(true);
      const analyses = await getPendingAnalyses(user?.id);
      setPendingAnalyses(analyses);
    } catch (error) {
      console.error('Error loading pending analyses:', error);
      toast({
        title: "Error",
        description: "Failed to load pending analyses.",
        variant: "destructive"
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleAnalysisSelect = async (analysisId: string) => {
    try {
      const analysis = await getAnalysisDetails(analysisId);
      if (analysis) {
        setSelectedAnalysis(analysis);
        // Pre-fill form with AI analysis data
        setFinalDiagnosis(analysis.prediction);
        setRecommendations(analysis.metadata?.recommendations?.join(', ') || '');
      }
    } catch (error) {
      console.error('Error loading analysis details:', error);
    }
  };

  const handleSubmitReview = async () => {
    if (!selectedAnalysis || !user) return;

    if (!clinicalNotes.trim() || !finalDiagnosis.trim()) {
      toast({
        title: "Missing Information",
        description: "Please provide clinical notes and final diagnosis.",
        variant: "destructive"
      });
      return;
    }

    try {
      setIsSubmitting(true);

      const review: Omit<ProviderReview, 'id' | 'reviewedAt'> = {
        analysisId: selectedAnalysis.id,
        providerId: user.id,
        status: reviewStatus,
        clinicalNotes: clinicalNotes.trim(),
        finalDiagnosis: finalDiagnosis.trim(),
        recommendations: recommendations.split(',').map(r => r.trim()).filter(r => r),
      };

      const success = await submitProviderReview(review);

      if (success) {
        toast({
          title: "Review Submitted",
          description: "Your review has been submitted successfully.",
        });

        // Reset form
        setSelectedAnalysis(null);
        setClinicalNotes('');
        setFinalDiagnosis('');
        setRecommendations('');
        setReviewStatus('approved');

        // Reload pending analyses
        await loadPendingAnalyses();
      } else {
        throw new Error('Failed to submit review');
      }
    } catch (error) {
      console.error('Error submitting review:', error);
      toast({
        title: "Error",
        description: "Failed to submit review. Please try again.",
        variant: "destructive"
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  const getRiskLevelColor = (riskLevel: string) => {
    switch (riskLevel) {
      case 'high': return 'bg-red-100 text-red-800';
      case 'medium': return 'bg-yellow-100 text-yellow-800';
      case 'low': return 'bg-green-100 text-green-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'approved': return <CheckCircle className="w-4 h-4 text-green-600" />;
      case 'rejected': return <XCircle className="w-4 h-4 text-red-600" />;
      case 'needs_review': return <AlertTriangle className="w-4 h-4 text-yellow-600" />;
      default: return <Clock className="w-4 h-4 text-gray-600" />;
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading pending analyses...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-foreground">AI Analysis Review</h1>
          <p className="text-muted-foreground mt-1">Review and approve AI diagnostic analyses</p>
        </div>
        <Badge variant="secondary" className="flex items-center space-x-1">
          <Brain className="w-3 h-3" />
          <span>{pendingAnalyses.length} Pending</span>
        </Badge>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Pending Analyses List */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Clock className="w-5 h-5" />
              <span>Pending Reviews</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {pendingAnalyses.length === 0 ? (
                <div className="text-center py-8 text-gray-500">
                  <Brain className="w-12 h-12 mx-auto mb-4 text-gray-300" />
                  <p>No pending analyses to review</p>
                </div>
              ) : (
                pendingAnalyses.map((analysis) => (
                  <div
                    key={analysis.id}
                    className={`p-4 border rounded-lg cursor-pointer transition-colors ${
                      selectedAnalysis?.id === analysis.id
                        ? 'border-blue-500 bg-blue-50'
                        : 'border-gray-200 hover:border-gray-300'
                    }`}
                    onClick={() => handleAnalysisSelect(analysis.id)}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center space-x-2">
                        <User className="w-4 h-4 text-gray-500" />
                        <span className="font-medium">{analysis.patientName}</span>
                      </div>
                      <Badge className={getRiskLevelColor(analysis.riskLevel)}>
                        {analysis.riskLevel.toUpperCase()}
                      </Badge>
                    </div>

                    <div className="space-y-1 text-sm text-gray-600">
                      <div className="flex items-center space-x-2">
                        <FileText className="w-3 h-3" />
                        <span className="capitalize">{analysis.imageType.replace('_', ' ')}</span>
                      </div>
                      <div className="flex items-center space-x-2">
                        <Brain className="w-3 h-3" />
                        <span>AI Prediction: {analysis.prediction}</span>
                        <span className="text-xs">({Math.round(analysis.confidence * 100)}%)</span>
                      </div>
                      <div className="flex items-center space-x-2">
                        <Calendar className="w-3 h-3" />
                        <span>{new Date(analysis.timestamp).toLocaleString()}</span>
                      </div>
                    </div>
                  </div>
                ))
              )}
            </div>
          </CardContent>
        </Card>

        {/* Review Form */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <CheckCircle className="w-5 h-5" />
              <span>Review Analysis</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            {selectedAnalysis ? (
              <div className="space-y-4">
                {/* Analysis Summary */}
                <div className="p-4 bg-gray-50 rounded-lg">
                  <h4 className="font-medium mb-2">AI Analysis Summary</h4>
                  <div className="space-y-2 text-sm">
                    <div><strong>Patient:</strong> {selectedAnalysis.patientName}</div>
                    <div><strong>Image Type:</strong> {selectedAnalysis.imageType.replace('_', ' ')}</div>
                    <div><strong>AI Prediction:</strong> {selectedAnalysis.prediction}</div>
                    <div><strong>Confidence:</strong> {Math.round(selectedAnalysis.confidence * 100)}%</div>
                    <div><strong>Risk Level:</strong>
                      <Badge className={`ml-2 ${getRiskLevelColor(selectedAnalysis.riskLevel)}`}>
                        {selectedAnalysis.riskLevel.toUpperCase()}
                      </Badge>
                    </div>
                  </div>
                </div>

                {/* Review Status */}
                <div className="space-y-2">
                  <Label>Review Decision</Label>
                  <div className="flex space-x-2">
                    <Button
                      variant={reviewStatus === 'approved' ? 'default' : 'outline'}
                      size="sm"
                      onClick={() => setReviewStatus('approved')}
                      className="flex items-center space-x-1"
                    >
                      <CheckCircle className="w-4 h-4" />
                      <span>Approve</span>
                    </Button>
                    <Button
                      variant={reviewStatus === 'rejected' ? 'destructive' : 'outline'}
                      size="sm"
                      onClick={() => setReviewStatus('rejected')}
                      className="flex items-center space-x-1"
                    >
                      <XCircle className="w-4 h-4" />
                      <span>Reject</span>
                    </Button>
                    <Button
                      variant={reviewStatus === 'needs_review' ? 'secondary' : 'outline'}
                      size="sm"
                      onClick={() => setReviewStatus('needs_review')}
                      className="flex items-center space-x-1"
                    >
                      <AlertTriangle className="w-4 h-4" />
                      <span>Needs Review</span>
                    </Button>
                  </div>
                </div>

                {/* Clinical Notes */}
                <div className="space-y-2">
                  <Label htmlFor="clinicalNotes">Clinical Notes</Label>
                  <Textarea
                    id="clinicalNotes"
                    placeholder="Enter your clinical assessment and notes..."
                    value={clinicalNotes}
                    onChange={(e) => setClinicalNotes(e.target.value)}
                    rows={3}
                  />
                </div>

                {/* Final Diagnosis */}
                <div className="space-y-2">
                  <Label htmlFor="finalDiagnosis">Final Diagnosis</Label>
                  <Input
                    id="finalDiagnosis"
                    placeholder="Enter final diagnosis..."
                    value={finalDiagnosis}
                    onChange={(e) => setFinalDiagnosis(e.target.value)}
                  />
                </div>

                {/* Recommendations */}
                <div className="space-y-2">
                  <Label htmlFor="recommendations">Recommendations (comma-separated)</Label>
                  <Textarea
                    id="recommendations"
                    placeholder="Enter recommendations for the patient..."
                    value={recommendations}
                    onChange={(e) => setRecommendations(e.target.value)}
                    rows={2}
                  />
                </div>

                {/* Submit Button */}
                <Button
                  onClick={handleSubmitReview}
                  disabled={isSubmitting}
                  className="w-full"
                >
                  {isSubmitting ? 'Submitting...' : 'Submit Review'}
                </Button>
              </div>
            ) : (
              <div className="text-center py-8 text-gray-500">
                <FileText className="w-12 h-12 mx-auto mb-4 text-gray-300" />
                <p>Select an analysis to review</p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
