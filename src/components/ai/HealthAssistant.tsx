import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { Bot, Send, AlertTriangle, Clock, Activity, User } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';

interface SymptomAssessment {
  assessment: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  recommendations: string[];
  urgency: 'routine' | 'soon' | 'urgent' | 'emergency';
  possibleConditions: Array<{
    condition: string;
    probability: number;
    description: string;
  }>;
}

interface HealthInsight {
  insights: string[];
  riskFactors: string[];
  preventiveMeasures: string[];
  followUpRecommendations: string[];
  evidenceLevel: 'low' | 'medium' | 'high';
}

export function HealthAssistant() {
  const [symptoms, setSymptoms] = useState('');
  const [patientInfo, setPatientInfo] = useState({
    age: '',
    gender: '',
    medicalHistory: '',
    currentMedications: ''
  });
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<SymptomAssessment | null>(null);
  const [insights, setInsights] = useState<HealthInsight | null>(null);
  const { toast } = useToast();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!symptoms.trim()) {
      toast({
        title: "Please describe your symptoms",
        description: "Enter at least one symptom to get an assessment.",
        variant: "destructive"
      });
      return;
    }

    setIsLoading(true);
    try {
      // Simulate API call for symptom assessment
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      const mockResult: SymptomAssessment = {
        assessment: "Based on the symptoms described, this appears to be a common respiratory condition that may require professional evaluation.",
        severity: Math.random() > 0.7 ? 'high' : Math.random() > 0.4 ? 'medium' : 'low',
        recommendations: [
          "Stay hydrated and get adequate rest",
          "Monitor symptoms for any worsening",
          "Consider over-the-counter pain relief if needed",
          "Schedule a consultation with your healthcare provider"
        ],
        urgency: Math.random() > 0.8 ? 'urgent' : Math.random() > 0.5 ? 'soon' : 'routine',
        possibleConditions: [
          {
            condition: "Upper Respiratory Infection",
            probability: 75,
            description: "Common viral infection affecting the upper respiratory tract"
          },
          {
            condition: "Allergic Reaction",
            probability: 20,
            description: "Environmental or food-related allergic response"
          },
          {
            condition: "Other Conditions",
            probability: 5,
            description: "Less common conditions that may require specialist evaluation"
          }
        ]
      };

      const mockInsights: HealthInsight = {
        insights: [
          "Symptoms align with seasonal patterns commonly seen this time of year",
          "Patient age group typically experiences good recovery rates",
          "No immediate red flags identified in the symptom profile"
        ],
        riskFactors: [
          "Close contact with others who are sick",
          "Recent travel or exposure to new environments",
          "Underlying health conditions if any"
        ],
        preventiveMeasures: [
          "Maintain good hand hygiene",
          "Ensure adequate sleep and nutrition",
          "Consider immune system support supplements",
          "Avoid known triggers or allergens"
        ],
        followUpRecommendations: [
          "Monitor temperature regularly",
          "Keep a symptom diary",
          "Return if symptoms worsen or persist beyond 7-10 days",
          "Seek immediate care if breathing difficulties develop"
        ],
        evidenceLevel: 'medium'
      };

      setResult(mockResult);
      setInsights(mockInsights);
      
      toast({
        title: "Assessment complete",
        description: "Your symptom assessment has been generated. Please review the recommendations.",
      });
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to generate assessment. Please try again.",
        variant: "destructive"
      });
    } finally {
      setIsLoading(false);
    }
  };

  const getSeverityVariant = (severity: 'low' | 'medium' | 'high' | 'critical') => {
    switch (severity) {
      case 'critical': return 'destructive';
      case 'high': return 'default';
      default: return 'secondary';
    }
  };

  const getUrgencyVariant = (urgency: 'routine' | 'soon' | 'urgent' | 'emergency') => {
    switch (urgency) {
      case 'emergency': return 'destructive';
      case 'urgent': return 'default';
      default: return 'secondary';
    }
  };

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <div className="text-center">
        <div className="flex justify-center mb-4">
          <div className="p-3 bg-blue-100 rounded-full">
            <Bot className="w-8 h-8 text-blue-600" />
          </div>
        </div>
        <h1 className="text-3xl font-bold text-gray-900 mb-2">AI Health Assistant</h1>
        <p className="text-gray-600 max-w-2xl mx-auto">
          Describe your symptoms and get AI-powered health insights and recommendations. 
          This is not a substitute for professional medical advice.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-1">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <User className="w-5 h-5" />
                Symptom Assessment
              </CardTitle>
              <CardDescription>
                Provide details about your symptoms and health information
              </CardDescription>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleSubmit} className="space-y-4">
                <div>
                  <label className="text-sm font-medium text-gray-700 mb-2 block">
                    Describe your symptoms *
                  </label>
                  <Textarea
                    value={symptoms}
                    onChange={(e) => setSymptoms(e.target.value)}
                    placeholder="Please describe your symptoms in detail..."
                    className="min-h-[100px]"
                    required
                  />
                </div>
                
                <Separator />
                
                <div className="space-y-3">
                  <h4 className="text-sm font-medium text-gray-700">Optional Information</h4>
                  
                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <label className="text-xs text-gray-600 mb-1 block">Age</label>
                      <Input
                        type="number"
                        value={patientInfo.age}
                        onChange={(e) => setPatientInfo(prev => ({ ...prev, age: e.target.value }))}
                        placeholder="Age"
                        min="0"
                        max="120"
                      />
                    </div>
                    <div>
                      <label className="text-xs text-gray-600 mb-1 block">Gender</label>
                      <Input
                        value={patientInfo.gender}
                        onChange={(e) => setPatientInfo(prev => ({ ...prev, gender: e.target.value }))}
                        placeholder="Gender"
                      />
                    </div>
                  </div>
                  
                  <div>
                    <label className="text-xs text-gray-600 mb-1 block">Medical History</label>
                    <Textarea
                      value={patientInfo.medicalHistory}
                      onChange={(e) => setPatientInfo(prev => ({ ...prev, medicalHistory: e.target.value }))}
                      placeholder="Any relevant medical history..."
                      rows={2}
                    />
                  </div>
                  
                  <div>
                    <label className="text-xs text-gray-600 mb-1 block">Current Medications</label>
                    <Textarea
                      value={patientInfo.currentMedications}
                      onChange={(e) => setPatientInfo(prev => ({ ...prev, currentMedications: e.target.value }))}
                      placeholder="List any current medications..."
                      rows={2}
                    />
                  </div>
                </div>
                
                <Button type="submit" disabled={isLoading} className="w-full">
                  {isLoading ? (
                    <>
                      <Activity className="w-4 h-4 mr-2 animate-spin" />
                      Analyzing...
                    </>
                  ) : (
                    <>
                      <Send className="w-4 h-4 mr-2" />
                      Get Assessment
                    </>
                  )}
                </Button>
              </form>
            </CardContent>
          </Card>
        </div>

        <div className="lg:col-span-2">
          {result && (
            <div className="space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <AlertTriangle className="w-5 h-5 text-orange-500" />
                    Assessment Results
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="p-4 bg-blue-50 rounded-lg">
                    <p className="text-sm text-blue-800">{result.assessment}</p>
                  </div>
                  
                  <div className="flex flex-wrap gap-4">
                    <div>
                      <span className="text-sm text-muted-foreground">Severity:</span>
                      <Badge
                        variant={getSeverityVariant(result.severity)}
                        className="ml-2"
                      >
                        {result.severity}
                      </Badge>
                    </div>
                    <div>
                      <span className="text-sm text-muted-foreground">Urgency:</span>
                      <Badge
                        variant={getUrgencyVariant(result.urgency)}
                        className="ml-2"
                      >
                        {result.urgency}
                      </Badge>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg">Recommendations</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <ul className="space-y-2">
                      {result.recommendations.map((rec, index) => (
                        <li key={index} className="flex items-start gap-2">
                          <div className="w-1.5 h-1.5 bg-blue-500 rounded-full mt-2 flex-shrink-0" />
                          <span className="text-sm text-gray-700">{rec}</span>
                        </li>
                      ))}
                    </ul>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg">Possible Conditions</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      {result.possibleConditions.map((condition, index) => (
                        <div key={index} className="border-l-4 border-blue-200 pl-3">
                          <div className="flex justify-between items-start mb-1">
                            <h5 className="font-medium text-sm">{condition.condition}</h5>
                            <span className="text-xs bg-gray-100 px-2 py-1 rounded">
                              {condition.probability}%
                            </span>
                          </div>
                          <p className="text-xs text-gray-600">{condition.description}</p>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              </div>

              {insights && (
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Bot className="w-5 h-5 text-purple-500" />
                      Health Insights
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div>
                      <h4 className="font-medium mb-2 text-sm">Clinical Insights</h4>
                      <ul className="space-y-1">
                        {insights.insights.map((insight, index) => (
                          <li key={index} className="text-sm text-gray-700 flex items-start gap-2">
                            <div className="w-1 h-1 bg-purple-400 rounded-full mt-2 flex-shrink-0" />
                            {insight}
                          </li>
                        ))}
                      </ul>
                    </div>

                    <Separator />

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div>
                        <h4 className="font-medium mb-2 text-sm text-red-700">Risk Factors</h4>
                        <ul className="space-y-1">
                          {insights.riskFactors.map((factor, index) => (
                            <li key={index} className="text-sm text-gray-700 flex items-start gap-2">
                              <div className="w-1 h-1 bg-red-400 rounded-full mt-2 flex-shrink-0" />
                              {factor}
                            </li>
                          ))}
                        </ul>
                      </div>

                      <div>
                        <h4 className="font-medium mb-2 text-sm text-green-700">Preventive Measures</h4>
                        <ul className="space-y-1">
                          {insights.preventiveMeasures.map((measure, index) => (
                            <li key={index} className="text-sm text-gray-700 flex items-start gap-2">
                              <div className="w-1 h-1 bg-green-400 rounded-full mt-2 flex-shrink-0" />
                              {measure}
                            </li>
                          ))}
                        </ul>
                      </div>
                    </div>

                    <Separator />

                    <div>
                      <h4 className="font-medium mb-2 text-sm flex items-center gap-2">
                        <Clock className="w-4 h-4 text-blue-500" />
                        Follow-up Recommendations
                      </h4>
                      <ul className="space-y-1">
                        {insights.followUpRecommendations.map((rec, index) => (
                          <li key={index} className="text-sm text-gray-700 flex items-start gap-2">
                            <div className="w-1 h-1 bg-blue-400 rounded-full mt-2 flex-shrink-0" />
                            {rec}
                          </li>
                        ))}
                      </ul>
                    </div>

                    <div className="text-xs text-gray-500 mt-4 p-3 bg-gray-50 rounded">
                      <strong>Evidence Level:</strong> {insights.evidenceLevel.toUpperCase()} • 
                      <strong> Disclaimer:</strong> This assessment is for informational purposes only and should not replace professional medical advice, diagnosis, or treatment.
                    </div>
                  </CardContent>
                </Card>
              )}
            </div>
          )}

          {!result && (
            <Card>
              <CardContent className="text-center py-12">
                <Bot className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-gray-900 mb-2">Ready to Help</h3>
                <p className="text-gray-600">
                  Fill out the form on the left to get your AI-powered health assessment
                </p>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}