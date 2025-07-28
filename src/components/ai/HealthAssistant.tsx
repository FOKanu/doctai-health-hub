import React, { useState } from 'react';
import { ApiResponse } from '@/types/common';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Loader2, Brain, MessageCircle, Lightbulb, BookOpen } from 'lucide-react';
import { apiServiceManager } from '@/services/api/apiServiceManager';

interface HealthAssistantProps {
  className?: string;
}

export const HealthAssistant: React.FC<HealthAssistantProps> = ({ className }) => {
  const [mode, setMode] = useState<'symptoms' | 'terms' | 'tips'>('symptoms');
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ApiResponse<unknown> | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async () => {
    if (!input.trim()) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      let response;

      switch (mode) {
        case 'symptoms': {
          const symptoms = input.split(',').map(s => s.trim());
          response = await apiServiceManager.generateHealthInsights(symptoms);
          break;
        }
        case 'terms': {
          response = await apiServiceManager.explainMedicalTerm(input);
          break;
        }
        case 'tips': {
          response = await apiServiceManager.generateHealthTips(input);
          break;
        }
      }

      if (response.success) {
        setResult(response.data);
      } else {
        setError(response.error || 'Failed to get response');
      }
    } catch (err: Error | unknown) {
      setError(err.message || 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const renderResult = () => {
    if (!result) return null;

    switch (mode) {
      case 'symptoms':
        return (
          <Card className="mt-4">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Brain className="h-5 w-5" />
                Health Analysis
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <h4 className="font-semibold mb-2">Possible Conditions:</h4>
                <div className="flex flex-wrap gap-2">
                  {result.possibleConditions?.map((condition: string, index: number) => (
                    <Badge key={index} variant="secondary">{condition}</Badge>
                  ))}
                </div>
              </div>

              <div className="flex gap-4">
                <div>
                  <span className="text-sm text-muted-foreground">Severity:</span>
                  <Badge
                    variant={result.severity === 'critical' ? 'destructive' :
                           result.severity === 'high' ? 'default' : 'secondary'}
                    className="ml-2"
                  >
                    {result.severity}
                  </Badge>
                </div>
                <div>
                  <span className="text-sm text-muted-foreground">Urgency:</span>
                  <Badge
                    variant={result.urgency === 'emergency' ? 'destructive' :
                           result.urgency === 'urgent' ? 'default' : 'secondary'}
                    className="ml-2"
                  >
                    {result.urgency}
                  </Badge>
                </div>
              </div>

              <div>
                <h4 className="font-semibold mb-2">Recommendations:</h4>
                <ul className="list-disc list-inside space-y-1">
                  {result.recommendations?.map((rec: string, index: number) => (
                    <li key={index} className="text-sm">{rec}</li>
                  ))}
                </ul>
              </div>

              <div>
                <h4 className="font-semibold mb-2">Explanation:</h4>
                <p className="text-sm text-muted-foreground">{result.explanation}</p>
              </div>

              <Alert>
                <AlertDescription className="text-xs">
                  {result.disclaimer}
                </AlertDescription>
              </Alert>
            </CardContent>
          </Card>
        );

      case 'terms':
        return (
          <Card className="mt-4">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <BookOpen className="h-5 w-5" />
                Medical Term Explanation
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <h4 className="font-semibold mb-2">Detailed Explanation:</h4>
                <p className="text-sm text-muted-foreground">{result.explanation}</p>
              </div>

              <div>
                <h4 className="font-semibold mb-2">Simplified for Patients:</h4>
                <p className="text-sm text-muted-foreground">{result.simplified}</p>
              </div>
            </CardContent>
          </Card>
        );

      case 'tips':
        return (
          <Card className="mt-4">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Lightbulb className="h-5 w-5" />
                Health Tips
              </CardTitle>
            </CardHeader>
            <CardContent>
              <ul className="space-y-2">
                {result.map((tip: string, index: number) => (
                  <li key={index} className="flex items-start gap-2">
                    <span className="text-primary font-bold">•</span>
                    <span className="text-sm">{tip}</span>
                  </li>
                ))}
              </ul>
            </CardContent>
          </Card>
        );

      default:
        return null;
    }
  };

  return (
    <div className={className}>
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="h-6 w-6" />
            AI Health Assistant
          </CardTitle>
          <CardDescription>
            Get health insights, understand medical terms, and receive personalized health tips
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Mode Selection */}
          <div className="flex gap-2">
            <Button
              variant={mode === 'symptoms' ? 'default' : 'outline'}
              size="sm"
              onClick={() => setMode('symptoms')}
            >
              <MessageCircle className="h-4 w-4 mr-2" />
              Symptom Analysis
            </Button>
            <Button
              variant={mode === 'terms' ? 'default' : 'outline'}
              size="sm"
              onClick={() => setMode('terms')}
            >
              <BookOpen className="h-4 w-4 mr-2" />
              Medical Terms
            </Button>
            <Button
              variant={mode === 'tips' ? 'default' : 'outline'}
              size="sm"
              onClick={() => setMode('tips')}
            >
              <Lightbulb className="h-4 w-4 mr-2" />
              Health Tips
            </Button>
          </div>

          {/* Input */}
          <div className="space-y-2">
            <label className="text-sm font-medium">
              {mode === 'symptoms' && 'Enter symptoms (comma-separated):'}
              {mode === 'terms' && 'Enter medical term:'}
              {mode === 'tips' && 'Enter health category (e.g., nutrition, exercise, sleep):'}
            </label>
            {mode === 'symptoms' ? (
              <Textarea
                placeholder="e.g., fever, cough, fatigue"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                rows={3}
              />
            ) : (
              <Input
                placeholder={
                  mode === 'terms'
                    ? 'e.g., hypertension, diabetes, MRI'
                    : 'e.g., nutrition, exercise, sleep, stress'
                }
                value={input}
                onChange={(e) => setInput(e.target.value)}
              />
            )}
          </div>

          {/* Submit Button */}
          <Button
            onClick={handleSubmit}
            disabled={loading || !input.trim()}
            className="w-full"
          >
            {loading ? (
              <>
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                Analyzing...
              </>
            ) : (
              <>
                <Brain className="h-4 w-4 mr-2" />
                Get Insights
              </>
            )}
          </Button>

          {/* Error */}
          {error && (
            <Alert variant="destructive">
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          {/* Result */}
          {renderResult()}
        </CardContent>
      </Card>
    </div>
  );
};
