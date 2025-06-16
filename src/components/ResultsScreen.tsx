import React from 'react';
import { useNavigate } from 'react-router-dom';
import { ScanResult } from '../types/analysis';
import { Card } from './ui/card';
import { Button } from './ui/button';
import { Progress } from './ui/progress';
import { Share2, Calendar, Activity, ArrowLeft } from 'lucide-react';

interface ResultsScreenProps {
  result: ScanResult;
  onShare: () => Promise<void>;
  onSchedule: () => void;
  onMonitor: () => void;
}

export default function ResultsScreen({
  result,
  onShare,
  onSchedule,
  onMonitor
}: ResultsScreenProps) {
  const navigate = useNavigate();

  const getRiskColor = (riskLevel: string) => {
    switch (riskLevel) {
      case 'high':
        return 'text-red-600 bg-red-50';
      case 'medium':
        return 'text-yellow-600 bg-yellow-50';
      case 'low':
        return 'text-green-600 bg-green-50';
      default:
        return 'text-gray-600 bg-gray-50';
    }
  };

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="flex items-center justify-between mb-8">
        <Button
          variant="ghost"
          onClick={() => navigate(-1)}
          className="flex items-center gap-2"
        >
          <ArrowLeft className="w-4 h-4" />
          Back
        </Button>
        <h1 className="text-2xl font-bold">Scan Results</h1>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Scan Image */}
        <Card className="p-6">
          <h2 className="text-xl font-semibold mb-4">Scan Image</h2>
          <div className="aspect-square bg-gray-100 rounded-lg overflow-hidden">
            <img
              src={result.image}
              alt="Scan result"
              className="w-full h-full object-cover"
            />
          </div>
        </Card>

        {/* Risk Assessment */}
        <Card className="p-6">
          <h2 className="text-xl font-semibold mb-4">Risk Assessment</h2>
          <div className="space-y-6">
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-gray-500">Risk Level</span>
                <span className={`px-3 py-1 rounded-full text-sm font-medium ${getRiskColor(result.riskLevel)}`}>
                  {result.riskLevel}
                </span>
              </div>
              <Progress value={result.confidence} className="h-2" />
              <p className="text-sm text-gray-500 mt-1">
                Confidence: {result.confidence}%
              </p>
            </div>

            <div>
              <h3 className="font-medium mb-2">Findings</h3>
              <ul className="list-disc list-inside space-y-2">
                {result.findings.map((finding, index) => (
                  <li key={index} className="text-gray-600">{finding}</li>
                ))}
              </ul>
            </div>

            <div>
              <h3 className="font-medium mb-2">Recommendations</h3>
              <ul className="list-disc list-inside space-y-2">
                {result.recommendations.map((recommendation, index) => (
                  <li key={index} className="text-gray-600">{recommendation}</li>
                ))}
              </ul>
            </div>
          </div>
        </Card>

        {/* Action Buttons */}
        <div className="md:col-span-2 flex flex-wrap gap-4">
          <Button
            onClick={onShare}
            className="flex-1 min-w-[200px]"
          >
            <Share2 className="w-4 h-4 mr-2" />
            Share with Doctor
          </Button>
          <Button
            onClick={onSchedule}
            variant="outline"
            className="flex-1 min-w-[200px]"
          >
            <Calendar className="w-4 h-4 mr-2" />
            Schedule Follow-up
          </Button>
          <Button
            onClick={onMonitor}
            variant="outline"
            className="flex-1 min-w-[200px]"
          >
            <Activity className="w-4 h-4 mr-2" />
            Monitor Progress
          </Button>
        </div>
      </div>
    </div>
  );
}
