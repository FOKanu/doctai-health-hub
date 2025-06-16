import React from 'react';
import { useNavigate } from 'react-router-dom';
import { AnalyticsData } from '../types/analysis';
import { Card } from './ui/card';
import { Button } from './ui/button';
import { ArrowLeft, TrendingUp, Calendar, Clock } from 'lucide-react';

interface AnalyticsScreenProps {
  data: AnalyticsData;
  onViewDetails: (scanId: string) => void;
}

export default function AnalyticsScreen({ data, onViewDetails }: AnalyticsScreenProps) {
  const navigate = useNavigate();

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString();
  };

  const getImprovementColor = (type: string) => {
    switch (type) {
      case 'improvement':
        return 'text-green-600';
      case 'deterioration':
        return 'text-red-600';
      default:
        return 'text-gray-600';
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
        <h1 className="text-2xl font-bold">Health Analytics</h1>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Trends Section */}
        <Card className="p-6">
          <h2 className="text-xl font-semibold mb-4">Risk Level Trends</h2>
          <div className="h-64">
            {/* Add your chart component here */}
            <div className="flex items-center justify-center h-full text-gray-500">
              Chart visualization will be implemented here
            </div>
          </div>
        </Card>

        {/* Progress Section */}
        <Card className="p-6">
          <h2 className="text-xl font-semibold mb-4">Progress Overview</h2>
          <div className="space-y-4">
            <div className="flex items-center gap-2">
              <Calendar className="w-5 h-5 text-gray-500" />
              <span>Started: {formatDate(data.insights.progress.startDate)}</span>
            </div>
            <div className="flex items-center gap-2">
              <Clock className="w-5 h-5 text-gray-500" />
              <span>{data.insights.progress.currentStatus}</span>
            </div>
            <div className="flex items-center gap-2">
              <TrendingUp className="w-5 h-5 text-gray-500" />
              <span className={getImprovementColor(data.comparisons.changes.type)}>
                {data.comparisons.changes.percentage.toFixed(1)}% {data.comparisons.changes.type}
              </span>
            </div>
          </div>
        </Card>

        {/* Previous Scans */}
        <Card className="p-6 md:col-span-2">
          <h2 className="text-xl font-semibold mb-4">Previous Scans</h2>
          <div className="space-y-4">
            {data.comparisons.previousScans.map((scan) => (
              <div
                key={scan.id}
                className="flex items-center justify-between p-4 bg-gray-50 rounded-lg"
              >
                <div>
                  <p className="font-medium">{scan.type}</p>
                  <p className="text-sm text-gray-500">
                    {formatDate(scan.timestamp)}
                  </p>
                </div>
                <Button
                  variant="outline"
                  onClick={() => onViewDetails(scan.id)}
                >
                  View Details
                </Button>
              </div>
            ))}
          </div>
        </Card>

        {/* Insights */}
        <Card className="p-6 md:col-span-2">
          <h2 className="text-xl font-semibold mb-4">Insights</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h3 className="font-medium mb-2">Patterns</h3>
              <ul className="list-disc list-inside space-y-2">
                {data.insights.patterns.map((pattern, index) => (
                  <li key={index} className="text-gray-600">{pattern}</li>
                ))}
              </ul>
            </div>
            <div>
              <h3 className="font-medium mb-2">Recommendations</h3>
              <ul className="list-disc list-inside space-y-2">
                {data.insights.recommendations.map((recommendation, index) => (
                  <li key={index} className="text-gray-600">{recommendation}</li>
                ))}
              </ul>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
}
