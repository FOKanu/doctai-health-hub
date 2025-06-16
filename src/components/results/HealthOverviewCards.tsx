
import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { 
  Heart, 
  Activity, 
  Apple, 
  Scan, 
  TrendingUp, 
  TrendingDown, 
  Minus,
  CheckCircle2,
  AlertCircle
} from 'lucide-react';

interface HealthSummary {
  skinHealth: {
    status: string;
    lastScan: string;
    confidence: number;
    trend: string;
  };
  fitnessStreak: {
    days: number;
    goal: number;
    percentage: number;
  };
  dietCompliance: {
    score: number;
    streak: number;
    trend: string;
  };
  overallHealth: {
    score: number;
    trend: string;
    change: string;
  };
}

interface HealthOverviewCardsProps {
  healthSummary: HealthSummary;
}

export const HealthOverviewCards: React.FC<HealthOverviewCardsProps> = ({ healthSummary }) => {
  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case 'improving':
        return <TrendingUp className="w-4 h-4 text-green-600" />;
      case 'declining':
        return <TrendingDown className="w-4 h-4 text-red-600" />;
      default:
        return <Minus className="w-4 h-4 text-gray-600" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status.toLowerCase()) {
      case 'clear':
      case 'excellent':
        return 'text-green-600 bg-green-50 border-green-200';
      case 'warning':
      case 'moderate':
        return 'text-yellow-600 bg-yellow-50 border-yellow-200';
      case 'alert':
      case 'poor':
        return 'text-red-600 bg-red-50 border-red-200';
      default:
        return 'text-gray-600 bg-gray-50 border-gray-200';
    }
  };

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
      {/* Skin Health */}
      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <CardTitle className="text-sm font-medium text-gray-600">Skin Health</CardTitle>
            <Scan className="w-4 h-4 text-gray-400" />
          </div>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <Badge className={getStatusColor(healthSummary.skinHealth.status)}>
                <CheckCircle2 className="w-3 h-3 mr-1" />
                {healthSummary.skinHealth.status}
              </Badge>
              {getTrendIcon(healthSummary.skinHealth.trend)}
            </div>
            <div>
              <p className="text-2xl font-bold text-gray-900">{healthSummary.skinHealth.confidence}%</p>
              <p className="text-xs text-gray-500">Confidence • {healthSummary.skinHealth.lastScan}</p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Fitness Streak */}
      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <CardTitle className="text-sm font-medium text-gray-600">Workout Streak</CardTitle>
            <Activity className="w-4 h-4 text-gray-400" />
          </div>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <span className="text-2xl font-bold text-gray-900">{healthSummary.fitnessStreak.days}</span>
              <span className="text-sm text-gray-500">of {healthSummary.fitnessStreak.goal} days</span>
            </div>
            <Progress value={healthSummary.fitnessStreak.percentage} className="h-2" />
            <p className="text-xs text-gray-500">{healthSummary.fitnessStreak.percentage}% to weekly goal</p>
          </div>
        </CardContent>
      </Card>

      {/* Diet Compliance */}
      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <CardTitle className="text-sm font-medium text-gray-600">Diet Compliance</CardTitle>
            <Apple className="w-4 h-4 text-gray-400" />
          </div>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <span className="text-2xl font-bold text-gray-900">{healthSummary.dietCompliance.score}%</span>
              {getTrendIcon(healthSummary.dietCompliance.trend)}
            </div>
            <div>
              <p className="text-sm text-gray-600">{healthSummary.dietCompliance.streak} day streak</p>
              <p className="text-xs text-gray-500">Weekly average</p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Overall Health Score */}
      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <CardTitle className="text-sm font-medium text-gray-600">Health Score</CardTitle>
            <Heart className="w-4 h-4 text-gray-400" />
          </div>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <span className="text-2xl font-bold text-gray-900">{healthSummary.overallHealth.score}</span>
              <Badge variant="outline" className="text-green-600 border-green-200">
                {healthSummary.overallHealth.change}
              </Badge>
            </div>
            <div className="flex items-center gap-1">
              {getTrendIcon(healthSummary.overallHealth.trend)}
              <p className="text-xs text-gray-500 capitalize">{healthSummary.overallHealth.trend}</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
