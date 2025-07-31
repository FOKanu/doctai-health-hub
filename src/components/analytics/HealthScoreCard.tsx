import React, { useState, useEffect, useCallback } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Alert, AlertDescription } from '@/components/ui/alert';
import {
  TrendingUp,
  TrendingDown,
  Minus,
  Heart,
  Brain,
  Activity,
  Moon,
  Zap,
  Target,
  AlertTriangle,
  CheckCircle,
  Info,
  Apple
} from 'lucide-react';
import { healthScoringService, type HealthScore, type HealthScoreBreakdown, type PersonalizedRecommendations } from '@/services/healthScoringService';

interface HealthScoreCardProps {
  userId: string;
  className?: string;
}

export function HealthScoreCard({ userId, className }: HealthScoreCardProps) {
  const [healthScore, setHealthScore] = useState<HealthScore | null>(null);
  const [breakdown, setBreakdown] = useState<HealthScoreBreakdown[]>([]);
  const [recommendations, setRecommendations] = useState<PersonalizedRecommendations[]>([]);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('overview');

  const loadHealthData = useCallback(async () => {
    try {
      setLoading(true);
      const [score, breakdownData, recommendationsData] = await Promise.all([
        healthScoringService.calculateHealthScore(userId),
        healthScoringService.getHealthScoreBreakdown(userId),
        healthScoringService.getPersonalizedRecommendations(userId)
      ]);

      setHealthScore(score);
      setBreakdown(breakdownData);
      setRecommendations(recommendationsData);
    } catch (error) {
      console.error('Error loading health data:', error);
    } finally {
      setLoading(false);
    }
  }, [userId]);

  useEffect(() => {
    loadHealthData();
  }, [loadHealthData]);

  const getScoreColor = (score: number) => {
    if (score >= 80) return 'text-green-600';
    if (score >= 60) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getScoreBgColor = (score: number) => {
    if (score >= 80) return 'bg-green-100';
    if (score >= 60) return 'bg-yellow-100';
    return 'bg-red-100';
  };

  const getRiskLevelColor = (riskLevel: string) => {
    switch (riskLevel) {
      case 'low': return 'bg-green-100 text-green-800';
      case 'medium': return 'bg-yellow-100 text-yellow-800';
      case 'high': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case 'improving':
        return <TrendingUp className="h-4 w-4 text-green-600" />;
      case 'declining':
        return <TrendingDown className="h-4 w-4 text-red-600" />;
      default:
        return <Minus className="h-4 w-4 text-gray-600" />;
    }
  };

  const getDomainIcon = (domain: string) => {
    switch (domain.toLowerCase()) {
      case 'cardiovascular':
        return <Heart className="h-4 w-4" />;
      case 'mental health':
        return <Brain className="h-4 w-4" />;
      case 'fitness':
        return <Activity className="h-4 w-4" />;
      case 'sleep':
        return <Moon className="h-4 w-4" />;
      case 'metabolic':
        return <Zap className="h-4 w-4" />;
      case 'diet':
        return <Apple className="h-4 w-4" />;
      default:
        return <Target className="h-4 w-4" />;
    }
  };

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'high': return 'border-red-200 bg-red-50';
      case 'medium': return 'border-yellow-200 bg-yellow-50';
      case 'low': return 'border-green-200 bg-green-50';
      default: return 'border-gray-200 bg-gray-50';
    }
  };

  if (loading) {
    return (
      <Card className={className}>
        <CardHeader>
          <CardTitle>Health Score</CardTitle>
          <CardDescription>Calculating your personalized health score...</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="animate-pulse space-y-4">
            <div className="h-8 bg-gray-200 rounded"></div>
            <div className="h-4 bg-gray-200 rounded"></div>
            <div className="h-4 bg-gray-200 rounded w-3/4"></div>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!healthScore) {
    return (
      <Card className={className}>
        <CardContent className="pt-6">
          <Alert>
            <AlertTriangle className="h-4 w-4" />
            <AlertDescription>
              Unable to calculate health score. Please ensure you have sufficient health data.
            </AlertDescription>
          </Alert>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className={className}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle>Health Score</CardTitle>
            <CardDescription>
              Your personalized health assessment based on multiple metrics
            </CardDescription>
          </div>
          <div className="flex items-center gap-2">
            {getTrendIcon(healthScore.trend)}
            <Badge className={getRiskLevelColor(healthScore.riskLevel)}>
              {healthScore.riskLevel.toUpperCase()} RISK
            </Badge>
          </div>
        </div>
      </CardHeader>

      <CardContent>
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="breakdown">Breakdown</TabsTrigger>
            <TabsTrigger value="recommendations">Recommendations</TabsTrigger>
          </TabsList>

          <TabsContent value="overview" className="space-y-4">
            {/* Overall Score */}
            <div className="text-center space-y-2">
              <div className={`text-4xl font-bold ${getScoreColor(healthScore.overall)}`}>
                {healthScore.overall}
              </div>
              <div className="text-sm text-gray-600">Overall Health Score</div>
              <Progress value={healthScore.overall} className="w-full" />
            </div>

            {/* Domain Scores Grid */}
            <div className="grid grid-cols-2 gap-4">
              <div className={`p-3 rounded-lg ${getScoreBgColor(healthScore.cardiovascular)}`}>
                <div className="flex items-center gap-2 mb-1">
                  <Heart className="h-4 w-4" />
                  <span className="text-sm font-medium">Cardiovascular</span>
                </div>
                <div className={`text-lg font-bold ${getScoreColor(healthScore.cardiovascular)}`}>
                  {healthScore.cardiovascular}
                </div>
              </div>

              <div className={`p-3 rounded-lg ${getScoreBgColor(healthScore.metabolic)}`}>
                <div className="flex items-center gap-2 mb-1">
                  <Zap className="h-4 w-4" />
                  <span className="text-sm font-medium">Metabolic</span>
                </div>
                <div className={`text-lg font-bold ${getScoreColor(healthScore.metabolic)}`}>
                  {healthScore.metabolic}
                </div>
              </div>

              <div className={`p-3 rounded-lg ${getScoreBgColor(healthScore.sleep)}`}>
                <div className="flex items-center gap-2 mb-1">
                  <Moon className="h-4 w-4" />
                  <span className="text-sm font-medium">Sleep</span>
                </div>
                <div className={`text-lg font-bold ${getScoreColor(healthScore.sleep)}`}>
                  {healthScore.sleep}
                </div>
              </div>

              <div className={`p-3 rounded-lg ${getScoreBgColor(healthScore.fitness)}`}>
                <div className="flex items-center gap-2 mb-1">
                  <Activity className="h-4 w-4" />
                  <span className="text-sm font-medium">Fitness</span>
                </div>
                <div className={`text-lg font-bold ${getScoreColor(healthScore.fitness)}`}>
                  {healthScore.fitness}
                </div>
              </div>

              <div className={`p-3 rounded-lg ${getScoreBgColor(healthScore.diet)}`}>
                <div className="flex items-center gap-2 mb-1">
                  <Apple className="h-4 w-4" />
                  <span className="text-sm font-medium">Diet</span>
                </div>
                <div className={`text-lg font-bold ${getScoreColor(healthScore.diet)}`}>
                  {healthScore.diet}
                </div>
              </div>
            </div>

            {/* Insights */}
            {healthScore.insights.length > 0 && (
              <div className="space-y-2">
                <h4 className="font-medium text-sm">Key Insights</h4>
                <div className="space-y-1">
                  {healthScore.insights.map((insight, index) => (
                    <div key={index} className="flex items-start gap-2 text-sm">
                      <Info className="h-4 w-4 text-blue-600 mt-0.5 flex-shrink-0" />
                      <span className="text-gray-700">{insight}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </TabsContent>

          <TabsContent value="breakdown" className="space-y-4">
            <div className="space-y-4">
              {breakdown.map((domain) => (
                <div key={domain.domain} className="space-y-2">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      {getDomainIcon(domain.domain)}
                      <span className="font-medium">{domain.domain}</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className={`text-lg font-bold ${getScoreColor(domain.score)}`}>
                        {domain.score}
                      </span>
                      <span className="text-xs text-gray-500">
                        Weight: {(domain.weight * 100).toFixed(0)}%
                      </span>
                    </div>
                  </div>

                  <Progress value={domain.score} className="w-full" />

                  <div className="space-y-1">
                    {domain.factors.map((factor, index) => (
                      <div key={index} className="flex items-center justify-between text-xs">
                        <span className="text-gray-600">{factor.metric}</span>
                        <div className="flex items-center gap-2">
                          <span className="text-gray-700">
                            {factor.value} / {factor.optimal}
                          </span>
                          {factor.impact === 'positive' && (
                            <CheckCircle className="h-3 w-3 text-green-600" />
                          )}
                          {factor.impact === 'negative' && (
                            <AlertTriangle className="h-3 w-3 text-red-600" />
                          )}
                          {factor.impact === 'neutral' && (
                            <Minus className="h-3 w-3 text-gray-400" />
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </TabsContent>

          <TabsContent value="recommendations" className="space-y-4">
            <div className="space-y-3">
              {recommendations.map((recommendation, index) => (
                <div
                  key={index}
                  className={`p-4 rounded-lg border ${getPriorityColor(recommendation.priority)}`}
                >
                  <div className="flex items-start justify-between mb-2">
                    <h4 className="font-medium">{recommendation.title}</h4>
                    <Badge variant="outline" className="text-xs">
                      {recommendation.priority.toUpperCase()}
                    </Badge>
                  </div>

                  <p className="text-sm text-gray-700 mb-3">
                    {recommendation.description}
                  </p>

                  <div className="space-y-2">
                    <h5 className="text-sm font-medium">Action Items:</h5>
                    <ul className="space-y-1">
                      {recommendation.actionItems.map((item, itemIndex) => (
                        <li key={itemIndex} className="text-sm text-gray-600 flex items-start gap-2">
                          <div className="w-1.5 h-1.5 bg-blue-600 rounded-full mt-2 flex-shrink-0" />
                          {item}
                        </li>
                      ))}
                    </ul>
                  </div>

                  <div className="flex items-center justify-between mt-3 pt-3 border-t border-gray-200">
                    <div className="text-xs text-gray-500">
                      Expected Impact: +{recommendation.expectedImpact} points
                    </div>
                    <div className="text-xs text-gray-500">
                      {recommendation.timeframe.replace('_', ' ')}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </TabsContent>
        </Tabs>

        <div className="mt-4 pt-4 border-t">
          <div className="flex items-center justify-between text-xs text-gray-500">
            <span>Last updated: {new Date(healthScore.lastUpdated).toLocaleDateString()}</span>
            <Button
              variant="outline"
              size="sm"
              onClick={loadHealthData}
              className="text-xs"
            >
              Refresh
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
