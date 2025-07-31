
import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import {
  Activity,
  Camera,
  Target,
  TrendingUp,
  Calendar,
  Award,
  Scan,
  RefreshCw,
  Settings,
  BarChart3,
  Brain,
  Heart,
  Flame,
  CheckCircle2,
  AlertCircle,
  Clock
} from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { WeeklyTrendsChart } from './results/WeeklyTrendsChart';
import { HealthOverviewCards } from './results/HealthOverviewCards';
import { QuickActions } from './results/QuickActions';

const ResultsScreen = () => {
  const navigate = useNavigate();

  // Sample health data - in a real app, this would come from API/database
  const healthSummary = {
    skinHealth: {
      status: 'Clear',
      lastScan: '2 days ago',
      confidence: 92,
      trend: 'improving'
    },
    fitnessStreak: {
      days: 5,
      goal: 7,
      percentage: 71
    },
    dietCompliance: {
      score: 85,
      streak: 3,
      trend: 'stable'
    },
    overallHealth: {
      score: 78,
      trend: 'improving',
      change: '+5'
    }
  };

  const weeklyMetrics = [
    { day: 'Mon', steps: 8500, calories: 420, workouts: 1 },
    { day: 'Tue', steps: 9200, calories: 380, workouts: 1 },
    { day: 'Wed', steps: 7800, calories: 340, workouts: 0 },
    { day: 'Thu', steps: 10200, calories: 450, workouts: 1 },
    { day: 'Fri', steps: 8900, calories: 400, workouts: 1 },
    { day: 'Sat', steps: 6500, calories: 280, workouts: 0 },
    { day: 'Sun', steps: 7200, calories: 320, workouts: 1 }
  ];

  const recentAchievements = [
    { title: 'Workout Warrior', description: '5-day streak completed', icon: Award, color: 'text-yellow-600' },
    { title: 'Health Scan', description: 'Clear results received', icon: CheckCircle2, color: 'text-green-600' },
    { title: 'Calorie Goal', description: 'Met daily target 4/7 days', icon: Target, color: 'text-blue-600' }
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Health Results</h1>
          <p className="text-gray-600">Your personalized health insights and progress summary</p>
        </div>
        <div className="flex items-center gap-3">
          <Badge variant="outline" className="text-green-600 border-green-200">
            <TrendingUp className="w-3 h-3 mr-1" />
            Overall Improving
          </Badge>
                      <Button variant="outline" size="sm" onClick={() => navigate('/patient/settings')}>
            <Settings className="w-4 h-4 mr-2" />
            Adjust Goals
          </Button>
        </div>
      </div>

      {/* Health Overview Cards */}
      <HealthOverviewCards healthSummary={healthSummary} />

      {/* Weekly Trends Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Trends Chart */}
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BarChart3 className="w-5 h-5" />
              Weekly Health Trends
            </CardTitle>
            <CardDescription>
              Your activity, fitness, and health metrics over the past week
            </CardDescription>
          </CardHeader>
          <CardContent>
            <WeeklyTrendsChart data={weeklyMetrics} />
          </CardContent>
        </Card>
      </div>

      {/* Recent Achievements */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Award className="w-5 h-5" />
            Recent Achievements
          </CardTitle>
          <CardDescription>
            Your latest health milestones and accomplishments
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {recentAchievements.map((achievement, index) => (
              <div key={index} className="flex items-center space-x-3 p-3 rounded-lg bg-gray-50">
                <achievement.icon className={`w-8 h-8 ${achievement.color}`} />
                <div>
                  <h3 className="font-semibold text-gray-900">{achievement.title}</h3>
                  <p className="text-sm text-gray-600">{achievement.description}</p>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Quick Actions */}
      <QuickActions onNavigate={navigate} />

      {/* AI Insights Section */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="w-5 h-5" />
            AI Health Insights
          </CardTitle>
          <CardDescription>
            Personalized recommendations based on your health data
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-start space-x-3 p-4 bg-blue-50 rounded-lg">
            <TrendingUp className="w-5 h-5 text-blue-600 mt-0.5" />
            <div>
              <h3 className="font-semibold text-blue-900">Fitness Progress</h3>
              <p className="text-sm text-blue-700">
                You're on track to meet your weekly step goal. Consider adding 2 more strength training sessions this week.
              </p>
            </div>
          </div>

          <div className="flex items-start space-x-3 p-4 bg-green-50 rounded-lg">
            <CheckCircle2 className="w-5 h-5 text-green-600 mt-0.5" />
            <div>
              <h3 className="font-semibold text-green-900">Skin Health</h3>
              <p className="text-sm text-green-700">
                Your recent scan shows excellent skin health. Continue your current skincare routine and schedule next scan in 30 days.
              </p>
            </div>
          </div>

          <div className="flex items-start space-x-3 p-4 bg-yellow-50 rounded-lg">
            <Clock className="w-5 h-5 text-yellow-600 mt-0.5" />
            <div>
              <h3 className="font-semibold text-yellow-900">Diet Optimization</h3>
              <p className="text-sm text-yellow-700">
                Consider increasing protein intake by 15g daily to support your fitness goals. Check our meal recommendations.
              </p>
            </div>
          </div>

          <div className="flex justify-center pt-4">
            <Button onClick={() => navigate('/patient/analytics')} className="bg-purple-600 hover:bg-purple-700">
              <Brain className="w-4 h-4 mr-2" />
              View Detailed AI Analysis
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default ResultsScreen;
