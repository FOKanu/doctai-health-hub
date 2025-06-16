
import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Brain, TrendingUp, Heart, Moon, Dumbbell, Apple, Calendar, Settings } from 'lucide-react';
import { WorkoutRecommendations } from './ai-recommendations/WorkoutRecommendations';
import { DietRecommendations } from './ai-recommendations/DietRecommendations';
import { RecoveryRecommendations } from './ai-recommendations/RecoveryRecommendations';
import { PriorityImprovements } from './ai-recommendations/PriorityImprovements';

const AIRecommendationsScreen = () => {
  const [activeTab, setActiveTab] = useState('overview');

  // Mock AI insights summary
  const aiInsights = {
    overallScore: 78,
    trend: 'improving',
    lastUpdated: '2 hours ago',
    topPriorities: [
      'Increase leg workout frequency',
      'Improve sleep consistency',
      'Add more vegetables to diet'
    ]
  };

  return (
    <div className="min-h-screen bg-gray-50 pb-20">
      {/* Header */}
      <div className="bg-gradient-to-r from-purple-600 to-blue-600 text-white">
        <div className="p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-white/20 rounded-lg">
                <Brain className="w-6 h-6" />
              </div>
              <div>
                <h1 className="text-xl font-semibold">AI Recommendations</h1>
                <p className="text-sm opacity-90">Personalized wellness insights</p>
              </div>
            </div>
            <Button variant="ghost" size="sm" className="text-white hover:bg-white/20">
              <Settings className="w-4 h-4 mr-2" />
              Settings
            </Button>
          </div>

          {/* AI Summary Card */}
          <div className="mt-4 bg-white/10 backdrop-blur-sm rounded-lg p-4">
            <div className="flex items-center justify-between mb-3">
              <h3 className="font-medium">Wellness Score</h3>
              <Badge variant="secondary" className="bg-white/20 text-white border-white/20">
                Updated {aiInsights.lastUpdated}
              </Badge>
            </div>
            <div className="flex items-center space-x-4">
              <div className="text-center">
                <div className="text-3xl font-bold">{aiInsights.overallScore}</div>
                <div className="text-xs opacity-80">Overall Score</div>
              </div>
              <div className="flex-1">
                <div className="flex items-center text-sm mb-1">
                  <TrendingUp className="w-4 h-4 mr-1" />
                  Trending {aiInsights.trend}
                </div>
                <div className="text-xs opacity-80">
                  Based on your fitness, diet, and recovery data
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="p-4">
        {/* Top Priorities Quick View */}
        <Card className="mb-4 border-l-4 border-l-orange-500">
          <CardHeader className="pb-3">
            <CardTitle className="text-base">Today's Priorities</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {aiInsights.topPriorities.map((priority, index) => (
                <div key={index} className="flex items-center justify-between p-2 bg-orange-50 rounded-lg">
                  <span className="text-sm font-medium text-orange-800">{priority}</span>
                  <Badge variant="outline" className="text-xs">
                    High
                  </Badge>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Main Content Tabs */}
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="overview" className="text-xs">Overview</TabsTrigger>
            <TabsTrigger value="workout" className="text-xs">Workout</TabsTrigger>
            <TabsTrigger value="nutrition" className="text-xs">Nutrition</TabsTrigger>
            <TabsTrigger value="recovery" className="text-xs">Recovery</TabsTrigger>
          </TabsList>

          <TabsContent value="overview" className="mt-4">
            <PriorityImprovements />
          </TabsContent>

          <TabsContent value="workout" className="mt-4">
            <WorkoutRecommendations />
          </TabsContent>

          <TabsContent value="nutrition" className="mt-4">
            <DietRecommendations />
          </TabsContent>

          <TabsContent value="recovery" className="mt-4">
            <RecoveryRecommendations />
          </TabsContent>
        </Tabs>

        {/* Schedule AI Updates */}
        <Card className="mt-6 bg-gradient-to-r from-blue-50 to-purple-50">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <Calendar className="w-5 h-5 text-blue-600" />
                <div>
                  <h4 className="font-medium">Weekly AI Plan Updates</h4>
                  <p className="text-sm text-gray-600">Every Sunday at 8:00 AM</p>
                </div>
              </div>
              <Button variant="outline" size="sm">
                Customize
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default AIRecommendationsScreen;
