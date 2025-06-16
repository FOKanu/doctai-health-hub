
import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { RefreshCw, Brain, TrendingUp, Target, Zap, Heart, Trophy, Star } from 'lucide-react';

export const MotivationalTips: React.FC = () => {
  const [currentTipIndex, setCurrentTipIndex] = useState(0);

  const aiInsights = [
    {
      type: 'performance',
      icon: TrendingUp,
      title: 'Performance Pattern Analysis',
      message: "Your workout consistency has improved 40% this month! You're most active on Tuesday and Thursday mornings.",
      actionable: "Schedule your toughest workouts on these peak performance days for maximum results.",
      color: 'bg-green-50 border-green-200 text-green-800'
    },
    {
      type: 'recovery',
      icon: Heart,
      title: 'Recovery Optimization',
      message: "AI detected elevated resting heart rate after intense workouts. Your body needs 48 hours between strength sessions.",
      actionable: "Consider adding yoga or light cardio on recovery days to maintain momentum.",
      color: 'bg-blue-50 border-blue-200 text-blue-800'
    },
    {
      type: 'motivation',
      icon: Zap,
      title: 'Motivation Boost',
      message: "You've completed 85% of planned workouts this month - you're in the top 15% of users!",
      actionable: "Reward yourself with new workout gear or a massage to celebrate this achievement.",
      color: 'bg-purple-50 border-purple-200 text-purple-800'
    },
    {
      type: 'goal',
      icon: Target,
      title: 'Goal Adjustment',
      message: "Based on your progress, you're likely to reach your weight goal 2 weeks early.",
      actionable: "Consider setting a new challenge: increase your strength training frequency or try a new sport.",
      color: 'bg-orange-50 border-orange-200 text-orange-800'
    }
  ];

  const dailyMotivation = [
    {
      quote: "The only bad workout is the one that didn't happen.",
      author: "Fitness Wisdom",
      category: "Consistency"
    },
    {
      quote: "Your body can do almost anything. It's your mind you have to convince.",
      author: "Health Psychology",
      category: "Mental Strength"
    },
    {
      quote: "Progress, not perfection, is the goal.",
      author: "Wellness Coach",
      category: "Growth Mindset"
    },
    {
      quote: "Every workout is a step closer to your best self.",
      author: "Fitness Philosophy",
      category: "Self-Improvement"
    },
    {
      quote: "Consistency beats intensity when it comes to long-term results.",
      author: "Exercise Science",
      category: "Sustainable Habits"
    }
  ];

  const weeklyGoals = [
    {
      title: "Hydration Hero",
      description: "Drink 8 glasses of water daily for 7 days",
      progress: 5,
      total: 7,
      reward: "50 bonus health points",
      icon: "💧"
    },
    {
      title: "Early Bird Special",
      description: "Complete morning workouts 3 times this week",
      progress: 2,
      total: 3,
      reward: "75 bonus health points",
      icon: "🌅"
    },
    {
      title: "Strength Warrior",
      description: "Complete 4 strength training sessions",
      progress: 1,
      total: 4,
      reward: "100 bonus health points",
      icon: "💪"
    },
    {
      title: "Mindful Movement",
      description: "Include 15 min flexibility work in 5 workouts",
      progress: 3,
      total: 5,
      reward: "60 bonus health points",
      icon: "🧘"
    }
  ];

  const getCurrentTip = () => dailyMotivation[currentTipIndex];
  
  const nextTip = () => {
    setCurrentTipIndex((prev) => (prev + 1) % dailyMotivation.length);
  };

  const currentTip = getCurrentTip();

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-2xl font-bold text-gray-900">Motivation & AI Insights</h2>
        <p className="text-gray-600">Personalized tips and encouragement based on your fitness patterns</p>
      </div>

      {/* Daily Motivation */}
      <Card className="bg-gradient-to-r from-indigo-50 to-purple-50 border-indigo-200">
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-2 text-indigo-900">
              <Star className="w-5 h-5" />
              Daily Motivation
            </CardTitle>
            <Button variant="outline" size="sm" onClick={nextTip}>
              <RefreshCw className="w-4 h-4" />
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          <div className="text-center space-y-4">
            <blockquote className="text-lg font-medium text-indigo-900 italic">
              "{currentTip.quote}"
            </blockquote>
            <div className="flex items-center justify-center space-x-4">
              <span className="text-sm text-indigo-600">— {currentTip.author}</span>
              <Badge className="bg-indigo-100 text-indigo-800">
                {currentTip.category}
              </Badge>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* AI Performance Insights */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="w-5 h-5" />
            AI Performance Insights
          </CardTitle>
          <CardDescription>
            Smart analysis of your fitness patterns and personalized recommendations
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {aiInsights.map((insight, index) => {
              const IconComponent = insight.icon;
              return (
                <div key={index} className={`p-4 rounded-lg border-2 ${insight.color}`}>
                  <div className="flex items-start space-x-3">
                    <div className="p-2 bg-white rounded-lg">
                      <IconComponent className="w-5 h-5" />
                    </div>
                    <div className="flex-1">
                      <h3 className="font-semibold mb-2">{insight.title}</h3>
                      <p className="text-sm mb-3">{insight.message}</p>
                      <div className="bg-white p-3 rounded-lg border">
                        <p className="text-sm font-medium">💡 Actionable Insight:</p>
                        <p className="text-sm mt-1">{insight.actionable}</p>
                      </div>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </CardContent>
      </Card>

      {/* Weekly Challenges */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Trophy className="w-5 h-5" />
            Weekly Challenges
          </CardTitle>
          <CardDescription>
            Special goals to keep you motivated and engaged
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {weeklyGoals.map((goal, index) => {
              const progressPercentage = (goal.progress / goal.total) * 100;
              const isCompleted = goal.progress >= goal.total;
              
              return (
                <div key={index} className={`p-4 rounded-lg border-2 ${
                  isCompleted ? 'border-green-300 bg-green-50' : 'border-gray-200 bg-white'
                }`}>
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex items-center space-x-2">
                      <span className="text-2xl">{goal.icon}</span>
                      <div>
                        <h3 className={`font-semibold ${isCompleted ? 'text-green-800' : 'text-gray-900'}`}>
                          {goal.title}
                        </h3>
                        <p className="text-sm text-gray-600">{goal.description}</p>
                      </div>
                    </div>
                    {isCompleted && <Trophy className="w-5 h-5 text-yellow-500" />}
                  </div>
                  
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>{goal.progress}/{goal.total} completed</span>
                      <span className="text-orange-600">{goal.reward}</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div 
                        className={`h-2 rounded-full transition-all duration-300 ${
                          isCompleted ? 'bg-green-500' : 'bg-blue-500'
                        }`}
                        style={{ width: `${progressPercentage}%` }}
                      ></div>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </CardContent>
      </Card>

      {/* Success Stories */}
      <Card className="bg-gradient-to-r from-green-50 to-emerald-50 border-green-200">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-green-900">
            <Trophy className="w-5 h-5" />
            Your Success Stories
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="text-center p-4 bg-white rounded-lg border border-green-200">
              <div className="text-2xl font-bold text-green-700">24</div>
              <div className="text-sm text-green-600">Workouts Completed</div>
              <div className="text-xs text-gray-500 mt-1">This month</div>
            </div>
            <div className="text-center p-4 bg-white rounded-lg border border-green-200">
              <div className="text-2xl font-bold text-green-700">15%</div>
              <div className="text-sm text-green-600">Strength Increase</div>
              <div className="text-xs text-gray-500 mt-1">In 4 weeks</div>
            </div>
            <div className="text-center p-4 bg-white rounded-lg border border-green-200">
              <div className="text-2xl font-bold text-green-700">3</div>
              <div className="text-sm text-green-600">New PRs Set</div>
              <div className="text-xs text-gray-500 mt-1">Personal records</div>
            </div>
          </div>
          <div className="mt-4 text-center">
            <p className="text-sm text-green-800 font-medium">
              🎉 Amazing progress! You're becoming stronger and more consistent every week.
            </p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
