import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import {
  Users,
  Calendar,
  AlertTriangle,
  TrendingUp,
  Activity,
  Heart,
  Brain,
  Eye,
  Clock,
  MessageSquare,
  FileText,
  Shield
} from 'lucide-react';
import { useAuth } from '@/contexts/AuthContext';

export function ProviderDashboard() {
  const { user } = useAuth();

  // Mock data for the dashboard
  const stats = [
    {
      title: "Active Patients",
      value: "247",
      change: "+12%",
      icon: Users,
      color: "text-blue-600",
      bgColor: "bg-blue-50"
    },
    {
      title: "Today's Appointments",
      value: "18",
      change: "+3",
      icon: Calendar,
      color: "text-green-600",
      bgColor: "bg-green-50"
    },
    {
      title: "Pending Alerts",
      value: "5",
      change: "-2",
      icon: AlertTriangle,
      color: "text-orange-600",
      bgColor: "bg-orange-50"
    },
    {
      title: "AI Diagnoses",
      value: "89%",
      change: "+5%",
      icon: Brain,
      color: "text-purple-600",
      bgColor: "bg-purple-50"
    }
  ];

  const recentPatients = [
    {
      id: "1",
      name: "Sarah Johnson",
      age: 34,
      lastVisit: "2 days ago",
      status: "Follow-up",
      risk: "Low",
      avatar: "/avatars/sarah.jpg"
    },
    {
      id: "2",
      name: "Michael Chen",
      age: 52,
      lastVisit: "1 week ago",
      status: "New Patient",
      risk: "Medium",
      avatar: "/avatars/michael.jpg"
    },
    {
      id: "3",
      name: "Emily Rodriguez",
      age: 28,
      lastVisit: "3 days ago",
      status: "Routine",
      risk: "Low",
      avatar: "/avatars/emily.jpg"
    }
  ];

  const aiInsights = [
    {
      type: "Risk Assessment",
      patient: "Sarah Johnson",
      insight: "Cardiovascular risk increased by 15%",
      confidence: 92,
      priority: "High"
    },
    {
      type: "Diagnostic Support",
      patient: "Michael Chen",
      insight: "X-ray analysis suggests early-stage pneumonia",
      confidence: 87,
      priority: "Medium"
    },
    {
      type: "Treatment Recommendation",
      patient: "Emily Rodriguez",
      insight: "Consider adjusting medication dosage",
      confidence: 94,
      priority: "Medium"
    }
  ];

  const getInitials = (name: string) => {
    return name
      .split(' ')
      .map(n => n[0])
      .join('')
      .toUpperCase()
      .slice(0, 2);
  };

  const getRiskColor = (risk: string) => {
    switch (risk.toLowerCase()) {
      case 'high':
        return 'bg-red-100 text-red-800';
      case 'medium':
        return 'bg-yellow-100 text-yellow-800';
      case 'low':
        return 'bg-green-100 text-green-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  const getPriorityColor = (priority: string) => {
    switch (priority.toLowerCase()) {
      case 'high':
        return 'bg-red-500';
      case 'medium':
        return 'bg-yellow-500';
      case 'low':
        return 'bg-green-500';
      default:
        return 'bg-gray-500';
    }
  };

  return (
    <div className="space-y-6">
      {/* Welcome Section */}
      <div className="bg-gradient-to-r from-blue-600 to-indigo-700 rounded-lg p-6 text-white">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold">Welcome back, {user?.name}</h1>
            <p className="text-blue-100 mt-1">
              {user?.specialty ? `${user.specialty} • ` : ''}AI-Powered Medical Care
            </p>
          </div>
          <div className="flex items-center space-x-4">
            <div className="text-right">
              <p className="text-sm text-blue-200">System Status</p>
              <p className="font-semibold">All Systems Operational</p>
            </div>
            <div className="w-3 h-3 bg-green-400 rounded-full"></div>
          </div>
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {stats.map((stat) => (
          <Card key={stat.title} className="card-glass rounded-2xl hover:shadow-lg transition-all duration-300">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium text-gray-600">
                {stat.title}
              </CardTitle>
              <div className={`p-2 rounded-lg ${stat.bgColor}`}>
                <stat.icon className={`w-4 h-4 ${stat.color}`} />
              </div>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{stat.value}</div>
              <p className="text-xs text-gray-500 mt-1">
                <span className="text-green-600">{stat.change}</span> from last month
              </p>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Recent Patients */}
        <Card className="lg:col-span-2 card-glass rounded-2xl">
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Users className="w-5 h-5" />
              <span>Recent Patients</span>
            </CardTitle>
            <CardDescription>
              Latest patient interactions and status updates
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {recentPatients.map((patient) => (
                <div key={patient.id} className="flex items-center space-x-4 p-3 rounded-xl hover:bg-gray-50/50 transition-colors cursor-pointer">
                  <Avatar className="h-10 w-10">
                    <AvatarImage src={patient.avatar} alt={patient.name} />
                    <AvatarFallback className="bg-blue-500 text-white">
                      {getInitials(patient.name)}
                    </AvatarFallback>
                  </Avatar>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="font-medium text-gray-900">{patient.name}</p>
                        <p className="text-sm text-gray-500">{patient.age} years • {patient.lastVisit}</p>
                      </div>
                      <div className="flex items-center space-x-2">
                        <Badge variant="outline" className="text-xs rounded-full">
                          {patient.status}
                        </Badge>
                        <Badge className={`text-xs rounded-full ${getRiskColor(patient.risk)}`}>
                          {patient.risk} Risk
                        </Badge>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
            <Button variant="outline" className="w-full mt-4 rounded-xl">
              View All Patients
            </Button>
          </CardContent>
        </Card>

        {/* AI Insights */}
        <Card className="card-glass rounded-2xl">
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Brain className="w-5 h-5" />
              <span>AI Insights</span>
            </CardTitle>
            <CardDescription>
              AI-powered diagnostic recommendations
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {aiInsights.map((insight, index) => (
                <div key={index} className="p-3 rounded-xl border border-gray-200/50 bg-white/30">
                  <div className="flex items-start justify-between mb-2">
                    <Badge variant="secondary" className="text-xs rounded-full">
                      {insight.type}
                    </Badge>
                    <div className={`w-2 h-2 rounded-full ${getPriorityColor(insight.priority)}`}></div>
                  </div>
                  <p className="text-sm font-medium text-gray-900 mb-1">
                    {insight.patient}
                  </p>
                  <p className="text-xs text-gray-600 mb-2">
                    {insight.insight}
                  </p>
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-gray-500">
                      Confidence: {insight.confidence}%
                    </span>
                    <Button size="sm" variant="ghost" className="text-xs rounded-lg">
                      Review
                    </Button>
                  </div>
                </div>
              ))}
            </div>
            <Button variant="outline" className="w-full mt-4 rounded-xl">
              View All Insights
            </Button>
          </CardContent>
        </Card>
      </div>

      {/* Quick Actions */}
      <Card className="card-glass rounded-2xl">
        <CardHeader>
          <CardTitle>Quick Actions</CardTitle>
          <CardDescription>
            Common tasks and shortcuts for efficient workflow
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <Button variant="outline" className="h-20 flex-col space-y-2 rounded-xl hover:bg-blue-50/50 transition-colors">
              <Users className="w-6 h-6" />
              <span className="text-sm">New Patient</span>
            </Button>
            <Button variant="outline" className="h-20 flex-col space-y-2 rounded-xl hover:bg-blue-50/50 transition-colors">
              <Calendar className="w-6 h-6" />
              <span className="text-sm">Schedule</span>
            </Button>
            <Button variant="outline" className="h-20 flex-col space-y-2 rounded-xl hover:bg-blue-50/50 transition-colors">
              <MessageSquare className="w-6 h-6" />
              <span className="text-sm">Messages</span>
            </Button>
            <Button variant="outline" className="h-20 flex-col space-y-2 rounded-xl hover:bg-blue-50/50 transition-colors">
              <FileText className="w-6 h-6" />
              <span className="text-sm">Records</span>
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
