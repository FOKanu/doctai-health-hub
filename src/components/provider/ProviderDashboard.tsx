import React from 'react';
import { useNavigate } from 'react-router-dom';
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
  Shield,
  CheckCircle
} from 'lucide-react';
import { useAuth } from '@/contexts/AuthContext';

export function ProviderDashboard() {
  const { user } = useAuth();
  const navigate = useNavigate();

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
      <div className="bg-gradient-to-r from-blue-600 to-indigo-700 rounded-2xl p-8 text-white">
        <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between">
          <div className="mb-4 lg:mb-0">
            <h1 className="text-3xl font-bold">Welcome back, Dr. Sarah Johnson</h1>
            <p className="text-blue-100 mt-2 text-lg">
              Cardiology • AI-Powered Medical Care
            </p>
          </div>
          <Card className="bg-white/10 border-white/20 backdrop-blur-sm">
            <CardContent className="p-4">
              <div className="flex items-center space-x-3">
                <CheckCircle className="w-5 h-5 text-green-400" />
                <div>
                  <p className="text-sm text-blue-200">System Status</p>
                  <p className="font-semibold text-white">All Systems Operational •</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-6">
        {stats.map((stat, index) => (
          <Card 
            key={stat.title} 
            className="card-glass rounded-2xl hover:shadow-lg transition-all duration-300 cursor-pointer group"
            onClick={() => {
              // Navigate to relevant pages based on KPI
              switch (index) {
                case 0: // Active Patients
                  navigate('/provider/patients');
                  break;
                case 1: // Today's Appointments
                  navigate('/provider/schedule');
                  break;
                case 2: // Pending Alerts
                  navigate('/provider/ai-support');
                  break;
                case 3: // AI Diagnoses
                  navigate('/provider/ai-support');
                  break;
                default:
                  break;
              }
            }}
          >
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium text-gray-600">
                {stat.title}
              </CardTitle>
              <div className={`p-3 rounded-xl ${stat.bgColor} group-hover:scale-110 transition-transform`}>
                <stat.icon className={`w-5 h-5 ${stat.color}`} />
              </div>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold mb-1">{stat.value}</div>
              <p className="text-sm text-gray-500">
                <span className={stat.change.startsWith('+') ? 'text-green-600' : 'text-red-500'}>
                  {stat.change}
                </span> from last month
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
                <div 
                  key={patient.id} 
                  className="flex items-center space-x-4 p-4 rounded-xl hover:bg-gray-50/50 transition-colors cursor-pointer border border-gray-100"
                  onClick={() => navigate(`/provider/patients/${patient.id}`)}
                >
                  <Avatar className="h-12 w-12">
                    <AvatarImage src={patient.avatar} alt={patient.name} />
                    <AvatarFallback className="bg-blue-500 text-white font-medium">
                      {getInitials(patient.name)}
                    </AvatarFallback>
                  </Avatar>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="font-semibold text-gray-900">{patient.name}</p>
                        <p className="text-sm text-gray-500">{patient.age} years old • Last seen {patient.lastVisit}</p>
                      </div>
                      <div className="flex items-center space-x-2">
                        <Badge variant="outline" className="text-xs rounded-full px-2 py-1">
                          {patient.status}
                        </Badge>
                        <Badge className={`text-xs rounded-full px-2 py-1 ${getRiskColor(patient.risk)}`}>
                          {patient.risk}
                        </Badge>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
            <Button 
              variant="outline" 
              className="w-full mt-6 rounded-xl"
              onClick={() => navigate('/provider/patients')}
            >
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
                <div key={index} className="p-4 rounded-xl border border-gray-200/50 bg-white/50 hover:bg-white/80 transition-colors">
                  <div className="flex items-start justify-between mb-3">
                    <Badge variant="secondary" className="text-xs rounded-full px-2 py-1">
                      {insight.type}
                    </Badge>
                    <div className={`w-3 h-3 rounded-full ${getPriorityColor(insight.priority)}`}></div>
                  </div>
                  <p className="text-sm font-semibold text-gray-900 mb-2">
                    {insight.patient}
                  </p>
                  <p className="text-sm text-gray-600 mb-3 line-clamp-2">
                    {insight.insight}
                  </p>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <span className="text-xs text-gray-500">Confidence:</span>
                      <Badge variant="outline" className="text-xs">
                        {insight.confidence}%
                      </Badge>
                    </div>
                    <Button 
                      size="sm" 
                      variant="ghost" 
                      className="text-xs rounded-lg hover:bg-blue-50"
                      onClick={() => navigate(`/provider/ai-support?caseId=${index + 1}&patientId=${insight.patient.replace(' ', '-').toLowerCase()}`)}
                    >
                      Review
                    </Button>
                  </div>
                </div>
              ))}
            </div>
            <Button 
              variant="outline" 
              className="w-full mt-6 rounded-xl"
              onClick={() => navigate('/provider/ai-support')}
            >
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
            <Button 
              variant="outline" 
              className="h-24 flex-col space-y-2 rounded-xl hover:bg-blue-50/50 transition-colors group"
              onClick={() => navigate('/provider/patients')}
            >
              <Users className="w-7 h-7 group-hover:scale-110 transition-transform" />
              <span className="text-sm font-medium">New Patient</span>
            </Button>
            <Button 
              variant="outline" 
              className="h-24 flex-col space-y-2 rounded-xl hover:bg-blue-50/50 transition-colors group"
              onClick={() => navigate('/provider/schedule')}
            >
              <Calendar className="w-7 h-7 group-hover:scale-110 transition-transform" />
              <span className="text-sm font-medium">Schedule</span>
            </Button>
            <Button 
              variant="outline" 
              className="h-24 flex-col space-y-2 rounded-xl hover:bg-blue-50/50 transition-colors group"
              onClick={() => navigate('/provider/messages')}
            >
              <MessageSquare className="w-7 h-7 group-hover:scale-110 transition-transform" />
              <span className="text-sm font-medium">Messages</span>
            </Button>
            <Button 
              variant="outline" 
              className="h-24 flex-col space-y-2 rounded-xl hover:bg-blue-50/50 transition-colors group"
              onClick={() => navigate('/provider/patients')}
            >
              <FileText className="w-7 h-7 group-hover:scale-110 transition-transform" />
              <span className="text-sm font-medium">Records</span>
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
