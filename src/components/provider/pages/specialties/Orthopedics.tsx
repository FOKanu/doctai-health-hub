import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { useNavigate } from 'react-router-dom';
import { 
  Bone, 
  Activity, 
  Users, 
  Calendar, 
  FileText,
  Target,
  Timer,
  TrendingUp,
  CheckCircle,
  Clock
} from 'lucide-react';

export function Orthopedics() {
  const navigate = useNavigate();

  const rehabTemplates = [
    { name: 'ACL Recovery Protocol', duration: '16-20 weeks', phases: 4, patients: 12 },
    { name: 'Shoulder Impingement Rehab', duration: '8-12 weeks', phases: 3, patients: 8 },
    { name: 'Hip Replacement Recovery', duration: '12-16 weeks', phases: 4, patients: 15 },
    { name: 'Lower Back Pain Program', duration: '6-8 weeks', phases: 3, patients: 20 },
    { name: 'Rotator Cuff Repair', duration: '16-20 weeks', phases: 4, patients: 7 },
    { name: 'Ankle Sprain Recovery', duration: '4-6 weeks', phases: 2, patients: 5 }
  ];

  const activeRehabPlans = [
    { 
      patient: 'John Martinez', 
      condition: 'ACL Reconstruction', 
      progress: 75, 
      phase: 'Phase 3 - Strength Building',
      nextSession: 'Tomorrow 2:00 PM'
    },
    { 
      patient: 'Sarah Kim', 
      condition: 'Shoulder Surgery', 
      progress: 45, 
      phase: 'Phase 2 - Range of Motion',
      nextSession: 'Friday 10:00 AM'
    },
    { 
      patient: 'Mike Johnson', 
      condition: 'Hip Replacement', 
      progress: 90, 
      phase: 'Phase 4 - Return to Activity',
      nextSession: 'Monday 3:00 PM'
    }
  ];

  const getProgressColor = (progress: number) => {
    if (progress >= 80) return 'text-green-600';
    if (progress >= 50) return 'text-blue-600';
    return 'text-orange-600';
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Orthopedics</h1>
          <p className="text-gray-600 mt-1">Musculoskeletal specialty tools and insights</p>
        </div>
        <Button 
          onClick={() => navigate('/provider/patients?specialty=orthopedics')}
          className="bg-orange-600 hover:bg-orange-700"
        >
          <Users className="w-4 h-4 mr-2" />
          View Orthopedic Patients
        </Button>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-2xl font-bold text-orange-600">89</div>
                <div className="text-sm text-gray-600">Active Patients</div>
              </div>
              <Bone className="w-8 h-8 text-orange-600 opacity-75" />
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-2xl font-bold text-blue-600">67</div>
                <div className="text-sm text-gray-600">Active Rehab Plans</div>
              </div>
              <Activity className="w-8 h-8 text-blue-600 opacity-75" />
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-2xl font-bold text-green-600">15</div>
                <div className="text-sm text-gray-600">Completed Programs</div>
              </div>
              <CheckCircle className="w-8 h-8 text-green-600 opacity-75" />
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-2xl font-bold text-purple-600">23</div>
                <div className="text-sm text-gray-600">Sessions This Week</div>
              </div>
              <Calendar className="w-8 h-8 text-purple-600 opacity-75" />
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Rehabilitation Plan Templates */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <FileText className="w-5 h-5 text-blue-600" />
                <span>Rehab Plan Templates</span>
              </div>
              <Button variant="outline" size="sm">
                <Target className="w-4 h-4 mr-2" />
                Create New
              </Button>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {rehabTemplates.map((template, index) => (
                <div key={index} className="flex items-center justify-between p-3 border rounded-lg hover:bg-gray-50 cursor-pointer">
                  <div>
                    <h4 className="font-medium text-sm">{template.name}</h4>
                    <p className="text-xs text-gray-600">
                      {template.duration} • {template.phases} phases • {template.patients} active patients
                    </p>
                  </div>
                  <Button variant="ghost" size="sm">
                    Use Template
                  </Button>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Active Rehabilitation Plans */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Activity className="w-5 h-5 text-green-600" />
              <span>Active Rehabilitation Plans</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {activeRehabPlans.map((plan, index) => (
              <div key={index} className="p-4 border rounded-lg">
                <div className="flex items-center justify-between mb-3">
                  <div>
                    <h4 className="font-medium">{plan.patient}</h4>
                    <p className="text-sm text-gray-600">{plan.condition}</p>
                  </div>
                  <Badge variant="outline">{plan.phase}</Badge>
                </div>
                
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span>Progress</span>
                    <span className={`font-medium ${getProgressColor(plan.progress)}`}>
                      {plan.progress}%
                    </span>
                  </div>
                  <Progress value={plan.progress} className="h-2" />
                </div>
                
                <div className="flex items-center justify-between mt-3 pt-3 border-t">
                  <div className="flex items-center text-sm text-gray-600">
                    <Clock className="w-3 h-3 mr-1" />
                    {plan.nextSession}
                  </div>
                  <Button variant="outline" size="sm">
                    View Plan
                  </Button>
                </div>
              </div>
            ))}
          </CardContent>
        </Card>

        {/* Exercise Library */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Target className="w-5 h-5 text-purple-600" />
              <span>Exercise Library</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="grid grid-cols-2 gap-2">
              <Button variant="outline" size="sm" className="justify-start">
                <Activity className="w-3 h-3 mr-2" />
                Range of Motion
              </Button>
              <Button variant="outline" size="sm" className="justify-start">
                <Target className="w-3 h-3 mr-2" />
                Strength Training
              </Button>
              <Button variant="outline" size="sm" className="justify-start">
                <TrendingUp className="w-3 h-3 mr-2" />
                Balance & Stability
              </Button>
              <Button variant="outline" size="sm" className="justify-start">
                <Timer className="w-3 h-3 mr-2" />
                Endurance
              </Button>
              <Button variant="outline" size="sm" className="justify-start">
                <Bone className="w-3 h-3 mr-2" />
                Joint Mobility
              </Button>
              <Button variant="outline" size="sm" className="justify-start">
                <Activity className="w-3 h-3 mr-2" />
                Sport-Specific
              </Button>
            </div>
            <Button className="w-full mt-4">
              <FileText className="w-4 h-4 mr-2" />
              Browse Full Library
            </Button>
          </CardContent>
        </Card>

        {/* Recovery Milestones */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <TrendingUp className="w-5 h-5 text-green-600" />
              <span>Weekly Recovery Milestones</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-between p-3 bg-green-50 border border-green-200 rounded-lg">
              <div>
                <p className="text-sm font-medium text-green-800">Range of Motion Goal Achieved</p>
                <p className="text-xs text-green-600">Mike Johnson - Hip Replacement</p>
              </div>
              <CheckCircle className="w-5 h-5 text-green-600" />
            </div>

            <div className="flex items-center justify-between p-3 bg-blue-50 border border-blue-200 rounded-lg">
              <div>
                <p className="text-sm font-medium text-blue-800">Strength Training Phase Started</p>
                <p className="text-xs text-blue-600">John Martinez - ACL Recovery</p>
              </div>
              <Activity className="w-5 h-5 text-blue-600" />
            </div>

            <div className="flex items-center justify-between p-3 bg-orange-50 border border-orange-200 rounded-lg">
              <div>
                <p className="text-sm font-medium text-orange-800">Return to Activity Assessment Due</p>
                <p className="text-xs text-orange-600">Sarah Kim - Shoulder Surgery</p>
              </div>
              <Timer className="w-5 h-5 text-orange-600" />
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}