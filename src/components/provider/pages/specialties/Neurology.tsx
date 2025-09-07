import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { useNavigate } from 'react-router-dom';
import { 
  Brain, 
  Zap, 
  AlertTriangle, 
  TrendingUp, 
  Users, 
  Activity,
  Bell,
  Calendar
} from 'lucide-react';

function Neurology() {
  const navigate = useNavigate();

  const seizureRiskData = [
    { patient: 'Emma Thompson', riskScore: 85, status: 'High Risk', lastSeizure: '3 days ago', trend: 'increasing' },
    { patient: 'James Wilson', riskScore: 65, status: 'Moderate Risk', lastSeizure: '2 weeks ago', trend: 'stable' },
    { patient: 'Maria Garcia', riskScore: 45, status: 'Low Risk', lastSeizure: '3 months ago', trend: 'decreasing' },
    { patient: 'Robert Chen', riskScore: 92, status: 'Critical Risk', lastSeizure: '1 day ago', trend: 'increasing' },
  ];

  const getRiskColor = (score: number) => {
    if (score >= 80) return 'text-red-600 bg-red-100 border-red-200';
    if (score >= 60) return 'text-orange-600 bg-orange-100 border-orange-200';
    if (score >= 40) return 'text-yellow-600 bg-yellow-100 border-yellow-200';
    return 'text-green-600 bg-green-100 border-green-200';
  };

  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case 'increasing':
        return <TrendingUp className="w-4 h-4 text-red-600" />;
      case 'stable':
        return <Activity className="w-4 h-4 text-yellow-600" />;
      case 'decreasing':
        return <TrendingUp className="w-4 h-4 text-green-600 transform rotate-180" />;
      default:
        return <Activity className="w-4 h-4 text-gray-600" />;
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Neurology</h1>
          <p className="text-gray-600 mt-1">Neurological specialty tools and insights</p>
        </div>
        <Button 
          onClick={() => navigate('/provider/patients?specialty=neurology')}
          className="bg-purple-600 hover:bg-purple-700"
        >
          <Users className="w-4 h-4 mr-2" />
          View Neurology Patients
        </Button>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-2xl font-bold text-purple-600">42</div>
                <div className="text-sm text-gray-600">Active Patients</div>
              </div>
              <Brain className="w-8 h-8 text-purple-600 opacity-75" />
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-2xl font-bold text-red-600">8</div>
                <div className="text-sm text-gray-600">High Seizure Risk</div>
              </div>
              <AlertTriangle className="w-8 h-8 text-red-600 opacity-75" />
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-2xl font-bold text-orange-600">15</div>
                <div className="text-sm text-gray-600">Recent Episodes</div>
              </div>
              <Zap className="w-8 h-8 text-orange-600 opacity-75" />
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-2xl font-bold text-blue-600">23</div>
                <div className="text-sm text-gray-600">EEG Reviews</div>
              </div>
              <Activity className="w-8 h-8 text-blue-600 opacity-75" />
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Seizure Risk Monitor */}
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Zap className="w-5 h-5 text-orange-600" />
              <span>Seizure Risk Monitor</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {seizureRiskData.map((patient, index) => (
                <div key={index} className="p-4 border rounded-lg">
                  <div className="flex items-center justify-between mb-3">
                    <div>
                      <h4 className="font-medium">{patient.patient}</h4>
                      <p className="text-sm text-gray-600">Last seizure: {patient.lastSeizure}</p>
                    </div>
                    <div className="flex items-center space-x-2">
                      {getTrendIcon(patient.trend)}
                      <Badge className={getRiskColor(patient.riskScore)}>
                        {patient.status}
                      </Badge>
                    </div>
                  </div>
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>Risk Score</span>
                      <span className="font-medium">{patient.riskScore}/100</span>
                    </div>
                    <Progress 
                      value={patient.riskScore} 
                      className="h-2"
                    />
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* EEG Analysis Queue */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Activity className="w-5 h-5 text-blue-600" />
              <span>EEG Analysis Queue</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex items-center justify-between p-3 bg-blue-50 border border-blue-200 rounded-lg">
              <div>
                <p className="text-sm font-medium">Routine EEG - Sarah Kim</p>
                <p className="text-xs text-gray-600">Scheduled: Today 2:00 PM</p>
              </div>
              <Badge variant="secondary">Pending</Badge>
            </div>
            
            <div className="flex items-center justify-between p-3 bg-green-50 border border-green-200 rounded-lg">
              <div>
                <p className="text-sm font-medium">Sleep Study - John Doe</p>
                <p className="text-xs text-gray-600">Completed: Yesterday</p>
              </div>
              <Badge className="bg-green-100 text-green-800">Ready for Review</Badge>
            </div>

            <div className="flex items-center justify-between p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
              <div>
                <p className="text-sm font-medium">24hr EEG - Lisa Brown</p>
                <p className="text-xs text-gray-600">In Progress: Started 6 hours ago</p>
              </div>
              <Badge variant="outline">Monitoring</Badge>
            </div>

            <Button variant="outline" className="w-full">
              <Calendar className="w-4 h-4 mr-2" />
              Schedule EEG
            </Button>
          </CardContent>
        </Card>

        {/* Patient Alerts */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Bell className="w-5 h-5 text-red-600" />
              <span>Neurological Alerts</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex items-start space-x-3 p-3 bg-red-50 border border-red-200 rounded-lg">
              <AlertTriangle className="w-4 h-4 text-red-600 mt-0.5" />
              <div>
                <p className="text-sm font-medium text-red-800">Seizure Activity Detected</p>
                <p className="text-xs text-red-600">Robert Chen - 30 min ago</p>
                <p className="text-xs text-red-600">Emergency protocol activated</p>
              </div>
            </div>
            
            <div className="flex items-start space-x-3 p-3 bg-orange-50 border border-orange-200 rounded-lg">
              <AlertTriangle className="w-4 h-4 text-orange-600 mt-0.5" />
              <div>
                <p className="text-sm font-medium text-orange-800">Medication Adjustment Needed</p>
                <p className="text-xs text-orange-600">Emma Thompson - 2 hours ago</p>
                <p className="text-xs text-orange-600">Seizure frequency increasing</p>
              </div>
            </div>

            <div className="flex items-start space-x-3 p-3 bg-blue-50 border border-blue-200 rounded-lg">
              <Bell className="w-4 h-4 text-blue-600 mt-0.5" />
              <div>
                <p className="text-sm font-medium text-blue-800">EEG Results Available</p>
                <p className="text-xs text-blue-600">Maria Garcia - 4 hours ago</p>
                <p className="text-xs text-blue-600">Abnormal patterns detected</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

export default Neurology;