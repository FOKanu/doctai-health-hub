import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { useNavigate } from 'react-router-dom';
import { 
  Heart, 
  Activity, 
  AlertTriangle, 
  TrendingUp, 
  Users, 
  FileText,
  Upload,
  Zap,
  Calculator
} from 'lucide-react';

export function Cardiology() {
  const navigate = useNavigate();

  const riskStratificationData = [
    { level: 'Low Risk', count: 45, percentage: 60, color: 'bg-green-500' },
    { level: 'Moderate Risk', count: 20, percentage: 27, color: 'bg-yellow-500' },
    { level: 'High Risk', count: 10, percentage: 13, color: 'bg-red-500' }
  ];

  const quickOrders = [
    'Echocardiogram',
    'Stress Test',
    'Holter Monitor', 
    'Lipid Panel',
    'BNP/NT-proBNP',
    'Troponin'
  ];

  const recentECGs = [
    { id: 1, patient: 'John Doe', timestamp: '10 min ago', status: 'Normal Sinus Rhythm', risk: 'low' },
    { id: 2, patient: 'Jane Smith', timestamp: '25 min ago', status: 'Atrial Fibrillation', risk: 'high' },
    { id: 3, patient: 'Bob Johnson', timestamp: '1 hour ago', status: 'ST Elevation', risk: 'high' },
  ];

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Cardiology</h1>
          <p className="text-gray-600 mt-1">Cardiovascular specialty tools and insights</p>
        </div>
        <Button 
          onClick={() => navigate('/provider/patients?specialty=cardiology')}
          className="bg-red-600 hover:bg-red-700"
        >
          <Users className="w-4 h-4 mr-2" />
          View Cardiology Patients
        </Button>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-2xl font-bold text-red-600">75</div>
                <div className="text-sm text-gray-600">Active Patients</div>
              </div>
              <Heart className="w-8 h-8 text-red-600 opacity-75" />
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-2xl font-bold text-orange-600">23</div>
                <div className="text-sm text-gray-600">High Risk Patients</div>
              </div>
              <AlertTriangle className="w-8 h-8 text-orange-600 opacity-75" />
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-2xl font-bold text-blue-600">12</div>
                <div className="text-sm text-gray-600">ECGs Today</div>
              </div>
              <Activity className="w-8 h-8 text-blue-600 opacity-75" />
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-2xl font-bold text-green-600">8.5</div>
                <div className="text-sm text-gray-600">Avg Risk Score</div>
              </div>
              <TrendingUp className="w-8 h-8 text-green-600 opacity-75" />
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Risk Stratification Widget */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Calculator className="w-5 h-5 text-red-600" />
              <span>Risk Stratification</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {riskStratificationData.map((risk, index) => (
              <div key={index} className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-sm font-medium">{risk.level}</span>
                  <span className="text-sm text-gray-600">{risk.count} patients ({risk.percentage}%)</span>
                </div>
                <Progress value={risk.percentage} className="h-2" />
              </div>
            ))}
            <div className="pt-2 border-t">
              <Button variant="outline" className="w-full">
                <Calculator className="w-4 h-4 mr-2" />
                Calculate Risk Score
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* ECG Upload & Analysis */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Activity className="w-5 h-5 text-blue-600" />
              <span>ECG Analysis</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center">
              <Upload className="w-8 h-8 text-gray-400 mx-auto mb-2" />
              <p className="text-sm text-gray-600 mb-2">Upload ECG for AI analysis</p>
              <Button variant="outline">
                Select ECG File
              </Button>
            </div>
            
            <div className="space-y-2">
              <h4 className="font-medium">Recent ECG Analyses</h4>
              {recentECGs.map((ecg) => (
                <div key={ecg.id} className="flex items-center justify-between p-2 bg-gray-50 rounded">
                  <div>
                    <span className="text-sm font-medium">{ecg.patient}</span>
                    <span className="text-xs text-gray-500 ml-2">{ecg.timestamp}</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Badge variant={ecg.risk === 'high' ? 'destructive' : 'secondary'}>
                      {ecg.status}
                    </Badge>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Quick Orders */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <FileText className="w-5 h-5 text-green-600" />
              <span>Quick Orders</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-2">
              {quickOrders.map((order, index) => (
                <Button key={index} variant="outline" size="sm" className="justify-start">
                  <Zap className="w-3 h-3 mr-2" />
                  {order}
                </Button>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Patient Alerts */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <AlertTriangle className="w-5 h-5 text-orange-600" />
              <span>Patient Alerts</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex items-start space-x-3 p-3 bg-red-50 border border-red-200 rounded-lg">
              <AlertTriangle className="w-4 h-4 text-red-600 mt-0.5" />
              <div>
                <p className="text-sm font-medium text-red-800">Critical: Elevated Troponin</p>
                <p className="text-xs text-red-600">Sarah Johnson - 15 min ago</p>
              </div>
            </div>
            
            <div className="flex items-start space-x-3 p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
              <AlertTriangle className="w-4 h-4 text-yellow-600 mt-0.5" />
              <div>
                <p className="text-sm font-medium text-yellow-800">Medication Adherence Alert</p>
                <p className="text-xs text-yellow-600">Mike Davis - 1 hour ago</p>
              </div>
            </div>

            <div className="flex items-start space-x-3 p-3 bg-orange-50 border border-orange-200 rounded-lg">
              <AlertTriangle className="w-4 h-4 text-orange-600 mt-0.5" />
              <div>
                <p className="text-sm font-medium text-orange-800">Follow-up Overdue</p>
                <p className="text-xs text-orange-600">Lisa Brown - 2 hours ago</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}