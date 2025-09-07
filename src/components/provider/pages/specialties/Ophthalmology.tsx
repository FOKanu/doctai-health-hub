import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { useNavigate } from 'react-router-dom';
import { 
  Eye, 
  Scan, 
  Clock, 
  AlertTriangle, 
  Users, 
  Camera,
  Calendar,
  FileImage,
  CheckCircle,
  Timer
} from 'lucide-react';

function Ophthalmology() {
  const navigate = useNavigate();

  const imagingQueue = [
    { 
      id: 1, 
      patient: 'Alice Johnson', 
      type: 'Retinal Scan', 
      priority: 'Urgent', 
      scheduledTime: '10:30 AM',
      status: 'In Progress'
    },
    { 
      id: 2, 
      patient: 'David Lee', 
      type: 'OCT Scan', 
      priority: 'Routine', 
      scheduledTime: '11:00 AM',
      status: 'Scheduled'
    },
    { 
      id: 3, 
      patient: 'Susan White', 
      type: 'Fundus Photography', 
      priority: 'Follow-up', 
      scheduledTime: '11:30 AM',
      status: 'Scheduled'
    },
    { 
      id: 4, 
      patient: 'Michael Brown', 
      type: 'Visual Field Test', 
      priority: 'Urgent', 
      scheduledTime: '12:00 PM',
      status: 'Ready for Review'
    }
  ];

  const recentScans = [
    { patient: 'Emma Davis', type: 'Diabetic Retinopathy Screening', result: 'Mild NPDR detected', severity: 'moderate' },
    { patient: 'Tom Wilson', type: 'Glaucoma Assessment', result: 'Normal pressure, no damage', severity: 'low' },
    { patient: 'Lucy Chen', type: 'Macular Degeneration Scan', result: 'Early AMD changes', severity: 'moderate' },
  ];

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'Urgent':
        return 'bg-red-100 text-red-800 border-red-200';
      case 'Follow-up':
        return 'bg-orange-100 text-orange-800 border-orange-200';
      default:
        return 'bg-blue-100 text-blue-800 border-blue-200';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'In Progress':
        return <Timer className="w-4 h-4 text-orange-600" />;
      case 'Ready for Review':
        return <CheckCircle className="w-4 h-4 text-green-600" />;
      default:
        return <Clock className="w-4 h-4 text-blue-600" />;
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Ophthalmology</h1>
          <p className="text-gray-600 mt-1">Vision and eye care specialty tools</p>
        </div>
        <Button 
          onClick={() => navigate('/provider/patients?specialty=ophthalmology')}
          className="bg-blue-600 hover:bg-blue-700"
        >
          <Users className="w-4 h-4 mr-2" />
          View Ophthalmology Patients
        </Button>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-2xl font-bold text-blue-600">63</div>
                <div className="text-sm text-gray-600">Active Patients</div>
              </div>
              <Eye className="w-8 h-8 text-blue-600 opacity-75" />
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-2xl font-bold text-green-600">18</div>
                <div className="text-sm text-gray-600">Scans Today</div>
              </div>
              <Scan className="w-8 h-8 text-green-600 opacity-75" />
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-2xl font-bold text-orange-600">7</div>
                <div className="text-sm text-gray-600">Urgent Reviews</div>
              </div>
              <AlertTriangle className="w-8 h-8 text-orange-600 opacity-75" />
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-2xl font-bold text-purple-600">24</div>
                <div className="text-sm text-gray-600">Images Pending</div>
              </div>
              <FileImage className="w-8 h-8 text-purple-600 opacity-75" />
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Imaging Queue */}
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <Camera className="w-5 h-5 text-blue-600" />
                <span>Imaging Queue</span>
              </div>
              <Button variant="outline" size="sm">
                <Calendar className="w-4 h-4 mr-2" />
                Schedule Imaging
              </Button>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {imagingQueue.map((item) => (
                <div key={item.id} className="flex items-center justify-between p-4 border rounded-lg">
                  <div className="flex items-center space-x-3">
                    {getStatusIcon(item.status)}
                    <div>
                      <h4 className="font-medium">{item.patient}</h4>
                      <p className="text-sm text-gray-600">{item.type} • {item.scheduledTime}</p>
                    </div>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Badge className={getPriorityColor(item.priority)}>
                      {item.priority}
                    </Badge>
                    <Badge variant="outline">
                      {item.status}
                    </Badge>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Recent Scan Results */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Scan className="w-5 h-5 text-green-600" />
              <span>Recent Scan Results</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {recentScans.map((scan, index) => (
              <div key={index} className="p-3 border rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <h4 className="font-medium text-sm">{scan.patient}</h4>
                  <Badge variant={scan.severity === 'moderate' ? 'destructive' : 'secondary'}>
                    {scan.severity === 'low' ? 'Normal' : 'Requires Follow-up'}
                  </Badge>
                </div>
                <p className="text-sm text-gray-600 mb-1">{scan.type}</p>
                <p className="text-sm font-medium">{scan.result}</p>
              </div>
            ))}
            <Button variant="outline" className="w-full">
              <Eye className="w-4 h-4 mr-2" />
              View All Results
            </Button>
          </CardContent>
        </Card>

        {/* Quick Actions */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <FileImage className="w-5 h-5 text-purple-600" />
              <span>Quick Actions</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <Button className="w-full justify-start" variant="outline">
              <Camera className="w-4 h-4 mr-2" />
              Capture Retinal Image
            </Button>
            <Button className="w-full justify-start" variant="outline">
              <Scan className="w-4 h-4 mr-2" />
              Start OCT Scan
            </Button>
            <Button className="w-full justify-start" variant="outline">
              <Eye className="w-4 h-4 mr-2" />
              Visual Field Assessment
            </Button>
            <Button className="w-full justify-start" variant="outline">
              <Calendar className="w-4 h-4 mr-2" />
              Schedule Follow-up
            </Button>
            <Button className="w-full justify-start" variant="outline">
              <FileImage className="w-4 h-4 mr-2" />
              Review Image Archive
            </Button>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

export default Ophthalmology;