import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { 
  Shield, 
  FileText, 
  Eye, 
  Activity, 
  Download, 
  Clock, 
  User, 
  CheckCircle,
  AlertTriangle,
  Filter,
  Calendar
} from 'lucide-react';

// Mock data for audit events
const mockAuditEvents = [
  {
    id: '1',
    timestamp: '2024-03-15T14:30:00Z',
    action: 'AI_APPROVAL',
    user: 'Dr. Sarah Weber',
    resource: 'Patient Diagnosis Review',
    status: 'approved',
    details: 'AI diagnosis for melanoma risk assessment approved',
    ipAddress: '192.168.1.100'
  },
  {
    id: '2',
    timestamp: '2024-03-15T13:45:00Z',
    action: 'RECORD_VIEW',
    user: 'Dr. Michael Brown',
    resource: 'Patient ID: 12345',
    status: 'success',
    details: 'Viewed patient medical history',
    ipAddress: '192.168.1.101'
  },
  {
    id: '3',
    timestamp: '2024-03-15T12:20:00Z',
    action: 'DATA_EXPORT',
    user: 'System Admin',
    resource: 'Compliance Report Q1',
    status: 'completed',
    details: 'Quarterly compliance data exported',
    ipAddress: '192.168.1.102'
  },
  {
    id: '4',
    timestamp: '2024-03-15T11:15:00Z',
    action: 'AI_APPROVAL',
    user: 'Dr. Lisa Chen',
    resource: 'ECG Analysis',
    status: 'rejected',
    details: 'AI ECG analysis rejected - manual review required',
    ipAddress: '192.168.1.103'
  }
];

const mockAccessLogs = [
  {
    id: '1',
    timestamp: '2024-03-15T15:00:00Z',
    user: 'Dr. Sarah Weber',
    action: 'LOGIN',
    resource: 'Provider Portal',
    duration: '2h 15m',
    ipAddress: '192.168.1.100'
  },
  {
    id: '2',
    timestamp: '2024-03-15T14:45:00Z',
    user: 'Dr. Michael Brown',
    action: 'PATIENT_ACCESS',
    resource: 'Patient Records',
    duration: '45m',
    ipAddress: '192.168.1.101'
  }
];

export function ComplianceCenter() {
  const [selectedTab, setSelectedTab] = useState('policies');

  const handleExportCSV = () => {
    const csvContent = [
      ['Timestamp', 'Action', 'User', 'Resource', 'Status', 'IP Address'],
      ...mockAuditEvents.map(event => [
        event.timestamp,
        event.action,
        event.user,
        event.resource,
        event.status,
        event.ipAddress
      ])
    ].map(row => row.join(',')).join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `compliance-audit-${new Date().toISOString().split('T')[0]}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'approved':
      case 'success':
      case 'completed':
        return <CheckCircle className="w-4 h-4 text-green-600" />;
      case 'rejected':
      case 'failed':
        return <AlertTriangle className="w-4 h-4 text-red-600" />;
      default:
        return <Clock className="w-4 h-4 text-yellow-600" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'approved':
      case 'success':
      case 'completed':
        return 'bg-green-100 text-green-800 border-green-200';
      case 'rejected':
      case 'failed':
        return 'bg-red-100 text-red-800 border-red-200';
      default:
        return 'bg-yellow-100 text-yellow-800 border-yellow-200';
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Compliance Center</h1>
          <p className="text-gray-600 mt-1">HIPAA compliance and audit logs</p>
        </div>
        <Button onClick={handleExportCSV} className="bg-blue-600 hover:bg-blue-700">
          <Download className="w-4 h-4 mr-2" />
          Export CSV
        </Button>
      </div>

      {/* Statistics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-2xl font-bold text-blue-600">147</div>
                <div className="text-sm text-gray-600">Total Audits</div>
              </div>
              <Activity className="w-8 h-8 text-blue-600 opacity-75" />
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-2xl font-bold text-green-600">98%</div>
                <div className="text-sm text-gray-600">Compliance Rate</div>
              </div>
              <CheckCircle className="w-8 h-8 text-green-600 opacity-75" />
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-2xl font-bold text-orange-600">23</div>
                <div className="text-sm text-gray-600">AI Approvals</div>
              </div>
              <Shield className="w-8 h-8 text-orange-600 opacity-75" />
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-2xl font-bold text-purple-600">12</div>
                <div className="text-sm text-gray-600">Data Exports</div>
              </div>
              <Download className="w-8 h-8 text-purple-600 opacity-75" />
            </div>
          </CardContent>
        </Card>
      </div>

      <Tabs value={selectedTab} onValueChange={setSelectedTab} className="space-y-4">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="policies">HIPAA Policies</TabsTrigger>
          <TabsTrigger value="access">Access Logs</TabsTrigger>
          <TabsTrigger value="audit">Audit Events</TabsTrigger>
          <TabsTrigger value="exports">Data Exports</TabsTrigger>
        </TabsList>

        <TabsContent value="policies" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <FileText className="w-5 h-5" />
                <span>HIPAA Policy Management</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <Alert>
                <Shield className="h-4 w-4" />
                <AlertDescription>
                  All policies are reviewed and updated annually to ensure HIPAA compliance.
                </AlertDescription>
              </Alert>
              
              <div className="space-y-3">
                <div className="flex items-center justify-between p-4 border rounded-lg">
                  <div>
                    <h4 className="font-medium">Patient Privacy Policy</h4>
                    <p className="text-sm text-gray-600">Last updated: March 1, 2024</p>
                  </div>
                  <Badge className="bg-green-100 text-green-800">Current</Badge>
                </div>
                
                <div className="flex items-center justify-between p-4 border rounded-lg">
                  <div>
                    <h4 className="font-medium">Data Security Standards</h4>
                    <p className="text-sm text-gray-600">Last updated: February 15, 2024</p>
                  </div>
                  <Badge className="bg-green-100 text-green-800">Current</Badge>
                </div>

                <div className="flex items-center justify-between p-4 border rounded-lg">
                  <div>
                    <h4 className="font-medium">Access Control Policy</h4>
                    <p className="text-sm text-gray-600">Last updated: January 20, 2024</p>
                  </div>
                  <Badge className="bg-green-100 text-green-800">Current</Badge>
                </div>

                <div className="flex items-center justify-between p-4 border rounded-lg">
                  <div>
                    <h4 className="font-medium">Audit Trail Requirements</h4>
                    <p className="text-sm text-gray-600">Last updated: December 10, 2023</p>
                  </div>
                  <Badge className="bg-yellow-100 text-yellow-800">Review Due</Badge>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="access" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Eye className="w-5 h-5" />
                <span>Access Logs</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {mockAccessLogs.map((log) => (
                  <div key={log.id} className="flex items-center justify-between p-4 border rounded-lg">
                    <div className="flex items-center space-x-3">
                      <User className="w-5 h-5 text-gray-500" />
                      <div>
                        <div className="font-medium">{log.user}</div>
                        <div className="text-sm text-gray-600">{log.action} - {log.resource}</div>
                        <div className="text-xs text-gray-500">
                          {new Date(log.timestamp).toLocaleString()}
                        </div>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-sm font-medium">{log.duration}</div>
                      <div className="text-xs text-gray-500">{log.ipAddress}</div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="audit" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Activity className="w-5 h-5" />
                <span>Audit Events</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {mockAuditEvents.map((event) => (
                  <div key={event.id} className="flex items-start justify-between p-4 border rounded-lg">
                    <div className="flex items-start space-x-3">
                      {getStatusIcon(event.status)}
                      <div>
                        <div className="flex items-center space-x-2">
                          <span className="font-medium">{event.action.replace('_', ' ')}</span>
                          <Badge className={getStatusColor(event.status)}>
                            {event.status.toUpperCase()}
                          </Badge>
                        </div>
                        <div className="text-sm text-gray-600 mt-1">{event.details}</div>
                        <div className="text-xs text-gray-500 mt-1">
                          {event.user} • {new Date(event.timestamp).toLocaleString()} • {event.ipAddress}
                        </div>
                      </div>
                    </div>
                    <div className="text-xs text-gray-500">{event.resource}</div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="exports" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Download className="w-5 h-5" />
                <span>Data Export History</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <Alert>
                <Calendar className="h-4 w-4" />
                <AlertDescription>
                  All data exports are logged and tracked for compliance purposes.
                </AlertDescription>
              </Alert>

              <div className="space-y-3">
                <div className="flex items-center justify-between p-4 border rounded-lg">
                  <div>
                    <h4 className="font-medium">Quarterly Compliance Report Q1 2024</h4>
                    <p className="text-sm text-gray-600">Exported on March 15, 2024 • CSV Format</p>
                  </div>
                  <Button variant="outline" size="sm">
                    <Download className="w-4 h-4 mr-2" />
                    Download
                  </Button>
                </div>

                <div className="flex items-center justify-between p-4 border rounded-lg">
                  <div>
                    <h4 className="font-medium">Audit Trail Export</h4>
                    <p className="text-sm text-gray-600">Exported on March 10, 2024 • JSON Format</p>
                  </div>
                  <Button variant="outline" size="sm">
                    <Download className="w-4 h-4 mr-2" />
                    Download
                  </Button>
                </div>

                <div className="flex items-center justify-between p-4 border rounded-lg">
                  <div>
                    <h4 className="font-medium">User Access Report</h4>
                    <p className="text-sm text-gray-600">Exported on March 5, 2024 • PDF Format</p>
                  </div>
                  <Button variant="outline" size="sm">
                    <Download className="w-4 h-4 mr-2" />
                    Download
                  </Button>
                </div>
              </div>

              <div className="pt-4 border-t">
                <Button onClick={handleExportCSV} className="w-full">
                  <Download className="w-4 h-4 mr-2" />
                  Generate New Export
                </Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}