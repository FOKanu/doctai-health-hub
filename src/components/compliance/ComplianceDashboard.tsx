import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Progress } from '@/components/ui/progress';
import {
  Shield,
  FileText,
  Users,
  Activity,
  AlertTriangle,
  CheckCircle,
  Clock,
  Lock,
  Eye,
  Trash2,
  Download
} from 'lucide-react';
import { hipaaService } from '@/services/compliance/hipaaService';
import { dataRetentionService } from '@/services/compliance/dataRetentionService';
import { accessControlService } from '@/services/compliance/accessControlService';

interface ComplianceMetrics {
  totalAuditLogs: number;
  recentBreaches: number;
  activeSessions: number;
  pendingAccessRequests: number;
  dataForDisposal: number;
  complianceScore: number;
}

const ComplianceDashboard: React.FC = () => {
  const [metrics, setMetrics] = useState<ComplianceMetrics>({
    totalAuditLogs: 0,
    recentBreaches: 0,
    activeSessions: 0,
    pendingAccessRequests: 0,
    dataForDisposal: 0,
    complianceScore: 0
  });
  const [auditLogs, setAuditLogs] = useState<any[]>([]);
  const [disposalRecords, setDisposalRecords] = useState<any[]>([]);
  const [accessReport, setAccessReport] = useState<any>(null);

  useEffect(() => {
    loadComplianceData();
  }, []);

  const loadComplianceData = () => {
    // Load audit logs
    const logs = hipaaService.getAuditLogs();
    setAuditLogs(logs);

    // Load disposal records
    const disposal = dataRetentionService.getDisposalRecords();
    setDisposalRecords(disposal);

    // Load access control report
    const access = accessControlService.generateAccessReport();
    setAccessReport(access);

    // Calculate metrics
    const recentBreaches = logs.filter(log =>
      log.action.includes('breach') || log.action.includes('unauthorized')
    ).length;

    const dataForDisposal = dataRetentionService.getDataForDisposal().length;

    setMetrics({
      totalAuditLogs: logs.length,
      recentBreaches,
      activeSessions: access.activeSessions,
      pendingAccessRequests: access.pendingRequests,
      dataForDisposal,
      complianceScore: calculateComplianceScore(logs, recentBreaches, access)
    });
  };

  const calculateComplianceScore = (logs: any[], breaches: number, access: any): number => {
    let score = 100;

    // Deduct points for breaches
    score -= breaches * 10;

    // Deduct points for expired sessions
    const expiredSessions = access.totalSessions - access.activeSessions;
    score -= expiredSessions * 2;

    // Deduct points for pending requests
    score -= access.pendingRequests * 5;

    return Math.max(0, score);
  };

  const getComplianceStatus = (score: number) => {
    if (score >= 90) return { status: 'Excellent', color: 'bg-green-500', icon: CheckCircle };
    if (score >= 75) return { status: 'Good', color: 'bg-yellow-500', icon: Clock };
    return { status: 'Needs Attention', color: 'bg-red-500', icon: AlertTriangle };
  };

  const complianceStatus = getComplianceStatus(metrics.complianceScore);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">HIPAA Compliance Dashboard</h1>
          <p className="text-muted-foreground">
            Monitor and manage healthcare compliance requirements
          </p>
        </div>
        <Button onClick={loadComplianceData}>
          <Activity className="mr-2 h-4 w-4" />
          Refresh Data
        </Button>
      </div>

      {/* Compliance Score */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Shield className="h-5 w-5" />
            Overall Compliance Score
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center gap-4">
            <div className="flex-1">
              <div className="flex items-center gap-2 mb-2">
                <Badge className={complianceStatus.color}>
                  <complianceStatus.icon className="h-3 w-3 mr-1" />
                  {complianceStatus.status}
                </Badge>
                <span className="text-2xl font-bold">{metrics.complianceScore}%</span>
              </div>
              <Progress value={metrics.complianceScore} className="h-2" />
            </div>
            <div className="text-right text-sm text-muted-foreground">
              <div>Audit Logs: {metrics.totalAuditLogs}</div>
              <div>Active Sessions: {metrics.activeSessions}</div>
              <div>Pending Requests: {metrics.pendingAccessRequests}</div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Audit Logs</CardTitle>
            <FileText className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{metrics.totalAuditLogs}</div>
            <p className="text-xs text-muted-foreground">
              Total activity records
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Security Breaches</CardTitle>
            <AlertTriangle className="h-4 w-4 text-red-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-red-600">{metrics.recentBreaches}</div>
            <p className="text-xs text-muted-foreground">
              Recent security incidents
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Active Sessions</CardTitle>
            <Users className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{metrics.activeSessions}</div>
            <p className="text-xs text-muted-foreground">
              Current user sessions
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Data for Disposal</CardTitle>
            <Trash2 className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{metrics.dataForDisposal}</div>
            <p className="text-xs text-muted-foreground">
              Records ready for disposal
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Detailed Views */}
      <Tabs defaultValue="audit" className="space-y-4">
        <TabsList>
          <TabsTrigger value="audit">Audit Logs</TabsTrigger>
          <TabsTrigger value="access">Access Control</TabsTrigger>
          <TabsTrigger value="retention">Data Retention</TabsTrigger>
          <TabsTrigger value="breaches">Security Alerts</TabsTrigger>
        </TabsList>

        <TabsContent value="audit" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Recent Audit Logs</CardTitle>
              <CardDescription>
                Monitor all system activities for compliance
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Timestamp</TableHead>
                    <TableHead>User</TableHead>
                    <TableHead>Action</TableHead>
                    <TableHead>Resource</TableHead>
                    <TableHead>Status</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {auditLogs.slice(0, 10).map((log) => (
                    <TableRow key={log.id}>
                      <TableCell>{new Date(log.timestamp).toLocaleString()}</TableCell>
                      <TableCell>{log.userId}</TableCell>
                      <TableCell>{log.action}</TableCell>
                      <TableCell>{log.resource}</TableCell>
                      <TableCell>
                        <Badge variant={log.success ? "default" : "destructive"}>
                          {log.success ? "Success" : "Failed"}
                        </Badge>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="access" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Access Control Overview</CardTitle>
              <CardDescription>
                Monitor user sessions and access requests
              </CardDescription>
            </CardHeader>
            <CardContent>
              {accessReport && (
                <div className="space-y-4">
                  <div className="grid grid-cols-3 gap-4">
                    <div className="text-center">
                      <div className="text-2xl font-bold">{accessReport.totalSessions}</div>
                      <div className="text-sm text-muted-foreground">Total Sessions</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold">{accessReport.activeSessions}</div>
                      <div className="text-sm text-muted-foreground">Active Sessions</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold">{accessReport.pendingRequests}</div>
                      <div className="text-sm text-muted-foreground">Pending Requests</div>
                    </div>
                  </div>

                  <div className="space-y-2">
                    <h4 className="font-semibold">User Roles</h4>
                    <div className="grid grid-cols-2 gap-2">
                      {accessReport.roles.map((role: any) => (
                        <div key={role.id} className="flex items-center justify-between p-2 border rounded">
                          <span>{role.name}</span>
                          <Badge variant={role.hipaaCompliant ? "default" : "destructive"}>
                            {role.hipaaCompliant ? "Compliant" : "Non-Compliant"}
                          </Badge>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="retention" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Data Retention Management</CardTitle>
              <CardDescription>
                Monitor data disposal and retention policies
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <Alert>
                  <AlertTriangle className="h-4 w-4" />
                  <AlertDescription>
                    {metrics.dataForDisposal} records are ready for disposal according to retention policies.
                  </AlertDescription>
                </Alert>

                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Resource Type</TableHead>
                      <TableHead>Disposal Method</TableHead>
                      <TableHead>Disposed By</TableHead>
                      <TableHead>Date</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {disposalRecords.slice(0, 5).map((record) => (
                      <TableRow key={record.id}>
                        <TableCell>{record.resourceType}</TableCell>
                        <TableCell>{record.disposalMethod}</TableCell>
                        <TableCell>{record.disposedBy}</TableCell>
                        <TableCell>{new Date(record.disposalDate).toLocaleDateString()}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="breaches" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Security Alerts</CardTitle>
              <CardDescription>
                Monitor potential security breaches and suspicious activities
              </CardDescription>
            </CardHeader>
            <CardContent>
              {metrics.recentBreaches > 0 ? (
                <Alert variant="destructive">
                  <AlertTriangle className="h-4 w-4" />
                  <AlertDescription>
                    {metrics.recentBreaches} security incidents detected. Review audit logs for details.
                  </AlertDescription>
                </Alert>
              ) : (
                <Alert>
                  <CheckCircle className="h-4 w-4" />
                  <AlertDescription>
                    No security breaches detected in the recent audit period.
                  </AlertDescription>
                </Alert>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default ComplianceDashboard;
