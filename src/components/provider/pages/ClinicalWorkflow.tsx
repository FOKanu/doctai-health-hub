import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';
import {
  Clipboard,
  Plus,
  TrendingUp,
  TrendingDown,
  Minus,
  Calendar,
  Clock,
  User,
  Heart,
  Thermometer,
  Activity,
  Droplets
} from 'lucide-react';
import { useProviderStore, type Order, type LabTest, type Prescription, type VitalRecord } from '@/stores/providerStore';
import { useToast } from '@/hooks/use-toast';
import { NewOrderForm } from '../forms/NewOrderForm';
import { NewLabTestForm } from '../forms/NewLabTestForm';
import { NewPrescriptionForm } from '../forms/NewPrescriptionForm';
import { NewVitalRecordForm } from '../forms/NewVitalRecordForm';

function getStatusVariant(status: string) {
  switch (status.toLowerCase()) {
    case 'pending':
      return 'secondary';
    case 'in progress':
      return 'secondary';
    case 'completed':
      return 'default';
    case 'cancelled':
      return 'destructive';
    case 'active':
      return 'default';
    case 'expired':
      return 'destructive';
    case 'pending renewal':
      return 'secondary';
    case 'urgent':
      return 'destructive';
    default:
      return 'outline';
  }
}

function getPriorityVariant(priority: string) {
  switch (priority.toLowerCase()) {
    case 'urgent':
      return 'destructive';
    case 'high':
      return 'destructive';
    case 'medium':
      return 'secondary';
    case 'low':
      return 'outline';
    default:
      return 'outline';
  }
}

function ClinicalWorkflow() {
  const { toast } = useToast();
  const {
    orders,
    labTests,
    prescriptions,
    vitalRecords,
    getPendingOrders,
    getPendingLabs,
    getActivePrescriptions,
    getRecentVitals
  } = useProviderStore();

  const [activeTab, setActiveTab] = useState('orders');

  const pendingOrders = getPendingOrders();
  const pendingLabs = getPendingLabs();
  const activePrescriptions = getActivePrescriptions();
  const recentVitals = getRecentVitals();

  const handleOrderCreated = () => {
    toast({
      title: "Order Created",
      description: "New order has been successfully created.",
    });
  };

  const handleLabTestCreated = () => {
    toast({
      title: "Lab Test Ordered",
      description: "New lab test has been successfully ordered.",
    });
  };

  const handlePrescriptionCreated = () => {
    toast({
      title: "Prescription Created",
      description: "New prescription has been successfully created.",
    });
  };

  const handleVitalRecordCreated = () => {
    toast({
      title: "Vital Record Added",
      description: "New vital record has been successfully added.",
    });
  };

  const getTrendIcon = (value: number, normal: number) => {
    if (value > normal * 1.1) return <TrendingUp className="w-4 h-4 text-red-500" />;
    if (value < normal * 0.9) return <TrendingDown className="w-4 h-4 text-blue-500" />;
    return <Minus className="w-4 h-4 text-green-500" />;
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-foreground">Clinical Workflow</h1>
          <p className="text-muted-foreground mt-1">Manage orders, labs, prescriptions, and vitals</p>
        </div>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="orders">Orders</TabsTrigger>
          <TabsTrigger value="labs">Labs</TabsTrigger>
          <TabsTrigger value="prescriptions">Prescriptions</TabsTrigger>
          <TabsTrigger value="vitals">Vitals</TabsTrigger>
        </TabsList>

        {/* Orders Section */}
        <TabsContent value="orders" className="space-y-6">
          <div className="grid gap-6 md:grid-cols-4">
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Pending Orders</CardTitle>
                <Clipboard className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{pendingOrders.length}</div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Total Orders</CardTitle>
                <Clipboard className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{orders.length}</div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Lab Orders</CardTitle>
                <Activity className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{orders.filter(o => o.type === 'Lab').length}</div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Imaging Orders</CardTitle>
                <Activity className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{orders.filter(o => o.type === 'Imaging').length}</div>
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle>All Orders</CardTitle>
                <Dialog>
                  <DialogTrigger asChild>
                    <Button size="sm">
                      <Plus className="w-4 h-4 mr-2" />
                      New Order
                    </Button>
                  </DialogTrigger>
                  <DialogContent className="max-w-2xl">
                    <DialogHeader>
                      <DialogTitle>Create New Order</DialogTitle>
                      <DialogDescription>Add a new medical order for a patient.</DialogDescription>
                    </DialogHeader>
                    <NewOrderForm onSuccess={handleOrderCreated} />
                  </DialogContent>
                </Dialog>
              </div>
            </CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Order ID</TableHead>
                    <TableHead>Patient</TableHead>
                    <TableHead>Type</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead>Ordered Date</TableHead>
                    <TableHead>Ordered By</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {orders.map((order) => (
                    <TableRow key={order.id}>
                      <TableCell className="font-medium">{order.id}</TableCell>
                      <TableCell>{order.patientName}</TableCell>
                      <TableCell>{order.type}</TableCell>
                      <TableCell>
                        <Badge variant={getStatusVariant(order.status)}>{order.status}</Badge>
                      </TableCell>
                      <TableCell>{order.orderedDate}</TableCell>
                      <TableCell>{order.orderedBy}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Labs Section */}
        <TabsContent value="labs" className="space-y-6">
          <div className="grid gap-6 md:grid-cols-4">
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Labs Pending</CardTitle>
                <Activity className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{pendingLabs.length}</div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Total Lab Tests</CardTitle>
                <Activity className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{labTests.length}</div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Blood Tests</CardTitle>
                <Droplets className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{labTests.filter(l => l.testType === 'Blood').length}</div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Urgent Tests</CardTitle>
                <Activity className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{labTests.filter(l => l.priority === 'Urgent').length}</div>
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle>Lab Tests</CardTitle>
                <Dialog>
                  <DialogTrigger asChild>
                    <Button size="sm">
                      <Plus className="w-4 h-4 mr-2" />
                      Order Lab Test
                    </Button>
                  </DialogTrigger>
                  <DialogContent className="max-w-2xl">
                    <DialogHeader>
                      <DialogTitle>Order New Lab Test</DialogTitle>
                      <DialogDescription>Order a new lab test for a patient.</DialogDescription>
                    </DialogHeader>
                    <NewLabTestForm onSuccess={handleLabTestCreated} />
                  </DialogContent>
                </Dialog>
              </div>
            </CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Test ID</TableHead>
                    <TableHead>Patient</TableHead>
                    <TableHead>Test Name</TableHead>
                    <TableHead>Type</TableHead>
                    <TableHead>Priority</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead>Ordered Date</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {labTests.map((lab) => (
                    <TableRow key={lab.id}>
                      <TableCell className="font-medium">{lab.id}</TableCell>
                      <TableCell>
                        <Button variant="link" className="p-0 h-auto font-normal">
                          {lab.patientName}
                        </Button>
                      </TableCell>
                      <TableCell>{lab.testName}</TableCell>
                      <TableCell>{lab.testType}</TableCell>
                      <TableCell>
                        <Badge variant={getPriorityVariant(lab.priority)}>{lab.priority}</Badge>
                      </TableCell>
                      <TableCell>
                        <Badge variant={getStatusVariant(lab.status)}>{lab.status}</Badge>
                      </TableCell>
                      <TableCell>{lab.orderedDate}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Prescriptions Section */}
        <TabsContent value="prescriptions" className="space-y-6">
          <div className="grid gap-6 md:grid-cols-4">
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Active Prescriptions</CardTitle>
                <Clipboard className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{activePrescriptions.length}</div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Total Prescriptions</CardTitle>
                <Clipboard className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{prescriptions.length}</div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Pending Renewals</CardTitle>
                <Clock className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{prescriptions.filter(p => p.status === 'Pending Renewal').length}</div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Expired</CardTitle>
                <Calendar className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{prescriptions.filter(p => p.status === 'Expired').length}</div>
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle>Prescriptions</CardTitle>
                <Dialog>
                  <DialogTrigger asChild>
                    <Button size="sm">
                      <Plus className="w-4 h-4 mr-2" />
                      New Prescription
                    </Button>
                  </DialogTrigger>
                  <DialogContent className="max-w-2xl">
                    <DialogHeader>
                      <DialogTitle>Create New Prescription</DialogTitle>
                      <DialogDescription>Create a new prescription for a patient.</DialogDescription>
                    </DialogHeader>
                    <NewPrescriptionForm onSuccess={handlePrescriptionCreated} />
                  </DialogContent>
                </Dialog>
              </div>
            </CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Prescription ID</TableHead>
                    <TableHead>Patient</TableHead>
                    <TableHead>Medication</TableHead>
                    <TableHead>Dosage</TableHead>
                    <TableHead>Frequency</TableHead>
                    <TableHead>Refills</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead>Renewal Date</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {prescriptions.map((prescription) => (
                    <TableRow key={prescription.id}>
                      <TableCell className="font-medium">{prescription.id}</TableCell>
                      <TableCell>{prescription.patientName}</TableCell>
                      <TableCell>{prescription.medicationName}</TableCell>
                      <TableCell>{prescription.dosage}</TableCell>
                      <TableCell>{prescription.frequency}</TableCell>
                      <TableCell>{prescription.refillsRemaining}</TableCell>
                      <TableCell>
                        <Badge variant={getStatusVariant(prescription.status)}>{prescription.status}</Badge>
                      </TableCell>
                      <TableCell>{prescription.renewalDate}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Vitals Section */}
        <TabsContent value="vitals" className="space-y-6">
          <div className="grid gap-6 md:grid-cols-4">
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Recent Vitals</CardTitle>
                <Heart className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{recentVitals.length}</div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Total Records</CardTitle>
                <Activity className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{vitalRecords.length}</div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Avg BP (Systolic)</CardTitle>
                <Heart className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {Math.round(vitalRecords.reduce((sum, v) => sum + (v.bloodPressureSystolic || 0), 0) / vitalRecords.length)}
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Avg Heart Rate</CardTitle>
                <Heart className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {Math.round(vitalRecords.reduce((sum, v) => sum + (v.heartRate || 0), 0) / vitalRecords.length)}
                </div>
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle>Patient Vitals</CardTitle>
                <Dialog>
                  <DialogTrigger asChild>
                    <Button size="sm">
                      <Plus className="w-4 h-4 mr-2" />
                      Record Vitals
                    </Button>
                  </DialogTrigger>
                  <DialogContent className="max-w-2xl">
                    <DialogHeader>
                      <DialogTitle>Record Patient Vitals</DialogTitle>
                      <DialogDescription>Record new vital signs for a patient.</DialogDescription>
                    </DialogHeader>
                    <NewVitalRecordForm onSuccess={handleVitalRecordCreated} />
                  </DialogContent>
                </Dialog>
              </div>
            </CardHeader>
            <CardContent>
              <div className="grid gap-4">
                {recentVitals.map((vital) => (
                  <Card key={vital.id}>
                    <CardContent className="pt-6">
                      <div className="flex items-center justify-between mb-2">
                        <h4 className="font-medium">{vital.patientName}</h4>
                        <span className="text-sm text-muted-foreground">{vital.recordedDate}</span>
                      </div>
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                        {vital.bloodPressureSystolic && (
                          <div className="flex items-center justify-between">
                            <span className="text-muted-foreground">BP:</span>
                            <div className="flex items-center gap-1">
                              <span>{vital.bloodPressureSystolic}/{vital.bloodPressureDiastolic}</span>
                              {getTrendIcon(vital.bloodPressureSystolic, 120)}
                            </div>
                          </div>
                        )}
                        {vital.heartRate && (
                          <div className="flex items-center justify-between">
                            <span className="text-muted-foreground">HR:</span>
                            <div className="flex items-center gap-1">
                              <span>{vital.heartRate} bpm</span>
                              {getTrendIcon(vital.heartRate, 70)}
                            </div>
                          </div>
                        )}
                        {vital.temperature && (
                          <div className="flex items-center justify-between">
                            <span className="text-muted-foreground">Temp:</span>
                            <div className="flex items-center gap-1">
                              <span>{vital.temperature}°F</span>
                              {getTrendIcon(vital.temperature, 98.6)}
                            </div>
                          </div>
                        )}
                        {vital.oxygenSaturation && (
                          <div className="flex items-center justify-between">
                            <span className="text-muted-foreground">SpO₂:</span>
                            <div className="flex items-center gap-1">
                              <span>{vital.oxygenSaturation}%</span>
                              {getTrendIcon(vital.oxygenSaturation, 98)}
                            </div>
                          </div>
                        )}
                      </div>
                      <div className="mt-2 text-xs text-muted-foreground">
                        Recorded by {vital.recordedBy}
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}

export default ClinicalWorkflow;