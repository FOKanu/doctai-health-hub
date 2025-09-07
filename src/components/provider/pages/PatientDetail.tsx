import React, { useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { 
  User, 
  ArrowLeft, 
  Phone, 
  Mail, 
  Calendar,
  MapPin,
  Activity,
  FileText,
  Pill,
  Camera,
  Clock,
  AlertTriangle,
  TrendingUp,
  Heart
} from 'lucide-react';

// Mock patient data - in real app, fetch based on ID
const mockPatient = {
  id: 'P001',
  name: 'Sarah Johnson',
  mrn: 'MRN-789012',
  age: 34,
  gender: 'Female',
  phone: '(555) 123-4567',
  email: 'sarah.j@email.com',
  address: '123 Oak Street, Springfield, IL 62701',
  insurance: 'BlueCross BlueShield PPO',
  primaryProvider: 'Dr. Michael Chen',
  riskLevel: 'Medium',
  lastVisit: '2024-01-15',
  nextAppointment: '2024-02-20',
  allergies: ['Penicillin', 'Shellfish'],
  conditions: ['Hypertension', 'Type 2 Diabetes'],
  vitals: {
    bloodPressure: '128/82',
    heartRate: '72 bpm',
    temperature: '98.6°F',
    weight: '165 lbs',
    height: '5\'6"'
  }
};

const mockVisits = [
  {
    id: 1,
    date: '2024-01-15',
    type: 'Annual Physical',
    provider: 'Dr. Michael Chen',
    duration: '45 mins',
    notes: 'Patient reports feeling well. Blood pressure slightly elevated.',
    diagnosis: 'Routine examination'
  },
  {
    id: 2,
    date: '2023-11-08',
    type: 'Follow-up',
    provider: 'Dr. Sarah Miller',
    duration: '30 mins',
    notes: 'Diabetes management discussion. A1C improved.',
    diagnosis: 'Type 2 Diabetes - stable'
  }
];

const mockLabs = [
  {
    id: 1,
    date: '2024-01-15',
    test: 'Complete Blood Count',
    status: 'Normal',
    results: 'WBC: 6.2, RBC: 4.8, HGB: 14.2',
    provider: 'Dr. Michael Chen'
  },
  {
    id: 2,
    date: '2024-01-15',
    test: 'HbA1c',
    status: 'Elevated',
    results: '7.2% (Target: <7.0%)',
    provider: 'Dr. Michael Chen'
  }
];

const mockMedications = [
  {
    id: 1,
    name: 'Metformin',
    dosage: '500mg',
    frequency: 'Twice daily',
    prescribed: '2023-06-15',
    prescriber: 'Dr. Michael Chen'
  },
  {
    id: 2,
    name: 'Lisinopril',
    dosage: '10mg',
    frequency: 'Once daily',
    prescribed: '2023-08-20',
    prescriber: 'Dr. Sarah Miller'
  }
];

function PatientDetail() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState('overview');

  const getRiskBadgeColor = (risk: string) => {
    switch (risk.toLowerCase()) {
      case 'high': return 'destructive';
      case 'medium': return 'default';
      case 'low': return 'secondary';
      default: return 'default';
    }
  };

  return (
    <div className="space-y-6">
      {/* Sticky Header */}
      <div className="sticky top-0 z-10 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 border-b pb-4">
        <div className="flex items-center justify-between mb-4">
          <Button 
            variant="ghost" 
            onClick={() => navigate('/provider/patients')}
            className="flex items-center space-x-2"
          >
            <ArrowLeft className="w-4 h-4" />
            <span>Back to Patients</span>
          </Button>
          
          <div className="flex items-center space-x-2">
            <Button variant="outline">
              <Phone className="w-4 h-4 mr-2" />
              Call Patient
            </Button>
            <Button>
              <Calendar className="w-4 h-4 mr-2" />
              Schedule Visit
            </Button>
          </div>
        </div>

        {/* Patient Info Header */}
        <div className="flex items-start justify-between">
          <div className="flex items-start space-x-4">
            <div className="w-16 h-16 bg-primary/10 rounded-full flex items-center justify-center">
              <User className="w-8 h-8 text-primary" />
            </div>
            
            <div>
              <h1 className="text-2xl font-bold text-foreground">{mockPatient.name}</h1>
              <div className="flex items-center space-x-4 text-sm text-muted-foreground mt-1">
                <span>MRN: {mockPatient.mrn}</span>
                <span>Age: {mockPatient.age}</span>
                <span>Gender: {mockPatient.gender}</span>
              </div>
              <div className="flex items-center space-x-2 mt-2">
                <Badge variant={getRiskBadgeColor(mockPatient.riskLevel)}>
                  {mockPatient.riskLevel} Risk
                </Badge>
                <Badge variant="outline">
                  Last Visit: {mockPatient.lastVisit}
                </Badge>
              </div>
            </div>
          </div>

          {/* Quick Stats */}
          <div className="grid grid-cols-3 gap-4 text-center">
            <div>
              <div className="text-lg font-semibold">{mockPatient.vitals.bloodPressure}</div>
              <div className="text-xs text-muted-foreground">Blood Pressure</div>
            </div>
            <div>
              <div className="text-lg font-semibold">{mockPatient.vitals.heartRate}</div>
              <div className="text-xs text-muted-foreground">Heart Rate</div>
            </div>
            <div>
              <div className="text-lg font-semibold">{mockPatient.vitals.weight}</div>
              <div className="text-xs text-muted-foreground">Weight</div>
            </div>
          </div>
        </div>
      </div>

      {/* Tabbed Content */}
      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
        <TabsList className="grid w-full grid-cols-6">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="visits">Visits</TabsTrigger>
          <TabsTrigger value="labs">Labs</TabsTrigger>
          <TabsTrigger value="imaging">Imaging</TabsTrigger>
          <TabsTrigger value="medications">Medications</TabsTrigger>
          <TabsTrigger value="notes">Notes</TabsTrigger>
        </TabsList>

        {/* Overview Tab */}
        <TabsContent value="overview" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Contact & Demographics */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <User className="w-5 h-5" />
                  <span>Contact & Demographics</span>
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="flex items-center space-x-2">
                  <Phone className="w-4 h-4 text-muted-foreground" />
                  <span className="text-sm">{mockPatient.phone}</span>
                </div>
                <div className="flex items-center space-x-2">
                  <Mail className="w-4 h-4 text-muted-foreground" />
                  <span className="text-sm">{mockPatient.email}</span>
                </div>
                <div className="flex items-start space-x-2">
                  <MapPin className="w-4 h-4 text-muted-foreground mt-0.5" />
                  <span className="text-sm">{mockPatient.address}</span>
                </div>
                <div className="pt-2 border-t">
                  <div className="text-sm font-medium">Insurance</div>
                  <div className="text-sm text-muted-foreground">{mockPatient.insurance}</div>
                </div>
              </CardContent>
            </Card>

            {/* Current Conditions */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Activity className="w-5 h-5" />
                  <span>Current Conditions</span>
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                {mockPatient.conditions.map((condition, index) => (
                  <div key={index} className="flex items-center justify-between p-2 bg-muted/50 rounded">
                    <span className="text-sm font-medium">{condition}</span>
                    <Badge variant="secondary">Active</Badge>
                  </div>
                ))}
                
                <div className="pt-2 border-t">
                  <div className="text-sm font-medium mb-2">Allergies</div>
                  <div className="flex flex-wrap gap-1">
                    {mockPatient.allergies.map((allergy, index) => (
                      <Badge key={index} variant="destructive" className="text-xs">
                        <AlertTriangle className="w-3 h-3 mr-1" />
                        {allergy}
                      </Badge>
                    ))}
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Vital Signs */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Heart className="w-5 h-5" />
                  <span>Latest Vitals</span>
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="grid grid-cols-2 gap-3">
                  <div className="text-center p-2 bg-muted/50 rounded">
                    <div className="text-sm font-medium">{mockPatient.vitals.bloodPressure}</div>
                    <div className="text-xs text-muted-foreground">BP (mmHg)</div>
                  </div>
                  <div className="text-center p-2 bg-muted/50 rounded">
                    <div className="text-sm font-medium">{mockPatient.vitals.heartRate}</div>
                    <div className="text-xs text-muted-foreground">Heart Rate</div>
                  </div>
                  <div className="text-center p-2 bg-muted/50 rounded">
                    <div className="text-sm font-medium">{mockPatient.vitals.temperature}</div>
                    <div className="text-xs text-muted-foreground">Temperature</div>
                  </div>
                  <div className="text-center p-2 bg-muted/50 rounded">
                    <div className="text-sm font-medium">{mockPatient.vitals.height}</div>
                    <div className="text-xs text-muted-foreground">Height</div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Visits Tab */}
        <TabsContent value="visits" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                <span className="flex items-center space-x-2">
                  <Calendar className="w-5 h-5" />
                  <span>Visit History</span>
                </span>
                <Button size="sm">
                  <Calendar className="w-4 h-4 mr-2" />
                  New Visit
                </Button>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {mockVisits.map((visit) => (
                  <div key={visit.id} className="border rounded-lg p-4 hover:bg-muted/50 transition-colors">
                    <div className="flex items-start justify-between">
                      <div>
                        <div className="font-medium">{visit.type}</div>
                        <div className="text-sm text-muted-foreground">{visit.date} • {visit.provider}</div>
                        <div className="text-sm mt-2">{visit.notes}</div>
                        <Badge variant="outline" className="mt-2">{visit.diagnosis}</Badge>
                      </div>
                      <div className="text-right text-sm text-muted-foreground">
                        <Clock className="w-4 h-4 inline mr-1" />
                        {visit.duration}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Labs Tab */}
        <TabsContent value="labs" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                <span className="flex items-center space-x-2">
                  <FileText className="w-5 h-5" />
                  <span>Laboratory Results</span>
                </span>
                <Button size="sm">
                  <FileText className="w-4 h-4 mr-2" />
                  Order Labs
                </Button>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {mockLabs.map((lab) => (
                  <div key={lab.id} className="border rounded-lg p-4">
                    <div className="flex items-start justify-between">
                      <div>
                        <div className="font-medium">{lab.test}</div>
                        <div className="text-sm text-muted-foreground">{lab.date} • {lab.provider}</div>
                        <div className="text-sm mt-2">{lab.results}</div>
                      </div>
                      <Badge variant={lab.status === 'Normal' ? 'secondary' : 'destructive'}>
                        {lab.status}
                      </Badge>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Imaging Tab */}
        <TabsContent value="imaging" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                <span className="flex items-center space-x-2">
                  <Camera className="w-5 h-5" />
                  <span>Imaging Studies</span>
                </span>
                <Button size="sm">
                  <Camera className="w-4 h-4 mr-2" />
                  Order Imaging
                </Button>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-center py-12">
                <Camera className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
                <h3 className="text-lg font-medium text-foreground mb-2">No Imaging Studies</h3>
                <p className="text-muted-foreground">No imaging studies have been performed for this patient.</p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Medications Tab */}
        <TabsContent value="medications" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                <span className="flex items-center space-x-2">
                  <Pill className="w-5 h-5" />
                  <span>Current Medications</span>
                </span>
                <Button size="sm">
                  <Pill className="w-4 h-4 mr-2" />
                  Prescribe
                </Button>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {mockMedications.map((med) => (
                  <div key={med.id} className="border rounded-lg p-4">
                    <div className="flex items-start justify-between">
                      <div>
                        <div className="font-medium">{med.name}</div>
                        <div className="text-sm text-muted-foreground">{med.dosage} • {med.frequency}</div>
                        <div className="text-sm mt-2">Prescribed: {med.prescribed} by {med.prescriber}</div>
                      </div>
                      <Badge variant="secondary">Active</Badge>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Notes Tab */}
        <TabsContent value="notes" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                <span className="flex items-center space-x-2">
                  <FileText className="w-5 h-5" />
                  <span>Clinical Notes</span>
                </span>
                <Button size="sm">
                  <FileText className="w-4 h-4 mr-2" />
                  Add Note
                </Button>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-center py-12">
                <FileText className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
                <h3 className="text-lg font-medium text-foreground mb-2">No Clinical Notes</h3>
                <p className="text-muted-foreground">No clinical notes have been added for this patient.</p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}

export default PatientDetail;