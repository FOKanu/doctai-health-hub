import React, { useState, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import { 
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { 
  Users, 
  Search, 
  Filter, 
  Plus, 
  ChevronLeft, 
  ChevronRight,
  MessageSquare,
  Calendar,
  FileText
} from 'lucide-react';
import { NewPatientModal } from '@/components/provider/modals/NewPatientModal';

function PatientManagement() {
  const navigate = useNavigate();
  const [searchQuery, setSearchQuery] = useState('');
  const [riskFilter, setRiskFilter] = useState('all');
  const [statusFilter, setStatusFilter] = useState('all');
  const [currentPage, setCurrentPage] = useState(1);
  const [isNewPatientModalOpen, setIsNewPatientModalOpen] = useState(false);
  const patientsPerPage = 10;

  // Mock patient data (25 patients as specified)
  const allPatients = [
    { id: '1', name: 'Sarah Johnson', age: 34, sex: 'F', lastVisit: '2024-03-15', risk: 'Low', status: 'Follow-up', primarySpecialty: 'Cardiology', mrn: 'MRN001234', phone: '(555) 123-4567', email: 'sarah.j@email.com' },
    { id: '2', name: 'Michael Chen', age: 52, sex: 'M', lastVisit: '2024-03-10', risk: 'Medium', status: 'New Patient', primarySpecialty: 'Neurology', mrn: 'MRN001235', phone: '(555) 234-5678', email: 'michael.c@email.com' },
    { id: '3', name: 'Emily Rodriguez', age: 28, sex: 'F', lastVisit: '2024-03-12', risk: 'Low', status: 'Routine', primarySpecialty: 'Cardiology', mrn: 'MRN001236', phone: '(555) 345-6789', email: 'emily.r@email.com' },
    { id: '4', name: 'David Thompson', age: 45, sex: 'M', lastVisit: '2024-03-08', risk: 'High', status: 'Follow-up', primarySpecialty: 'Orthopedics', mrn: 'MRN001237', phone: '(555) 456-7890', email: 'david.t@email.com' },
    { id: '5', name: 'Lisa Williams', age: 39, sex: 'F', lastVisit: '2024-03-14', risk: 'Medium', status: 'New Patient', primarySpecialty: 'Ophthalmology', mrn: 'MRN001238', phone: '(555) 567-8901', email: 'lisa.w@email.com' },
    { id: '6', name: 'James Anderson', age: 62, sex: 'M', lastVisit: '2024-03-07', risk: 'High', status: 'Routine', primarySpecialty: 'Cardiology', mrn: 'MRN001239', phone: '(555) 678-9012', email: 'james.a@email.com' },
    { id: '7', name: 'Maria Garcia', age: 31, sex: 'F', lastVisit: '2024-03-13', risk: 'Low', status: 'Follow-up', primarySpecialty: 'Neurology', mrn: 'MRN001240', phone: '(555) 789-0123', email: 'maria.g@email.com' },
    { id: '8', name: 'Robert Brown', age: 58, sex: 'M', lastVisit: '2024-03-09', risk: 'Medium', status: 'New Patient', primarySpecialty: 'Orthopedics', mrn: 'MRN001241', phone: '(555) 890-1234', email: 'robert.b@email.com' },
    { id: '9', name: 'Jennifer Davis', age: 41, sex: 'F', lastVisit: '2024-03-11', risk: 'Low', status: 'Routine', primarySpecialty: 'Ophthalmology', mrn: 'MRN001242', phone: '(555) 901-2345', email: 'jennifer.d@email.com' },
    { id: '10', name: 'Christopher Lee', age: 36, sex: 'M', lastVisit: '2024-03-06', risk: 'High', status: 'Follow-up', primarySpecialty: 'Cardiology', mrn: 'MRN001243', phone: '(555) 012-3456', email: 'chris.l@email.com' },
    { id: '11', name: 'Amanda Wilson', age: 33, sex: 'F', lastVisit: '2024-03-05', risk: 'Medium', status: 'New Patient', primarySpecialty: 'Neurology', mrn: 'MRN001244', phone: '(555) 123-4567', email: 'amanda.w@email.com' },
    { id: '12', name: 'Mark Taylor', age: 47, sex: 'M', lastVisit: '2024-03-04', risk: 'Low', status: 'Routine', primarySpecialty: 'Orthopedics', mrn: 'MRN001245', phone: '(555) 234-5678', email: 'mark.t@email.com' },
    { id: '13', name: 'Susan Miller', age: 54, sex: 'F', lastVisit: '2024-03-03', risk: 'High', status: 'Follow-up', primarySpecialty: 'Ophthalmology', mrn: 'MRN001246', phone: '(555) 345-6789', email: 'susan.m@email.com' },
    { id: '14', name: 'Kevin Moore', age: 29, sex: 'M', lastVisit: '2024-03-02', risk: 'Low', status: 'New Patient', primarySpecialty: 'Cardiology', mrn: 'MRN001247', phone: '(555) 456-7890', email: 'kevin.m@email.com' },
    { id: '15', name: 'Rachel White', age: 42, sex: 'F', lastVisit: '2024-03-01', risk: 'Medium', status: 'Routine', primarySpecialty: 'Neurology', mrn: 'MRN001248', phone: '(555) 567-8901', email: 'rachel.w@email.com' },
    { id: '16', name: 'Daniel Clark', age: 50, sex: 'M', lastVisit: '2024-02-28', risk: 'High', status: 'Follow-up', primarySpecialty: 'Orthopedics', mrn: 'MRN001249', phone: '(555) 678-9012', email: 'daniel.c@email.com' },
    { id: '17', name: 'Nicole Martinez', age: 37, sex: 'F', lastVisit: '2024-02-27', risk: 'Low', status: 'New Patient', primarySpecialty: 'Ophthalmology', mrn: 'MRN001250', phone: '(555) 789-0123', email: 'nicole.m@email.com' },
    { id: '18', name: 'Steven Harris', age: 44, sex: 'M', lastVisit: '2024-02-26', risk: 'Medium', status: 'Routine', primarySpecialty: 'Cardiology', mrn: 'MRN001251', phone: '(555) 890-1234', email: 'steven.h@email.com' },
    { id: '19', name: 'Michelle Lewis', age: 35, sex: 'F', lastVisit: '2024-02-25', risk: 'Low', status: 'Follow-up', primarySpecialty: 'Neurology', mrn: 'MRN001252', phone: '(555) 901-2345', email: 'michelle.l@email.com' },
    { id: '20', name: 'Andrew Young', age: 48, sex: 'M', lastVisit: '2024-02-24', risk: 'High', status: 'New Patient', primarySpecialty: 'Orthopedics', mrn: 'MRN001253', phone: '(555) 012-3456', email: 'andrew.y@email.com' },
    { id: '21', name: 'Stephanie King', age: 32, sex: 'F', lastVisit: '2024-02-23', risk: 'Medium', status: 'Routine', primarySpecialty: 'Ophthalmology', mrn: 'MRN001254', phone: '(555) 123-4567', email: 'stephanie.k@email.com' },
    { id: '22', name: 'Brian Scott', age: 56, sex: 'M', lastVisit: '2024-02-22', risk: 'Low', status: 'Follow-up', primarySpecialty: 'Cardiology', mrn: 'MRN001255', phone: '(555) 234-5678', email: 'brian.s@email.com' },
    { id: '23', name: 'Jessica Adams', age: 40, sex: 'F', lastVisit: '2024-02-21', risk: 'High', status: 'New Patient', primarySpecialty: 'Neurology', mrn: 'MRN001256', phone: '(555) 345-6789', email: 'jessica.a@email.com' },
    { id: '24', name: 'Ryan Turner', age: 43, sex: 'M', lastVisit: '2024-02-20', risk: 'Medium', status: 'Routine', primarySpecialty: 'Orthopedics', mrn: 'MRN001257', phone: '(555) 456-7890', email: 'ryan.t@email.com' },
    { id: '25', name: 'Megan Phillips', age: 38, sex: 'F', lastVisit: '2024-02-19', risk: 'Low', status: 'Follow-up', primarySpecialty: 'Ophthalmology', mrn: 'MRN001258', phone: '(555) 567-8901', email: 'megan.p@email.com' }
  ];

  // Filtering and searching logic
  const filteredPatients = useMemo(() => {
    return allPatients.filter(patient => {
      const matchesSearch = 
        patient.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        patient.mrn.toLowerCase().includes(searchQuery.toLowerCase());
      
      const matchesRisk = riskFilter === 'all' || patient.risk === riskFilter;
      const matchesStatus = statusFilter === 'all' || patient.status === statusFilter;
      
      return matchesSearch && matchesRisk && matchesStatus;
    });
  }, [searchQuery, riskFilter, statusFilter]);

  // Pagination logic
  const totalPages = Math.ceil(filteredPatients.length / patientsPerPage);
  const paginatedPatients = filteredPatients.slice(
    (currentPage - 1) * patientsPerPage,
    currentPage * patientsPerPage
  );

  const handlePatientClick = (patientId: string) => {
    navigate(`/provider/patients/${patientId}`);
  };

  const handleNewPatientSuccess = (newPatient: any) => {
    // In a real app, this would update the patient list from the server
    console.log('New patient created:', newPatient);
    setIsNewPatientModalOpen(false);
    // Navigate to the new patient's detail page
    navigate(`/provider/patients/${newPatient.id || 'new-patient'}`);
  };

  const getRiskColor = (risk: string) => {
    switch (risk.toLowerCase()) {
      case 'high':
        return 'bg-red-100 text-red-800 border-red-200';
      case 'medium':
        return 'bg-yellow-100 text-yellow-800 border-yellow-200';
      case 'low':
        return 'bg-green-100 text-green-800 border-green-200';
      default:
        return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'New Patient':
        return 'bg-blue-100 text-blue-800 border-blue-200';
      case 'Follow-up':
        return 'bg-purple-100 text-purple-800 border-purple-200';
      case 'Routine':
        return 'bg-gray-100 text-gray-800 border-gray-200';
      default:
        return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };

  const getInitials = (name: string) => {
    return name
      .split(' ')
      .map(n => n[0])
      .join('')
      .toUpperCase()
      .slice(0, 2);
  };
  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Patient Management</h1>
          <p className="text-gray-600 mt-1">Manage your patient roster and records</p>
        </div>
        <Button
          onClick={() => setIsNewPatientModalOpen(true)}
          className="bg-blue-600 hover:bg-blue-700 rounded-xl"
        >
          <Plus className="w-4 h-4 mr-2" />
          New Patient
        </Button>
      </div>

      {/* Toolbar */}
      <Card className="card-glass rounded-2xl">
        <CardContent className="p-6">
          <div className="flex flex-col lg:flex-row lg:items-center gap-4">
            {/* Search */}
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
              <Input
                placeholder="Search by name or MRN..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-10 rounded-xl"
              />
            </div>

            {/* Filters */}
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2">
                <Filter className="w-4 h-4 text-gray-500" />
                <span className="text-sm text-gray-600 hidden sm:block">Filters:</span>
              </div>
              
              <Select value={riskFilter} onValueChange={setRiskFilter}>
                <SelectTrigger className="w-32 rounded-xl">
                  <SelectValue placeholder="Risk" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Risk</SelectItem>
                  <SelectItem value="Low">Low</SelectItem>
                  <SelectItem value="Medium">Medium</SelectItem>
                  <SelectItem value="High">High</SelectItem>
                </SelectContent>
              </Select>

              <Select value={statusFilter} onValueChange={setStatusFilter}>
                <SelectTrigger className="w-36 rounded-xl">
                  <SelectValue placeholder="Status" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Status</SelectItem>
                  <SelectItem value="New Patient">New Patient</SelectItem>
                  <SelectItem value="Follow-up">Follow-up</SelectItem>
                  <SelectItem value="Routine">Routine</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>

          {/* Results count */}
          <div className="mt-4 pt-4 border-t border-gray-200">
            <p className="text-sm text-gray-600">
              Showing {paginatedPatients.length} of {filteredPatients.length} patients
            </p>
          </div>
        </CardContent>
      </Card>

      {/* Patient Table */}
      <Card className="card-glass rounded-2xl">
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Users className="w-5 h-5" />
            <span>Patient Roster</span>
          </CardTitle>
        </CardHeader>
        <CardContent className="p-0">
          {/* Desktop Table */}
          <div className="hidden lg:block overflow-x-auto">
            <table className="w-full">
              <thead className="bg-gray-50 border-b">
                <tr>
                  <th className="text-left px-6 py-4 text-sm font-semibold text-gray-900">Patient</th>
                  <th className="text-left px-6 py-4 text-sm font-semibold text-gray-900">Age</th>
                  <th className="text-left px-6 py-4 text-sm font-semibold text-gray-900">Sex</th>
                  <th className="text-left px-6 py-4 text-sm font-semibold text-gray-900">Last Visit</th>
                  <th className="text-left px-6 py-4 text-sm font-semibold text-gray-900">Risk</th>
                  <th className="text-left px-6 py-4 text-sm font-semibold text-gray-900">Status</th>
                  <th className="text-left px-6 py-4 text-sm font-semibold text-gray-900">Primary Specialty</th>
                  <th className="text-left px-6 py-4 text-sm font-semibold text-gray-900">Actions</th>
                </tr>
              </thead>
              <tbody>
                {paginatedPatients.map((patient, index) => (
                  <tr
                    key={patient.id}
                    className="border-b hover:bg-gray-50/50 transition-colors cursor-pointer"
                    onClick={() => handlePatientClick(patient.id)}
                  >
                    <td className="px-6 py-4">
                      <div className="flex items-center space-x-3">
                        <Avatar className="h-10 w-10">
                          <AvatarFallback className="bg-blue-500 text-white text-sm">
                            {getInitials(patient.name)}
                          </AvatarFallback>
                        </Avatar>
                        <div>
                          <p className="font-semibold text-gray-900">{patient.name}</p>
                          <p className="text-sm text-gray-500">{patient.mrn}</p>
                        </div>
                      </div>
                    </td>
                    <td className="px-6 py-4 text-sm text-gray-900">{patient.age}</td>
                    <td className="px-6 py-4 text-sm text-gray-900">{patient.sex}</td>
                    <td className="px-6 py-4 text-sm text-gray-900">{patient.lastVisit}</td>
                    <td className="px-6 py-4">
                      <Badge className={`text-xs rounded-full px-2 py-1 border ${getRiskColor(patient.risk)}`}>
                        {patient.risk}
                      </Badge>
                    </td>
                    <td className="px-6 py-4">
                      <Badge className={`text-xs rounded-full px-2 py-1 border ${getStatusColor(patient.status)}`}>
                        {patient.status}
                      </Badge>
                    </td>
                    <td className="px-6 py-4 text-sm text-gray-900">{patient.primarySpecialty}</td>
                    <td className="px-6 py-4">
                      <div className="flex items-center space-x-2">
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={(e) => {
                            e.stopPropagation();
                            navigate('/provider/messages');
                          }}
                          className="h-8 w-8 p-0"
                        >
                          <MessageSquare className="w-4 h-4" />
                        </Button>
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={(e) => {
                            e.stopPropagation();
                            navigate('/provider/schedule');
                          }}
                          className="h-8 w-8 p-0"
                        >
                          <Calendar className="w-4 h-4" />
                        </Button>
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={(e) => {
                            e.stopPropagation();
                            handlePatientClick(patient.id);
                          }}
                          className="h-8 w-8 p-0"
                        >
                          <FileText className="w-4 h-4" />
                        </Button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Mobile Cards */}
          <div className="lg:hidden space-y-4 p-4">
            {paginatedPatients.map((patient) => (
              <Card
                key={patient.id}
                className="cursor-pointer hover:shadow-md transition-shadow border border-gray-200 rounded-xl"
                onClick={() => handlePatientClick(patient.id)}
              >
                <CardContent className="p-4">
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex items-center space-x-3">
                      <Avatar className="h-12 w-12">
                        <AvatarFallback className="bg-blue-500 text-white">
                          {getInitials(patient.name)}
                        </AvatarFallback>
                      </Avatar>
                      <div>
                        <p className="font-semibold text-gray-900">{patient.name}</p>
                        <p className="text-sm text-gray-500">{patient.mrn}</p>
                      </div>
                    </div>
                    <div className="flex space-x-1">
                      <Badge className={`text-xs rounded-full px-2 py-1 border ${getRiskColor(patient.risk)}`}>
                        {patient.risk}
                      </Badge>
                      <Badge className={`text-xs rounded-full px-2 py-1 border ${getStatusColor(patient.status)}`}>
                        {patient.status}
                      </Badge>
                    </div>
                  </div>
                  <div className="grid grid-cols-2 gap-2 text-sm text-gray-600 mb-3">
                    <div>Age: {patient.age}</div>
                    <div>Sex: {patient.sex}</div>
                    <div>Last Visit: {patient.lastVisit}</div>
                    <div>Specialty: {patient.primarySpecialty}</div>
                  </div>
                  <div className="flex justify-end space-x-2">
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={(e) => {
                        e.stopPropagation();
                        navigate('/provider/messages');
                      }}
                    >
                      <MessageSquare className="w-4 h-4" />
                    </Button>
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={(e) => {
                        e.stopPropagation();
                        navigate('/provider/schedule');
                      }}
                    >
                      <Calendar className="w-4 h-4" />
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>

          {/* Pagination */}
          {totalPages > 1 && (
            <div className="flex items-center justify-between px-6 py-4 border-t bg-gray-50 rounded-b-2xl">
              <div className="text-sm text-gray-600">
                Page {currentPage} of {totalPages}
              </div>
              <div className="flex items-center space-x-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setCurrentPage(prev => Math.max(prev - 1, 1))}
                  disabled={currentPage === 1}
                  className="rounded-lg"
                >
                  <ChevronLeft className="w-4 h-4" />
                  Previous
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setCurrentPage(prev => Math.min(prev + 1, totalPages))}
                  disabled={currentPage === totalPages}
                  className="rounded-lg"
                >
                  Next
                  <ChevronRight className="w-4 h-4" />
                </Button>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* New Patient Modal */}
      <NewPatientModal
        isOpen={isNewPatientModalOpen}
        onClose={() => setIsNewPatientModalOpen(false)}
        onSuccess={handleNewPatientSuccess}
      />
    </div>
  );
}

export default PatientManagement;