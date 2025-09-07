import React, { useState, useMemo } from 'react';
import { Calendar, momentLocalizer, Event } from 'react-big-calendar';
import moment from 'moment';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Input } from '@/components/ui/input';
import { 
  Calendar as CalendarIcon, 
  Clock, 
  Plus, 
  Filter,
  Search,
  Users,
  Video,
  MapPin,
  Phone
} from 'lucide-react';
import { AppointmentModal } from '../modals/AppointmentModal';
import 'react-big-calendar/lib/css/react-big-calendar.css';
import '../../../styles/calendar-custom.css';

const localizer = momentLocalizer(moment);

interface AppointmentData {
  id: string;
  title: string;
  patientName: string;
  patientId: string;
  start: Date;
  end: Date;
  type: 'in-person' | 'telemedicine' | 'phone';
  status: 'scheduled' | 'confirmed' | 'completed' | 'cancelled' | 'no-show';
  reason: string;
  duration: number;
  notes?: string;
}

// Mock appointments data
const mockAppointments: AppointmentData[] = [
  {
    id: 'A001',
    title: 'Sarah Johnson - Annual Physical',
    patientName: 'Sarah Johnson',
    patientId: 'P001',
    start: new Date(2024, 1, 20, 9, 0),
    end: new Date(2024, 1, 20, 10, 0),
    type: 'in-person',
    status: 'confirmed',
    reason: 'Annual Physical Exam',
    duration: 60,
    notes: 'First annual visit'
  },
  {
    id: 'A002',
    title: 'Michael Chen - Follow-up',
    patientName: 'Michael Chen',
    patientId: 'P002',
    start: new Date(2024, 1, 20, 10, 30),
    end: new Date(2024, 1, 20, 11, 0),
    type: 'telemedicine',
    status: 'scheduled',
    reason: 'Diabetes Follow-up',
    duration: 30
  },
  {
    id: 'A003',
    title: 'Emma Davis - Consultation',
    patientName: 'Emma Davis',
    patientId: 'P003',
    start: new Date(2024, 1, 20, 14, 0),
    end: new Date(2024, 1, 20, 14, 45),
    type: 'phone',
    status: 'confirmed',
    reason: 'Lab Results Discussion',
    duration: 45
  },
  {
    id: 'A004',
    title: 'Robert Wilson - Check-up',
    patientName: 'Robert Wilson',
    patientId: 'P004',
    start: new Date(2024, 1, 21, 11, 0),
    end: new Date(2024, 1, 21, 12, 0),
    type: 'in-person',
    status: 'scheduled',
    reason: 'Routine Check-up',
    duration: 60
  },
  {
    id: 'A005',
    title: 'Lisa Anderson - Telemedicine',
    patientName: 'Lisa Anderson',
    patientId: 'P005',
    start: new Date(2024, 1, 22, 15, 30),
    end: new Date(2024, 1, 22, 16, 0),
    type: 'telemedicine',
    status: 'confirmed',
    reason: 'Medication Review',
    duration: 30
  }
];

function Schedule() {
  const [view, setView] = useState<'month' | 'week' | 'day'>('week');
  const [selectedDate, setSelectedDate] = useState(new Date());
  const [appointments, setAppointments] = useState<AppointmentData[]>(mockAppointments);
  const [selectedAppointment, setSelectedAppointment] = useState<AppointmentData | null>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [typeFilter, setTypeFilter] = useState<string>('all');

  const filteredAppointments = useMemo(() => {
    return appointments.filter(apt => {
      const matchesSearch = apt.patientName.toLowerCase().includes(searchTerm.toLowerCase()) ||
                          apt.reason.toLowerCase().includes(searchTerm.toLowerCase());
      const matchesStatus = statusFilter === 'all' || apt.status === statusFilter;
      const matchesType = typeFilter === 'all' || apt.type === typeFilter;
      
      return matchesSearch && matchesStatus && matchesType;
    });
  }, [appointments, searchTerm, statusFilter, typeFilter]);

  const todayAppointments = useMemo(() => {
    const today = new Date();
    return appointments.filter(apt => {
      const aptDate = new Date(apt.start);
      return aptDate.toDateString() === today.toDateString();
    }).sort((a, b) => a.start.getTime() - b.start.getTime());
  }, [appointments]);

  const handleSelectEvent = (event: AppointmentData) => {
    setSelectedAppointment(event);
    setIsModalOpen(true);
  };

  const handleSelectSlot = ({ start }: { start: Date }) => {
    const newAppointment: AppointmentData = {
      id: '',
      title: 'New Appointment',
      patientName: '',
      patientId: '',
      start,
      end: new Date(start.getTime() + 30 * 60000), // 30 minutes default
      type: 'in-person',
      status: 'scheduled',
      reason: '',
      duration: 30
    };
    setSelectedAppointment(newAppointment);
    setIsModalOpen(true);
  };

  const getStatusBadgeVariant = (status: string) => {
    switch (status) {
      case 'confirmed': return 'default';
      case 'scheduled': return 'secondary';
      case 'completed': return 'secondary';
      case 'cancelled': return 'destructive';
      case 'no-show': return 'destructive';
      default: return 'secondary';
    }
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'telemedicine': return <Video className="w-4 h-4" />;
      case 'phone': return <Phone className="w-4 h-4" />;
      default: return <MapPin className="w-4 h-4" />;
    }
  };

  const eventStyleGetter = (event: AppointmentData) => {
    let backgroundColor = '#3174ad';
    
    switch (event.status) {
      case 'confirmed':
        backgroundColor = '#22c55e';
        break;
      case 'scheduled':
        backgroundColor = '#3b82f6';
        break;
      case 'completed':
        backgroundColor = '#64748b';
        break;
      case 'cancelled':
        backgroundColor = '#ef4444';
        break;
      case 'no-show':
        backgroundColor = '#f97316';
        break;
    }

    return {
      style: {
        backgroundColor,
        borderRadius: '6px',
        opacity: 0.8,
        color: 'white',
        border: '0px',
        display: 'block'
      }
    };
  };

  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto p-6 space-y-6">
        {/* Header */}
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div className="space-y-1">
            <h1 className="text-2xl sm:text-3xl font-bold text-foreground tracking-tight">Schedule Management</h1>
            <p className="text-muted-foreground">Manage appointments and availability</p>
          </div>
          
          <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-3">
            <Select value={view} onValueChange={(value) => setView(value as 'month' | 'week' | 'day')}>
              <SelectTrigger className="w-full sm:w-32 transition-all duration-200">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="month">Month</SelectItem>
                <SelectItem value="week">Week</SelectItem>
                <SelectItem value="day">Day</SelectItem>
              </SelectContent>
            </Select>
            
            <Button 
              onClick={() => {
                setSelectedAppointment(null);
                setIsModalOpen(true);
              }}
              className="w-full sm:w-auto transition-all duration-200 hover:scale-105"
            >
              <Plus className="w-4 h-4 mr-2" />
              New Appointment
            </Button>
          </div>
        </div>

        {/* Main Content */}
        <div className="flex flex-col xl:flex-row gap-6 transition-all duration-300 ease-in-out">
          {/* Calendar Section */}
          <div className="flex-1 min-w-0 transition-all duration-300 ease-in-out">
            <Card className="h-full shadow-sm border-0 bg-card">
              <CardContent className="p-3 sm:p-6">
                {/* Calendar Navigation */}
                <div className="flex items-center justify-between mb-4 p-2 bg-muted/30 rounded-lg">
                  <div className="flex items-center space-x-2">
                    <Button variant="outline" size="sm" onClick={() => setSelectedDate(new Date())}>
                      Today
                    </Button>
                    <Button variant="ghost" size="sm">
                      <span className="hidden sm:inline">← </span>Back
                    </Button>
                    <Button variant="ghost" size="sm">
                      Next<span className="hidden sm:inline"> →</span>
                    </Button>
                  </div>
                  <div className="text-sm font-medium text-muted-foreground">
                    {selectedDate.toLocaleDateString('en-US', { 
                      month: 'long', 
                      year: 'numeric',
                      ...(view !== 'month' && { day: 'numeric' })
                    })}
                  </div>
                </div>

                {/* Calendar Container */}
                <div className="w-full overflow-hidden rounded-lg border bg-background">
                  <div className="h-[500px] sm:h-[600px] lg:h-[650px] transition-all duration-200">
                    <Calendar
                      localizer={localizer}
                      events={filteredAppointments}
                      startAccessor="start"
                      endAccessor="end"
                      view={view}
                      onView={(newView) => setView(newView)}
                      date={selectedDate}
                      onNavigate={(date) => setSelectedDate(date)}
                      onSelectEvent={handleSelectEvent}
                      onSelectSlot={handleSelectSlot}
                      selectable
                      eventPropGetter={eventStyleGetter}
                      step={15}
                      timeslots={4}
                      min={new Date(0, 0, 0, 7, 0, 0)}
                      max={new Date(0, 0, 0, 19, 0, 0)}
                      className="rbc-calendar-custom"
                    />
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Sidebar */}
          <div className="w-full xl:w-80 flex-shrink-0 transition-all duration-300 ease-in-out">
            <div className="space-y-4 xl:space-y-6 sticky top-6">
              {/* Today's Appointments */}
              <Card className="shadow-sm border-0 bg-card transition-all duration-200 hover:shadow-md">
                <CardHeader className="pb-3">
                  <CardTitle className="flex items-center space-x-2 text-base">
                    <div className="p-2 bg-primary/10 rounded-lg">
                      <Clock className="w-4 h-4 text-primary" />
                    </div>
                    <span>Today's Schedule</span>
                  </CardTitle>
                </CardHeader>
                <CardContent className="pt-0">
                  {todayAppointments.length > 0 ? (
                    <div className="space-y-3 max-h-60 overflow-y-auto">
                      {todayAppointments.map((apt) => (
                        <div 
                          key={apt.id} 
                          className="p-3 border rounded-lg hover:bg-muted/50 cursor-pointer transition-all duration-200 hover:scale-[1.02] hover:shadow-sm"
                          onClick={() => handleSelectEvent(apt)}
                        >
                          <div className="flex items-center justify-between mb-2">
                            <span className="font-medium text-sm truncate">{apt.patientName}</span>
                            <Badge variant={getStatusBadgeVariant(apt.status)} className="text-xs shrink-0">
                              {apt.status}
                            </Badge>
                          </div>
                          <div className="flex items-center space-x-2 text-xs text-muted-foreground mb-1">
                            <Clock className="w-3 h-3 shrink-0" />
                            <span>{moment(apt.start).format('HH:mm')}</span>
                            <div className="shrink-0">{getTypeIcon(apt.type)}</div>
                            <span className="capitalize truncate">{apt.type}</span>
                          </div>
                          <div className="text-xs text-muted-foreground truncate">
                            {apt.reason}
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="text-center py-8 text-muted-foreground text-sm">
                      <Clock className="w-8 h-8 mx-auto mb-2 opacity-50" />
                      <p>No appointments today</p>
                    </div>
                  )}
                </CardContent>
              </Card>

              {/* Quick Stats */}
              <Card className="shadow-sm border-0 bg-card transition-all duration-200 hover:shadow-md">
                <CardHeader className="pb-3">
                  <CardTitle className="flex items-center space-x-2 text-base">
                    <div className="p-2 bg-blue-500/10 rounded-lg">
                      <Users className="w-4 h-4 text-blue-600" />
                    </div>
                    <span>Quick Stats</span>
                  </CardTitle>
                </CardHeader>
                <CardContent className="pt-0 space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="text-center p-3 bg-muted/30 rounded-lg">
                      <div className="text-2xl font-bold text-foreground">{todayAppointments.length}</div>
                      <div className="text-xs text-muted-foreground">Today</div>
                    </div>
                    <div className="text-center p-3 bg-muted/30 rounded-lg">
                      <div className="text-2xl font-bold text-foreground">{appointments.length}</div>
                      <div className="text-xs text-muted-foreground">This Week</div>
                    </div>
                  </div>
                  
                  <div className="space-y-2">
                    <div className="flex justify-between items-center">
                      <div className="flex items-center space-x-2">
                        <Video className="w-3 h-3 text-green-600" />
                        <span className="text-sm text-muted-foreground">Telemedicine</span>
                      </div>
                      <span className="font-medium">{appointments.filter(a => a.type === 'telemedicine').length}</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <div className="flex items-center space-x-2">
                        <MapPin className="w-3 h-3 text-blue-600" />
                        <span className="text-sm text-muted-foreground">In-Person</span>
                      </div>
                      <span className="font-medium">{appointments.filter(a => a.type === 'in-person').length}</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <div className="flex items-center space-x-2">
                        <Phone className="w-3 h-3 text-orange-600" />
                        <span className="text-sm text-muted-foreground">Phone</span>
                      </div>
                      <span className="font-medium">{appointments.filter(a => a.type === 'phone').length}</span>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Filters */}
              <Card className="shadow-sm border-0 bg-card transition-all duration-200 hover:shadow-md">
                <CardHeader className="pb-3">
                  <CardTitle className="flex items-center space-x-2 text-base">
                    <div className="p-2 bg-purple-500/10 rounded-lg">
                      <Filter className="w-4 h-4 text-purple-600" />
                    </div>
                    <span>Filters</span>
                  </CardTitle>
                </CardHeader>
                <CardContent className="pt-0 space-y-4">
                  <div>
                    <label className="text-sm font-medium mb-2 block">Search</label>
                    <div className="relative">
                      <Search className="w-4 h-4 absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground" />
                      <Input 
                        placeholder="Patient or reason..."
                        value={searchTerm}
                        onChange={(e) => setSearchTerm(e.target.value)}
                        className="pl-9 transition-all duration-200 focus:scale-[1.02]"
                      />
                    </div>
                  </div>

                  <div>
                    <label className="text-sm font-medium mb-2 block">Status</label>
                    <Select value={statusFilter} onValueChange={setStatusFilter}>
                      <SelectTrigger className="transition-all duration-200 focus:scale-[1.02]">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="all">All Status</SelectItem>
                        <SelectItem value="scheduled">Scheduled</SelectItem>
                        <SelectItem value="confirmed">Confirmed</SelectItem>
                        <SelectItem value="completed">Completed</SelectItem>
                        <SelectItem value="cancelled">Cancelled</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div>
                    <label className="text-sm font-medium mb-2 block">Type</label>
                    <Select value={typeFilter} onValueChange={setTypeFilter}>
                      <SelectTrigger className="transition-all duration-200 focus:scale-[1.02]">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="all">All Types</SelectItem>
                        <SelectItem value="in-person">In-Person</SelectItem>
                        <SelectItem value="telemedicine">Telemedicine</SelectItem>
                        <SelectItem value="phone">Phone</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>
        </div>
      </div>

      {/* Appointment Modal */}
      <AppointmentModal
        isOpen={isModalOpen}
        onClose={() => {
          setIsModalOpen(false);
          setSelectedAppointment(null);
        }}
        appointment={selectedAppointment}
        onSave={(appointment) => {
          if (appointment.id) {
            // Update existing
            setAppointments(prev => 
              prev.map(apt => apt.id === appointment.id ? appointment : apt)
            );
          } else {
            // Create new
            const newAppointment = {
              ...appointment,
              id: `A${String(appointments.length + 1).padStart(3, '0')}`
            };
            setAppointments(prev => [...prev, newAppointment]);
          }
          setIsModalOpen(false);
          setSelectedAppointment(null);
        }}
        onDelete={(appointmentId) => {
          setAppointments(prev => prev.filter(apt => apt.id !== appointmentId));
          setIsModalOpen(false);
          setSelectedAppointment(null);
        }}
      />
    </div>
  );
}

export default Schedule;