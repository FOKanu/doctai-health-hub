import React, { useState, useEffect } from 'react';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Textarea } from '@/components/ui/textarea';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { Calendar, Clock, User, MapPin, Video, Phone, Trash2, Save } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';

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

interface Patient {
  id: string;
  name: string;
  mrn: string;
}

// Mock patients for selection
const mockPatients: Patient[] = [
  { id: 'P001', name: 'Sarah Johnson', mrn: 'MRN-789012' },
  { id: 'P002', name: 'Michael Chen', mrn: 'MRN-789013' },
  { id: 'P003', name: 'Emma Davis', mrn: 'MRN-789014' },
  { id: 'P004', name: 'Robert Wilson', mrn: 'MRN-789015' },
  { id: 'P005', name: 'Lisa Anderson', mrn: 'MRN-789016' }
];

interface AppointmentModalProps {
  isOpen: boolean;
  onClose: () => void;
  appointment: AppointmentData | null;
  onSave: (appointment: AppointmentData) => void;
  onDelete?: (appointmentId: string) => void;
}

export function AppointmentModal({ 
  isOpen, 
  onClose, 
  appointment, 
  onSave, 
  onDelete 
}: AppointmentModalProps) {
  const { toast } = useToast();
  const [formData, setFormData] = useState<AppointmentData>({
    id: '',
    title: '',
    patientName: '',
    patientId: '',
    start: new Date(),
    end: new Date(),
    type: 'in-person',
    status: 'scheduled',
    reason: '',
    duration: 30,
    notes: ''
  });

  const [selectedPatient, setSelectedPatient] = useState<Patient | null>(null);

  useEffect(() => {
    if (appointment) {
      setFormData(appointment);
      // Find the patient if editing
      if (appointment.patientId) {
        const patient = mockPatients.find(p => p.id === appointment.patientId);
        setSelectedPatient(patient || null);
      }
    } else {
      // Reset for new appointment
      const now = new Date();
      const startTime = new Date(now);
      startTime.setMinutes(Math.ceil(now.getMinutes() / 15) * 15); // Round to next 15min
      const endTime = new Date(startTime.getTime() + 30 * 60000); // 30min default
      
      setFormData({
        id: '',
        title: '',
        patientName: '',
        patientId: '',
        start: startTime,
        end: endTime,
        type: 'in-person',
        status: 'scheduled',
        reason: '',
        duration: 30,
        notes: ''
      });
      setSelectedPatient(null);
    }
  }, [appointment]);

  const handlePatientSelect = (patientId: string) => {
    const patient = mockPatients.find(p => p.id === patientId);
    if (patient) {
      setSelectedPatient(patient);
      setFormData(prev => ({
        ...prev,
        patientId: patient.id,
        patientName: patient.name,
        title: `${patient.name} - ${prev.reason || 'Appointment'}`
      }));
    }
  };

  const handleDateTimeChange = (field: 'start' | 'end', value: string) => {
    const newDate = new Date(value);
    setFormData(prev => {
      const updated = { ...prev, [field]: newDate };
      
      // Auto-adjust end time when start time changes
      if (field === 'start') {
        updated.end = new Date(newDate.getTime() + prev.duration * 60000);
      }
      
      // Update duration when end time changes
      if (field === 'end') {
        updated.duration = Math.max(15, (newDate.getTime() - prev.start.getTime()) / 60000);
      }
      
      return updated;
    });
  };

  const handleDurationChange = (minutes: number) => {
    setFormData(prev => ({
      ...prev,
      duration: minutes,
      end: new Date(prev.start.getTime() + minutes * 60000)
    }));
  };

  const handleReasonChange = (reason: string) => {
    setFormData(prev => ({
      ...prev,
      reason,
      title: selectedPatient ? `${selectedPatient.name} - ${reason}` : reason
    }));
  };

  const handleSave = () => {
    if (!formData.patientId) {
      toast({
        title: "Patient Required",
        description: "Please select a patient for this appointment.",
        variant: "destructive",
      });
      return;
    }

    if (!formData.reason.trim()) {
      toast({
        title: "Reason Required",
        description: "Please provide a reason for the appointment.",
        variant: "destructive",
      });
      return;
    }

    onSave(formData);
    toast({
      title: "Appointment Saved",
      description: `Appointment ${appointment?.id ? 'updated' : 'created'} successfully.`,
    });
  };

  const handleDelete = () => {
    if (appointment?.id && onDelete) {
      onDelete(appointment.id);
      toast({
        title: "Appointment Deleted",
        description: "The appointment has been cancelled and removed.",
        variant: "destructive",
      });
    }
  };

  const formatDateTime = (date: Date) => {
    return date.toISOString().slice(0, 16);
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'telemedicine': return <Video className="w-4 h-4" />;
      case 'phone': return <Phone className="w-4 h-4" />;
      default: return <MapPin className="w-4 h-4" />;
    }
  };

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle className="flex items-center space-x-2">
            <Calendar className="w-5 h-5" />
            <span>{appointment?.id ? 'Edit Appointment' : 'New Appointment'}</span>
            {appointment?.id && (
              <Badge variant="secondary" className="ml-2">
                {appointment.id}
              </Badge>
            )}
          </DialogTitle>
        </DialogHeader>

        <div className="space-y-6">
          {/* Patient Selection */}
          <div className="space-y-2">
            <Label htmlFor="patient">Patient *</Label>
            <Select 
              value={formData.patientId} 
              onValueChange={handlePatientSelect}
            >
              <SelectTrigger>
                <SelectValue placeholder="Select a patient">
                  {selectedPatient && (
                    <div className="flex items-center space-x-2">
                      <User className="w-4 h-4" />
                      <span>{selectedPatient.name}</span>
                      <Badge variant="outline" className="text-xs">
                        {selectedPatient.mrn}
                      </Badge>
                    </div>
                  )}
                </SelectValue>
              </SelectTrigger>
              <SelectContent>
                {mockPatients.map((patient) => (
                  <SelectItem key={patient.id} value={patient.id}>
                    <div className="flex items-center space-x-2">
                      <span>{patient.name}</span>
                      <Badge variant="outline" className="text-xs">
                        {patient.mrn}
                      </Badge>
                    </div>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="grid grid-cols-2 gap-4">
            {/* Appointment Type */}
            <div className="space-y-2">
              <Label>Appointment Type</Label>
              <Select 
                value={formData.type} 
                onValueChange={(value) => setFormData(prev => ({ ...prev, type: value as any }))}
              >
                <SelectTrigger>
                  <SelectValue>
                    <div className="flex items-center space-x-2">
                      {getTypeIcon(formData.type)}
                      <span className="capitalize">{formData.type}</span>
                    </div>
                  </SelectValue>
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="in-person">
                    <div className="flex items-center space-x-2">
                      <MapPin className="w-4 h-4" />
                      <span>In-Person</span>
                    </div>
                  </SelectItem>
                  <SelectItem value="telemedicine">
                    <div className="flex items-center space-x-2">
                      <Video className="w-4 h-4" />
                      <span>Telemedicine</span>
                    </div>
                  </SelectItem>
                  <SelectItem value="phone">
                    <div className="flex items-center space-x-2">
                      <Phone className="w-4 h-4" />
                      <span>Phone</span>
                    </div>
                  </SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Status */}
            <div className="space-y-2">
              <Label>Status</Label>
              <Select 
                value={formData.status} 
                onValueChange={(value) => setFormData(prev => ({ ...prev, status: value as any }))}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="scheduled">Scheduled</SelectItem>
                  <SelectItem value="confirmed">Confirmed</SelectItem>
                  <SelectItem value="completed">Completed</SelectItem>
                  <SelectItem value="cancelled">Cancelled</SelectItem>
                  <SelectItem value="no-show">No Show</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>

          {/* Reason */}
          <div className="space-y-2">
            <Label htmlFor="reason">Reason for Visit *</Label>
            <Input
              id="reason"
              value={formData.reason}
              onChange={(e) => handleReasonChange(e.target.value)}
              placeholder="e.g., Annual Physical, Follow-up, Consultation"
            />
          </div>

          {/* Date and Time */}
          <div className="grid grid-cols-3 gap-4">
            <div className="space-y-2">
              <Label>Start Time</Label>
              <Input
                type="datetime-local"
                value={formatDateTime(formData.start)}
                onChange={(e) => handleDateTimeChange('start', e.target.value)}
              />
            </div>

            <div className="space-y-2">
              <Label>Duration (minutes)</Label>
              <Select 
                value={formData.duration.toString()} 
                onValueChange={(value) => handleDurationChange(parseInt(value))}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="15">15 minutes</SelectItem>
                  <SelectItem value="30">30 minutes</SelectItem>
                  <SelectItem value="45">45 minutes</SelectItem>
                  <SelectItem value="60">1 hour</SelectItem>
                  <SelectItem value="90">1.5 hours</SelectItem>
                  <SelectItem value="120">2 hours</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label>End Time</Label>
              <div className="flex items-center h-10 px-3 py-2 border rounded-md bg-muted text-sm">
                <Clock className="w-4 h-4 mr-2 text-muted-foreground" />
                {formData.end.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
              </div>
            </div>
          </div>

          {/* Notes */}
          <div className="space-y-2">
            <Label htmlFor="notes">Notes (Optional)</Label>
            <Textarea
              id="notes"
              value={formData.notes || ''}
              onChange={(e) => setFormData(prev => ({ ...prev, notes: e.target.value }))}
              placeholder="Any additional notes or special instructions..."
              rows={3}
            />
          </div>

          <Separator />

          {/* Actions */}
          <div className="flex items-center justify-between">
            <div>
              {appointment?.id && onDelete && (
                <Button variant="destructive" size="sm" onClick={handleDelete}>
                  <Trash2 className="w-4 h-4 mr-2" />
                  Delete Appointment
                </Button>
              )}
            </div>
            
            <div className="flex items-center space-x-3">
              <Button variant="outline" onClick={onClose}>
                Cancel
              </Button>
              <Button onClick={handleSave}>
                <Save className="w-4 h-4 mr-2" />
                {appointment?.id ? 'Update' : 'Save'} Appointment
              </Button>
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}