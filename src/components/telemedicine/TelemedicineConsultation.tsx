import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Badge } from '@/components/ui/badge';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Calendar } from '@/components/ui/calendar';
import {
  Video,
  Phone,
  MessageSquare,
  Clock,
  Star,
  MapPin,
  Calendar as CalendarIcon,
  User,
  Stethoscope,
  AlertTriangle,
  CheckCircle,
  X,
  Info
} from 'lucide-react';
import { telemedicineService, type HealthcareProvider, type Appointment, type TelemedicineConsultation } from '@/services/telemedicineService';

interface TelemedicineConsultationProps {
  userId: string;
  className?: string;
}

export function TelemedicineConsultation({ userId, className }: TelemedicineConsultationProps) {
  const [providers, setProviders] = useState<HealthcareProvider[]>([]);
  const [selectedProvider, setSelectedProvider] = useState<HealthcareProvider | null>(null);
  const [selectedDate, setSelectedDate] = useState<Date | undefined>(undefined);
  const [selectedTime, setSelectedTime] = useState<string>('');
  const [availableSlots, setAvailableSlots] = useState<any[]>([]);
  const [consultationType, setConsultationType] = useState<'video' | 'audio' | 'chat'>('video');
  const [reason, setReason] = useState('');
  const [symptoms, setSymptoms] = useState<string[]>([]);
  const [isUrgent, setIsUrgent] = useState(false);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('book');
  const [appointments, setAppointments] = useState<Appointment[]>([]);
  const [consultations, setConsultations] = useState<TelemedicineConsultation[]>([]);

  useEffect(() => {
    loadProviders();
    loadUserData();
  }, [userId]);

  useEffect(() => {
    if (selectedProvider && selectedDate) {
      loadAvailableSlots();
    }
  }, [selectedProvider, selectedDate]);

  const loadProviders = async () => {
    try {
      const data = await telemedicineService.getHealthcareProviders();
      setProviders(data);
    } catch (error) {
      console.error('Error loading providers:', error);
    }
  };

  const loadUserData = async () => {
    try {
      const [appointmentsData, consultationsData] = await Promise.all([
        telemedicineService.getAppointments({ userId }),
        telemedicineService.getConsultations({ userId })
      ]);
      setAppointments(appointmentsData);
      setConsultations(consultationsData);
    } catch (error) {
      console.error('Error loading user data:', error);
    }
  };

  const loadAvailableSlots = async () => {
    if (!selectedProvider || !selectedDate) return;

    try {
      const slots = await telemedicineService.getProviderAvailability(
        selectedProvider.id,
        selectedDate.toISOString().split('T')[0]
      );
      setAvailableSlots(slots.filter(slot => slot.is_available));
    } catch (error) {
      console.error('Error loading available slots:', error);
    }
  };

  const handleBookAppointment = async () => {
    if (!selectedProvider || !selectedDate || !selectedTime || !reason) {
      return;
    }

    try {
      setLoading(true);
      const scheduledAt = new Date(selectedDate);
      const [hours, minutes] = selectedTime.split(':');
      scheduledAt.setHours(parseInt(hours), parseInt(minutes), 0, 0);

      await telemedicineService.bookAppointment({
        patientId: userId,
        providerId: selectedProvider.id,
        appointmentType: 'consultation',
        scheduledAt: scheduledAt.toISOString(),
        reason,
        symptoms,
        isUrgent
      });

      // Reset form
      setSelectedProvider(null);
      setSelectedDate(undefined);
      setSelectedTime('');
      setReason('');
      setSymptoms([]);
      setIsUrgent(false);

      // Reload user data
      await loadUserData();

      alert('Appointment booked successfully!');
    } catch (error) {
      console.error('Error booking appointment:', error);
      alert('Failed to book appointment. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleJoinConsultation = async (consultationId: string) => {
    try {
      const { meetingUrl } = await telemedicineService.joinConsultation(consultationId, userId);
      window.open(meetingUrl, '_blank');
    } catch (error) {
      console.error('Error joining consultation:', error);
      alert('Failed to join consultation. Please try again.');
    }
  };

  const getConsultationTypeIcon = (type: string) => {
    switch (type) {
      case 'video':
        return <Video className="h-4 w-4" />;
      case 'audio':
        return <Phone className="h-4 w-4" />;
      case 'chat':
        return <MessageSquare className="h-4 w-4" />;
      default:
        return <Video className="h-4 w-4" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'scheduled':
        return 'bg-blue-100 text-blue-800';
      case 'in_progress':
        return 'bg-green-100 text-green-800';
      case 'completed':
        return 'bg-gray-100 text-gray-800';
      case 'cancelled':
        return 'bg-red-100 text-red-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  const addSymptom = (symptom: string) => {
    if (symptom && !symptoms.includes(symptom)) {
      setSymptoms([...symptoms, symptom]);
    }
  };

  const removeSymptom = (symptom: string) => {
    setSymptoms(symptoms.filter(s => s !== symptom));
  };

  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Stethoscope className="h-5 w-5" />
          Telemedicine Consultation
        </CardTitle>
        <CardDescription>
          Book virtual consultations with healthcare providers
        </CardDescription>
      </CardHeader>

      <CardContent>
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="book">Book Consultation</TabsTrigger>
            <TabsTrigger value="appointments">My Appointments</TabsTrigger>
            <TabsTrigger value="consultations">Past Consultations</TabsTrigger>
          </TabsList>

          <TabsContent value="book" className="space-y-6">
            {/* Provider Selection */}
            <div className="space-y-4">
              <h3 className="font-medium">Select Healthcare Provider</h3>
              <div className="grid gap-4">
                {providers.map((provider) => (
                  <div
                    key={provider.id}
                    className={`p-4 border rounded-lg cursor-pointer transition-colors ${
                      selectedProvider?.id === provider.id
                        ? 'border-blue-500 bg-blue-50'
                        : 'border-gray-200 hover:border-gray-300'
                    }`}
                    onClick={() => setSelectedProvider(provider)}
                  >
                    <div className="flex items-start gap-4">
                      <Avatar className="h-12 w-12">
                        <AvatarImage src={provider.profile_image_url} />
                        <AvatarFallback>
                          {provider.provider_name.split(' ').map(n => n[0]).join('')}
                        </AvatarFallback>
                      </Avatar>

                      <div className="flex-1 space-y-2">
                        <div className="flex items-center justify-between">
                          <h4 className="font-medium">{provider.provider_name}</h4>
                          <Badge variant="outline">{provider.specialty.replace('_', ' ')}</Badge>
                        </div>

                        <div className="flex items-center gap-4 text-sm text-gray-600">
                          <div className="flex items-center gap-1">
                            <Star className="h-4 w-4 text-yellow-500" />
                            <span>{provider.rating}</span>
                          </div>
                          <div className="flex items-center gap-1">
                            <User className="h-4 w-4" />
                            <span>{provider.total_consultations} consultations</span>
                          </div>
                          <div className="flex items-center gap-1">
                            <Clock className="h-4 w-4" />
                            <span>{provider.experience_years} years</span>
                          </div>
                        </div>

                        {provider.bio && (
                          <p className="text-sm text-gray-600">{provider.bio}</p>
                        )}

                        <div className="flex items-center justify-between">
                          <div className="text-sm">
                            <span className="font-medium">${provider.consultation_fee}</span>
                            <span className="text-gray-600"> per consultation</span>
                          </div>
                          {provider.is_available && (
                            <Badge className="bg-green-100 text-green-800">Available</Badge>
                          )}
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {selectedProvider && (
              <>
                {/* Consultation Type */}
                <div className="space-y-4">
                  <h3 className="font-medium">Consultation Type</h3>
                  <div className="grid grid-cols-3 gap-4">
                    {[
                      { value: 'video', label: 'Video Call', icon: Video },
                      { value: 'audio', label: 'Audio Call', icon: Phone },
                      { value: 'chat', label: 'Chat', icon: MessageSquare }
                    ].map((type) => (
                      <div
                        key={type.value}
                        className={`p-4 border rounded-lg cursor-pointer transition-colors ${
                          consultationType === type.value
                            ? 'border-blue-500 bg-blue-50'
                            : 'border-gray-200 hover:border-gray-300'
                        }`}
                        onClick={() => setConsultationType(type.value as any)}
                      >
                        <div className="flex flex-col items-center gap-2">
                          <type.icon className="h-6 w-6" />
                          <span className="text-sm font-medium">{type.label}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Date and Time Selection */}
                <div className="space-y-4">
                  <h3 className="font-medium">Select Date & Time</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <label className="text-sm font-medium mb-2 block">Date</label>
                      <Calendar
                        mode="single"
                        selected={selectedDate}
                        onSelect={setSelectedDate}
                        disabled={(date) => date < new Date()}
                        className="rounded-md border"
                      />
                    </div>

                    <div>
                      <label className="text-sm font-medium mb-2 block">Time</label>
                      <Select value={selectedTime} onValueChange={setSelectedTime}>
                        <SelectTrigger>
                          <SelectValue placeholder="Select time" />
                        </SelectTrigger>
                        <SelectContent>
                          {availableSlots.map((slot, index) => (
                            <SelectItem key={index} value={slot.start_time}>
                              {slot.start_time} - {slot.end_time}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                  </div>
                </div>

                {/* Consultation Details */}
                <div className="space-y-4">
                  <h3 className="font-medium">Consultation Details</h3>

                  <div>
                    <label className="text-sm font-medium mb-2 block">Reason for Visit</label>
                    <Textarea
                      placeholder="Describe your symptoms or reason for consultation..."
                      value={reason}
                      onChange={(e) => setReason(e.target.value)}
                      rows={3}
                    />
                  </div>

                  <div>
                    <label className="text-sm font-medium mb-2 block">Symptoms</label>
                    <div className="space-y-2">
                      <div className="flex gap-2">
                        <Input
                          placeholder="Add a symptom..."
                          onKeyPress={(e) => {
                            if (e.key === 'Enter') {
                              e.preventDefault();
                              addSymptom((e.target as HTMLInputElement).value);
                              (e.target as HTMLInputElement).value = '';
                            }
                          }}
                        />
                        <Button
                          variant="outline"
                          onClick={() => {
                            const input = document.querySelector('input[placeholder="Add a symptom..."]') as HTMLInputElement;
                            if (input?.value) {
                              addSymptom(input.value);
                              input.value = '';
                            }
                          }}
                        >
                          Add
                        </Button>
                      </div>

                      {symptoms.length > 0 && (
                        <div className="flex flex-wrap gap-2">
                          {symptoms.map((symptom, index) => (
                            <Badge key={index} variant="secondary" className="gap-1">
                              {symptom}
                              <X
                                className="h-3 w-3 cursor-pointer"
                                onClick={() => removeSymptom(symptom)}
                              />
                            </Badge>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>

                  <div className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      id="urgent"
                      checked={isUrgent}
                      onChange={(e) => setIsUrgent(e.target.checked)}
                    />
                    <label htmlFor="urgent" className="text-sm">
                      Mark as urgent
                    </label>
                  </div>
                </div>

                {/* Book Button */}
                <Button
                  onClick={handleBookAppointment}
                  disabled={loading || !selectedProvider || !selectedDate || !selectedTime || !reason}
                  className="w-full"
                >
                  {loading ? 'Booking...' : 'Book Consultation'}
                </Button>
              </>
            )}
          </TabsContent>

          <TabsContent value="appointments" className="space-y-4">
            <h3 className="font-medium">Upcoming Appointments</h3>
            {appointments.length === 0 ? (
              <Alert>
                <Info className="h-4 w-4" />
                <AlertDescription>
                  No upcoming appointments. Book a consultation to get started.
                </AlertDescription>
              </Alert>
            ) : (
              <div className="space-y-4">
                {appointments.map((appointment) => (
                  <div key={appointment.id} className="p-4 border rounded-lg">
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <CalendarIcon className="h-4 w-4" />
                        <span className="font-medium">
                          {new Date(appointment.scheduled_at).toLocaleDateString()}
                        </span>
                        <span className="text-gray-600">
                          {new Date(appointment.scheduled_at).toLocaleTimeString()}
                        </span>
                      </div>
                      <Badge className={getStatusColor(appointment.status)}>
                        {appointment.status.replace('_', ' ')}
                      </Badge>
                    </div>

                    <div className="space-y-1 text-sm">
                      <p><strong>Type:</strong> {appointment.appointment_type.replace('_', ' ')}</p>
                      {appointment.reason && <p><strong>Reason:</strong> {appointment.reason}</p>}
                      {appointment.symptoms && appointment.symptoms.length > 0 && (
                        <div>
                          <strong>Symptoms:</strong>
                          <div className="flex flex-wrap gap-1 mt-1">
                            {appointment.symptoms.map((symptom, index) => (
                              <Badge key={index} variant="outline" className="text-xs">
                                {symptom}
                              </Badge>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </TabsContent>

          <TabsContent value="consultations" className="space-y-4">
            <h3 className="font-medium">Past Consultations</h3>
            {consultations.length === 0 ? (
              <Alert>
                <Info className="h-4 w-4" />
                <AlertDescription>
                  No past consultations found.
                </AlertDescription>
              </Alert>
            ) : (
              <div className="space-y-4">
                {consultations.map((consultation) => (
                  <div key={consultation.id} className="p-4 border rounded-lg">
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-2">
                        {getConsultationTypeIcon(consultation.consultation_type)}
                        <span className="font-medium">
                          {new Date(consultation.scheduled_at).toLocaleDateString()}
                        </span>
                      </div>
                      <Badge className={getStatusColor(consultation.status)}>
                        {consultation.status.replace('_', ' ')}
                      </Badge>
                    </div>

                    <div className="space-y-1 text-sm">
                      <p><strong>Type:</strong> {consultation.consultation_type.replace('_', ' ')}</p>
                      {consultation.duration_minutes && (
                        <p><strong>Duration:</strong> {consultation.duration_minutes} minutes</p>
                      )}
                      {consultation.diagnosis && (
                        <p><strong>Diagnosis:</strong> {consultation.diagnosis}</p>
                      )}
                      {consultation.recommendations && consultation.recommendations.length > 0 && (
                        <div>
                          <strong>Recommendations:</strong>
                          <ul className="list-disc list-inside mt-1">
                            {consultation.recommendations.map((rec, index) => (
                              <li key={index}>{rec}</li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>

                    {consultation.status === 'scheduled' && consultation.meeting_url && (
                      <Button
                        onClick={() => handleJoinConsultation(consultation.id)}
                        className="mt-3"
                        size="sm"
                      >
                        Join Consultation
                      </Button>
                    )}
                  </div>
                ))}
              </div>
            )}
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}
