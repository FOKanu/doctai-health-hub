import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Badge } from '@/components/ui/badge';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Calendar } from '@/components/ui/calendar';
import {
  Calendar as CalendarIcon,
  Clock,
  User,
  MapPin,
  Video,
  Phone,
  CheckCircle,
  AlertCircle,
  Star,
  Search
} from 'lucide-react';
import { useToast } from '@/hooks/use-toast';
import { useAuth } from '../../contexts/AuthContext';
import {
  getProviderAvailability,
  bookAppointment,
  getPatientAppointments,
  EnhancedAppointment,
  TimeSlot
} from '../../services/enhancedAppointmentService';

interface HealthcareProvider {
  id: string;
  name: string;
  specialty: string;
  rating: number;
  experienceYears: number;
  consultationFee: number;
  isAvailable: boolean;
  bio?: string;
  profileImageUrl?: string;
}

export function PatientAppointmentBooking() {
  const { user } = useAuth();
  const { toast } = useToast();
  const [providers, setProviders] = useState<HealthcareProvider[]>([]);
  const [selectedProvider, setSelectedProvider] = useState<HealthcareProvider | null>(null);
  const [selectedDate, setSelectedDate] = useState<Date | undefined>(undefined);
  const [selectedTimeSlot, setSelectedTimeSlot] = useState<TimeSlot | null>(null);
  const [availableSlots, setAvailableSlots] = useState<TimeSlot[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isBooking, setIsBooking] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedSpecialty, setSelectedSpecialty] = useState('all');

  // Booking form state
  const [bookingForm, setBookingForm] = useState({
    appointmentType: 'consultation' as 'consultation' | 'follow-up' | 'procedure' | 'emergency' | 'telemedicine',
    reason: '',
    symptoms: '',
    notes: '',
    insurance: 'TK - Techniker Krankenkasse'
  });

  // Mock providers data
  const mockProviders: HealthcareProvider[] = [
    {
      id: 'provider_001',
      name: 'Dr. Sarah Weber',
      specialty: 'Dermatology',
      rating: 4.8,
      experienceYears: 12,
      consultationFee: 120,
      isAvailable: true,
      bio: 'Specialized in skin cancer detection and treatment with 12 years of experience.',
      profileImageUrl: '/api/placeholder/100/100'
    },
    {
      id: 'provider_002',
      name: 'Dr. Michael Brown',
      specialty: 'Radiology',
      rating: 4.9,
      experienceYears: 15,
      consultationFee: 150,
      isAvailable: true,
      bio: 'Expert in medical imaging interpretation and AI-assisted diagnostics.',
      profileImageUrl: '/api/placeholder/100/100'
    },
    {
      id: 'provider_003',
      name: 'Dr. Emily Rodriguez',
      specialty: 'Oncology',
      rating: 4.7,
      experienceYears: 10,
      consultationFee: 180,
      isAvailable: true,
      bio: 'Specialized in cancer diagnosis and treatment planning.',
      profileImageUrl: '/api/placeholder/100/100'
    },
    {
      id: 'provider_004',
      name: 'Dr. James Wilson',
      specialty: 'General Practice',
      rating: 4.6,
      experienceYears: 8,
      consultationFee: 100,
      isAvailable: true,
      bio: 'Comprehensive primary care with focus on preventive medicine.',
      profileImageUrl: '/api/placeholder/100/100'
    }
  ];

  useEffect(() => {
    setProviders(mockProviders);
  }, []);

  useEffect(() => {
    if (selectedProvider && selectedDate) {
      loadAvailableSlots();
    }
  }, [selectedProvider, selectedDate]);

  const loadAvailableSlots = async () => {
    if (!selectedProvider || !selectedDate) return;

    try {
      setIsLoading(true);
      const dateString = selectedDate.toISOString().split('T')[0];
      const slots = await getProviderAvailability(selectedProvider.id, dateString);
      setAvailableSlots(slots);
    } catch (error) {
      console.error('Error loading available slots:', error);
      toast({
        title: "Error",
        description: "Failed to load available time slots.",
        variant: "destructive"
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleBookAppointment = async () => {
    if (!selectedProvider || !selectedDate || !selectedTimeSlot || !user) {
      toast({
        title: "Missing Information",
        description: "Please select a provider, date, and time slot.",
        variant: "destructive"
      });
      return;
    }

    if (!bookingForm.reason.trim()) {
      toast({
        title: "Missing Information",
        description: "Please provide a reason for the appointment.",
        variant: "destructive"
      });
      return;
    }

    try {
      setIsBooking(true);

      // Create appointment date/time
      const appointmentDateTime = new Date(selectedDate);
      const [hours, minutes] = selectedTimeSlot.time.split(':').map(Number);
      appointmentDateTime.setHours(hours, minutes, 0, 0);

      const appointmentData = {
        patientId: user.id,
        providerId: selectedProvider.id,
        patientName: user.name || 'Patient',
        providerName: selectedProvider.name,
        appointmentType: bookingForm.appointmentType,
        scheduledAt: appointmentDateTime.toISOString(),
        duration: selectedTimeSlot.duration,
        status: 'scheduled' as const,
        reason: bookingForm.reason.trim(),
        symptoms: bookingForm.symptoms ? bookingForm.symptoms.split(',').map(s => s.trim()) : [],
        notes: bookingForm.notes.trim(),
        insurance: bookingForm.insurance
      };

      const appointment = await bookAppointment(appointmentData);

      if (appointment) {
        toast({
          title: "Appointment Booked",
          description: `Your appointment with ${selectedProvider.name} has been scheduled successfully.`,
        });

        // Reset form
        setSelectedProvider(null);
        setSelectedDate(undefined);
        setSelectedTimeSlot(null);
        setBookingForm({
          appointmentType: 'consultation',
          reason: '',
          symptoms: '',
          notes: '',
          insurance: 'TK - Techniker Krankenkasse'
        });
      } else {
        throw new Error('Failed to book appointment');
      }
    } catch (error) {
      console.error('Error booking appointment:', error);
      toast({
        title: "Booking Failed",
        description: "Failed to book appointment. Please try again.",
        variant: "destructive"
      });
    } finally {
      setIsBooking(false);
    }
  };

  const filteredProviders = providers.filter(provider => {
    const matchesSearch = provider.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         provider.specialty.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesSpecialty = selectedSpecialty === 'all' || provider.specialty.toLowerCase() === selectedSpecialty.toLowerCase();
    return matchesSearch && matchesSpecialty && provider.isAvailable;
  });

  const getAppointmentTypeIcon = (type: string) => {
    switch (type) {
      case 'telemedicine': return <Video className="w-4 h-4" />;
      case 'consultation': return <User className="w-4 h-4" />;
      case 'procedure': return <AlertCircle className="w-4 h-4" />;
      default: return <CalendarIcon className="w-4 h-4" />;
    }
  };

  const getAppointmentTypeColor = (type: string) => {
    switch (type) {
      case 'telemedicine': return 'bg-blue-100 text-blue-800';
      case 'consultation': return 'bg-green-100 text-green-800';
      case 'procedure': return 'bg-orange-100 text-orange-800';
      case 'emergency': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-foreground">Book Appointment</h1>
          <p className="text-muted-foreground mt-1">Schedule an appointment with a healthcare provider</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Provider Selection */}
        <Card className="lg:col-span-1">
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <User className="w-5 h-5" />
              <span>Select Provider</span>
            </CardTitle>
            <div className="space-y-3">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
                <Input
                  placeholder="Search providers..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="pl-10"
                />
              </div>
              <Select value={selectedSpecialty} onValueChange={setSelectedSpecialty}>
                <SelectTrigger>
                  <SelectValue placeholder="Select specialty" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Specialties</SelectItem>
                  <SelectItem value="dermatology">Dermatology</SelectItem>
                  <SelectItem value="radiology">Radiology</SelectItem>
                  <SelectItem value="oncology">Oncology</SelectItem>
                  <SelectItem value="general practice">General Practice</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {filteredProviders.map((provider) => (
                <div
                  key={provider.id}
                  className={`p-4 border rounded-lg cursor-pointer transition-colors ${
                    selectedProvider?.id === provider.id
                      ? 'border-blue-500 bg-blue-50'
                      : 'border-gray-200 hover:border-gray-300'
                  }`}
                  onClick={() => setSelectedProvider(provider)}
                >
                  <div className="flex items-start space-x-3">
                    <div className="w-12 h-12 bg-gray-200 rounded-full flex items-center justify-center">
                      <User className="w-6 h-6 text-gray-500" />
                    </div>
                    <div className="flex-1">
                      <h3 className="font-medium text-gray-900">{provider.name}</h3>
                      <p className="text-sm text-gray-600">{provider.specialty}</p>
                      <div className="flex items-center space-x-2 mt-1">
                        <div className="flex items-center space-x-1">
                          <Star className="w-3 h-3 text-yellow-500 fill-current" />
                          <span className="text-xs text-gray-600">{provider.rating}</span>
                        </div>
                        <span className="text-xs text-gray-500">•</span>
                        <span className="text-xs text-gray-600">{provider.experienceYears} years</span>
                      </div>
                      <p className="text-sm text-gray-700 mt-1">{provider.bio}</p>
                      <div className="flex items-center justify-between mt-2">
                        <span className="text-sm font-medium text-green-600">
                          €{provider.consultationFee}
                        </span>
                        <Badge variant="outline" className="text-xs">
                          Available
                        </Badge>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Date and Time Selection */}
        <Card className="lg:col-span-1">
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <CalendarIcon className="w-5 h-5" />
              <span>Select Date & Time</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {/* Date Selection */}
              <div>
                <label className="text-sm font-medium text-gray-700 mb-2 block">
                  Select Date
                </label>
                <Calendar
                  mode="single"
                  selected={selectedDate}
                  onSelect={setSelectedDate}
                  disabled={(date) => date < new Date()}
                  className="rounded-md border"
                />
              </div>

              {/* Time Slots */}
              {selectedDate && (
                <div>
                  <label className="text-sm font-medium text-gray-700 mb-2 block">
                    Available Time Slots
                  </label>
                  {isLoading ? (
                    <div className="text-center py-4">
                      <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600 mx-auto"></div>
                      <p className="text-sm text-gray-600 mt-2">Loading slots...</p>
                    </div>
                  ) : (
                    <div className="grid grid-cols-3 gap-2">
                      {availableSlots.map((slot, index) => (
                        <Button
                          key={index}
                          variant={selectedTimeSlot?.time === slot.time ? "default" : "outline"}
                          size="sm"
                          onClick={() => setSelectedTimeSlot(slot)}
                          disabled={!slot.isAvailable}
                          className="text-xs"
                        >
                          {slot.time}
                        </Button>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          </CardContent>
        </Card>

        {/* Appointment Details */}
        <Card className="lg:col-span-1">
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Clock className="w-5 h-5" />
              <span>Appointment Details</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {/* Appointment Type */}
              <div>
                <label className="text-sm font-medium text-gray-700 mb-2 block">
                  Appointment Type
                </label>
                <Select
                  value={bookingForm.appointmentType}
                  onValueChange={(value: any) => setBookingForm(prev => ({ ...prev, appointmentType: value }))}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="consultation">In-Person Consultation</SelectItem>
                    <SelectItem value="telemedicine">Telemedicine</SelectItem>
                    <SelectItem value="follow-up">Follow-up</SelectItem>
                    <SelectItem value="procedure">Procedure</SelectItem>
                    <SelectItem value="emergency">Emergency</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* Reason */}
              <div>
                <label className="text-sm font-medium text-gray-700 mb-2 block">
                  Reason for Visit *
                </label>
                <Input
                  placeholder="e.g., Follow-up for AI analysis results"
                  value={bookingForm.reason}
                  onChange={(e) => setBookingForm(prev => ({ ...prev, reason: e.target.value }))}
                />
              </div>

              {/* Symptoms */}
              <div>
                <label className="text-sm font-medium text-gray-700 mb-2 block">
                  Symptoms (comma-separated)
                </label>
                <Input
                  placeholder="e.g., Skin lesion, chest pain"
                  value={bookingForm.symptoms}
                  onChange={(e) => setBookingForm(prev => ({ ...prev, symptoms: e.target.value }))}
                />
              </div>

              {/* Notes */}
              <div>
                <label className="text-sm font-medium text-gray-700 mb-2 block">
                  Additional Notes
                </label>
                <Textarea
                  placeholder="Any additional information..."
                  value={bookingForm.notes}
                  onChange={(e) => setBookingForm(prev => ({ ...prev, notes: e.target.value }))}
                  rows={3}
                />
              </div>

              {/* Insurance */}
              <div>
                <label className="text-sm font-medium text-gray-700 mb-2 block">
                  Insurance Provider
                </label>
                <Select
                  value={bookingForm.insurance}
                  onValueChange={(value) => setBookingForm(prev => ({ ...prev, insurance: value }))}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="TK - Techniker Krankenkasse">TK - Techniker Krankenkasse</SelectItem>
                    <SelectItem value="AOK - Allgemeine Ortskrankenkasse">AOK - Allgemeine Ortskrankenkasse</SelectItem>
                    <SelectItem value="Barmer">Barmer</SelectItem>
                    <SelectItem value="DAK-Gesundheit">DAK-Gesundheit</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* Booking Summary */}
              {selectedProvider && selectedDate && selectedTimeSlot && (
                <div className="p-4 bg-gray-50 rounded-lg">
                  <h4 className="font-medium text-gray-900 mb-2">Booking Summary</h4>
                  <div className="space-y-1 text-sm text-gray-600">
                    <div><strong>Provider:</strong> {selectedProvider.name}</div>
                    <div><strong>Date:</strong> {selectedDate.toLocaleDateString()}</div>
                    <div><strong>Time:</strong> {selectedTimeSlot.time}</div>
                    <div><strong>Duration:</strong> {selectedTimeSlot.duration} minutes</div>
                    <div><strong>Type:</strong>
                      <Badge className={`ml-2 ${getAppointmentTypeColor(bookingForm.appointmentType)}`}>
                        {bookingForm.appointmentType}
                      </Badge>
                    </div>
                    <div><strong>Fee:</strong> €{selectedProvider.consultationFee}</div>
                  </div>
                </div>
              )}

              {/* Book Button */}
              <Button
                onClick={handleBookAppointment}
                disabled={isBooking || !selectedProvider || !selectedDate || !selectedTimeSlot}
                className="w-full"
              >
                {isBooking ? 'Booking...' : 'Book Appointment'}
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
