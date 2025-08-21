import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { 
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { useToast } from '@/components/ui/use-toast';
import { User, Phone, Mail } from 'lucide-react';

interface NewPatientModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSuccess: (patient: any) => void;
}

interface PatientFormData {
  firstName: string;
  lastName: string;
  dateOfBirth: string;
  sex: string;
  mrn: string;
  phone: string;
  email: string;
  primarySpecialty: string;
  risk: string;
  status: string;
}

export function NewPatientModal({ isOpen, onClose, onSuccess }: NewPatientModalProps) {
  const { toast } = useToast();
  const [isLoading, setIsLoading] = useState(false);
  const [formData, setFormData] = useState<PatientFormData>({
    firstName: '',
    lastName: '',
    dateOfBirth: '',
    sex: '',
    mrn: '',
    phone: '',
    email: '',
    primarySpecialty: '',
    risk: 'Low',
    status: 'New Patient'
  });

  const [errors, setErrors] = useState<Partial<PatientFormData>>({});

  // Generate unique MRN
  const generateMRN = () => {
    const timestamp = Date.now().toString().slice(-6);
    const random = Math.floor(Math.random() * 999).toString().padStart(3, '0');
    return `MRN${timestamp}${random}`;
  };

  // Validate form
  const validateForm = (): boolean => {
    const newErrors: Partial<PatientFormData> = {};

    if (!formData.firstName.trim()) newErrors.firstName = 'First name is required';
    if (!formData.lastName.trim()) newErrors.lastName = 'Last name is required';
    if (!formData.dateOfBirth) newErrors.dateOfBirth = 'Date of birth is required';
    if (!formData.sex) newErrors.sex = 'Sex is required';
    if (!formData.phone.trim()) newErrors.phone = 'Phone number is required';
    if (!formData.email.trim()) newErrors.email = 'Email is required';
    if (!formData.primarySpecialty) newErrors.primarySpecialty = 'Primary specialty is required';

    // Validate phone format (basic)
    if (formData.phone && !/^\(\d{3}\) \d{3}-\d{4}$/.test(formData.phone)) {
      newErrors.phone = 'Phone must be in format (555) 123-4567';
    }

    // Validate email format
    if (formData.email && !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(formData.email)) {
      newErrors.email = 'Please enter a valid email address';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  // Format phone number as user types
  const formatPhoneNumber = (value: string) => {
    const phoneNumber = value.replace(/\D/g, '');
    const phoneNumberLength = phoneNumber.length;

    if (phoneNumberLength < 4) return phoneNumber;
    if (phoneNumberLength < 7) {
      return `(${phoneNumber.slice(0, 3)}) ${phoneNumber.slice(3)}`;
    }
    return `(${phoneNumber.slice(0, 3)}) ${phoneNumber.slice(3, 6)}-${phoneNumber.slice(6, 10)}`;
  };

  const handlePhoneChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const formattedPhone = formatPhoneNumber(e.target.value);
    setFormData({ ...formData, phone: formattedPhone });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!validateForm()) {
      return;
    }

    setIsLoading(true);

    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1500));
      
      // Generate MRN if not provided
      const mrn = formData.mrn || generateMRN();

      const newPatient = {
        id: Date.now().toString(),
        name: `${formData.firstName} ${formData.lastName}`,
        age: new Date().getFullYear() - new Date(formData.dateOfBirth).getFullYear(),
        sex: formData.sex,
        lastVisit: new Date().toISOString().split('T')[0],
        risk: formData.risk,
        status: formData.status,
        primarySpecialty: formData.primarySpecialty,
        mrn,
        phone: formData.phone,
        email: formData.email,
        dateOfBirth: formData.dateOfBirth
      };

      // Show success toast
      toast({
        title: "Patient Created Successfully",
        description: `${newPatient.name} has been added to your patient roster.`,
      });

      // Reset form
      setFormData({
        firstName: '',
        lastName: '',
        dateOfBirth: '',
        sex: '',
        mrn: '',
        phone: '',
        email: '',
        primarySpecialty: '',
        risk: 'Low',
        status: 'New Patient'
      });
      setErrors({});

      // Call success callback
      onSuccess(newPatient);
      
    } catch (error) {
      toast({
        title: "Error Creating Patient",
        description: "There was an error creating the patient. Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleClose = () => {
    // Reset form when closing
    setFormData({
      firstName: '',
      lastName: '',
      dateOfBirth: '',
      sex: '',
      mrn: '',
      phone: '',
      email: '',
      primarySpecialty: '',
      risk: 'Low',
      status: 'New Patient'
    });
    setErrors({});
    onClose();
  };

  return (
    <Dialog open={isOpen} onOpenChange={handleClose}>
      <DialogContent className="sm:max-w-md md:max-w-lg rounded-2xl">
        <DialogHeader>
          <DialogTitle className="flex items-center space-x-2">
            <User className="w-5 h-5" />
            <span>New Patient</span>
          </DialogTitle>
          <DialogDescription>
            Add a new patient to your roster. All fields marked with * are required.
          </DialogDescription>
        </DialogHeader>

        <form onSubmit={handleSubmit} className="space-y-4">
          {/* Name Fields */}
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="firstName">First Name *</Label>
              <Input
                id="firstName"
                value={formData.firstName}
                onChange={(e) => setFormData({ ...formData, firstName: e.target.value })}
                className={`rounded-xl ${errors.firstName ? 'border-red-500' : ''}`}
                placeholder="Enter first name"
              />
              {errors.firstName && (
                <p className="text-xs text-red-500">{errors.firstName}</p>
              )}
            </div>
            <div className="space-y-2">
              <Label htmlFor="lastName">Last Name *</Label>
              <Input
                id="lastName"
                value={formData.lastName}
                onChange={(e) => setFormData({ ...formData, lastName: e.target.value })}
                className={`rounded-xl ${errors.lastName ? 'border-red-500' : ''}`}
                placeholder="Enter last name"
              />
              {errors.lastName && (
                <p className="text-xs text-red-500">{errors.lastName}</p>
              )}
            </div>
          </div>

          {/* DOB and Sex */}
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="dateOfBirth">Date of Birth *</Label>
              <Input
                id="dateOfBirth"
                type="date"
                value={formData.dateOfBirth}
                onChange={(e) => setFormData({ ...formData, dateOfBirth: e.target.value })}
                className={`rounded-xl ${errors.dateOfBirth ? 'border-red-500' : ''}`}
              />
              {errors.dateOfBirth && (
                <p className="text-xs text-red-500">{errors.dateOfBirth}</p>
              )}
            </div>
            <div className="space-y-2">
              <Label htmlFor="sex">Sex *</Label>
              <Select value={formData.sex} onValueChange={(value) => setFormData({ ...formData, sex: value })}>
                <SelectTrigger className={`rounded-xl ${errors.sex ? 'border-red-500' : ''}`}>
                  <SelectValue placeholder="Select sex" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="M">Male</SelectItem>
                  <SelectItem value="F">Female</SelectItem>
                  <SelectItem value="Other">Other</SelectItem>
                </SelectContent>
              </Select>
              {errors.sex && (
                <p className="text-xs text-red-500">{errors.sex}</p>
              )}
            </div>
          </div>

          {/* MRN (optional, auto-generated) */}
          <div className="space-y-2">
            <Label htmlFor="mrn">MRN (Optional)</Label>
            <Input
              id="mrn"
              value={formData.mrn}
              onChange={(e) => setFormData({ ...formData, mrn: e.target.value })}
              className="rounded-xl"
              placeholder="Auto-generated if left blank"
            />
          </div>

          {/* Contact Information */}
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="phone">Phone *</Label>
              <div className="relative">
                <Phone className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
                <Input
                  id="phone"
                  value={formData.phone}
                  onChange={handlePhoneChange}
                  className={`pl-10 rounded-xl ${errors.phone ? 'border-red-500' : ''}`}
                  placeholder="(555) 123-4567"
                  maxLength={14}
                />
              </div>
              {errors.phone && (
                <p className="text-xs text-red-500">{errors.phone}</p>
              )}
            </div>
            <div className="space-y-2">
              <Label htmlFor="email">Email *</Label>
              <div className="relative">
                <Mail className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
                <Input
                  id="email"
                  type="email"
                  value={formData.email}
                  onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                  className={`pl-10 rounded-xl ${errors.email ? 'border-red-500' : ''}`}
                  placeholder="patient@example.com"
                />
              </div>
              {errors.email && (
                <p className="text-xs text-red-500">{errors.email}</p>
              )}
            </div>
          </div>

          {/* Medical Information */}
          <div className="space-y-2">
            <Label htmlFor="primarySpecialty">Primary Specialty *</Label>
            <Select 
              value={formData.primarySpecialty} 
              onValueChange={(value) => setFormData({ ...formData, primarySpecialty: value })}
            >
              <SelectTrigger className={`rounded-xl ${errors.primarySpecialty ? 'border-red-500' : ''}`}>
                <SelectValue placeholder="Select primary specialty" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="Cardiology">Cardiology</SelectItem>
                <SelectItem value="Neurology">Neurology</SelectItem>
                <SelectItem value="Ophthalmology">Ophthalmology</SelectItem>
                <SelectItem value="Orthopedics">Orthopedics</SelectItem>
                <SelectItem value="General Practice">General Practice</SelectItem>
                <SelectItem value="Internal Medicine">Internal Medicine</SelectItem>
              </SelectContent>
            </Select>
            {errors.primarySpecialty && (
              <p className="text-xs text-red-500">{errors.primarySpecialty}</p>
            )}
          </div>

          {/* Risk and Status */}
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="risk">Risk Level</Label>
              <Select value={formData.risk} onValueChange={(value) => setFormData({ ...formData, risk: value })}>
                <SelectTrigger className="rounded-xl">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="Low">Low</SelectItem>
                  <SelectItem value="Medium">Medium</SelectItem>
                  <SelectItem value="High">High</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label htmlFor="status">Status</Label>
              <Select value={formData.status} onValueChange={(value) => setFormData({ ...formData, status: value })}>
                <SelectTrigger className="rounded-xl">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="New Patient">New Patient</SelectItem>
                  <SelectItem value="Follow-up">Follow-up</SelectItem>
                  <SelectItem value="Routine">Routine</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>

          <DialogFooter className="flex gap-2 pt-4">
            <Button
              type="button"
              variant="outline"
              onClick={handleClose}
              disabled={isLoading}
              className="rounded-xl"
            >
              Cancel
            </Button>
            <Button
              type="submit"
              disabled={isLoading}
              className="bg-blue-600 hover:bg-blue-700 rounded-xl"
            >
              {isLoading ? 'Creating...' : 'Create Patient'}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}