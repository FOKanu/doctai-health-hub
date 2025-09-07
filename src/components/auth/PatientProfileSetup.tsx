import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { format } from 'date-fns';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Textarea } from '@/components/ui/textarea';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Progress } from '@/components/ui/progress';
import { Calendar } from '@/components/ui/calendar';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';
import { Checkbox } from '@/components/ui/checkbox';
import { User, Phone, MapPin, Users, Heart, Shield, CalendarIcon, ArrowRight, ArrowLeft, CheckCircle } from 'lucide-react';
import { supabase } from '@/integrations/supabase/client';
import { useAuth } from '@/contexts/AuthContext';
import { cn } from '@/lib/utils';
import { useToast } from '@/hooks/use-toast';

interface PatientProfileData {
  // Step 1: Basic Information
  dateOfBirth: Date | undefined;
  phoneNumber: string;
  address: string;
  city: string;
  state: string;
  zipCode: string;
  
  // Step 2: Emergency Contact
  emergencyContactName: string;
  emergencyContactPhone: string;
  emergencyContactRelation: string;
  
  // Step 3: Medical History (Basic)
  allergies: string[];
  currentMedications: string[];
  medicalConditions: string[];
  surgeries: string[];
  familyHistory: string;
  
  // Step 4: Insurance Information
  hasInsurance: boolean;
  insuranceProvider: string;
  policyNumber: string;
  groupNumber: string;
  subscriberName: string;
  subscriberDOB: Date | undefined;
}

interface ProfileSetupStep {
  id: number;
  title: string;
  description: string;
  icon: React.ReactNode;
}

const SETUP_STEPS: ProfileSetupStep[] = [
  {
    id: 1,
    title: 'Basic Information',
    description: 'Personal details and contact information',
    icon: <User className="w-5 h-5" />
  },
  {
    id: 2,
    title: 'Emergency Contact',
    description: 'Someone we can contact in case of emergency',
    icon: <Users className="w-5 h-5" />
  },
  {
    id: 3,
    title: 'Medical History',
    description: 'Health information to provide better care',
    icon: <Heart className="w-5 h-5" />
  },
  {
    id: 4,
    title: 'Insurance Information',
    description: 'Insurance details (optional)',
    icon: <Shield className="w-5 h-5" />
  }
];

export default function PatientProfileSetup() {
  const [currentStep, setCurrentStep] = useState(1);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const { user } = useAuth();
  const navigate = useNavigate();
  const { toast } = useToast();

  const [profileData, setProfileData] = useState<PatientProfileData>({
    dateOfBirth: undefined,
    phoneNumber: '',
    address: '',
    city: '',
    state: '',
    zipCode: '',
    emergencyContactName: '',
    emergencyContactPhone: '',
    emergencyContactRelation: '',
    allergies: [],
    currentMedications: [],
    medicalConditions: [],
    surgeries: [],
    familyHistory: '',
    hasInsurance: false,
    insuranceProvider: '',
    policyNumber: '',
    groupNumber: '',
    subscriberName: '',
    subscriberDOB: undefined
  });

  // Additional state for managing dynamic lists
  const [newAllergy, setNewAllergy] = useState('');
  const [newMedication, setNewMedication] = useState('');
  const [newCondition, setNewCondition] = useState('');
  const [newSurgery, setNewSurgery] = useState('');

  const handleInputChange = (field: keyof PatientProfileData, value: string | Date | boolean | string[]) => {
    setProfileData(prev => ({ ...prev, [field]: value }));
    // Clear error when user starts typing
    if (error) setError('');
  };

  const addToList = (field: 'allergies' | 'currentMedications' | 'medicalConditions' | 'surgeries', value: string) => {
    if (value.trim()) {
      const currentList = profileData[field] as string[];
      handleInputChange(field, [...currentList, value.trim()]);
    }
  };

  const removeFromList = (field: 'allergies' | 'currentMedications' | 'medicalConditions' | 'surgeries', index: number) => {
    const currentList = profileData[field] as string[];
    const newList = currentList.filter((_, i) => i !== index);
    handleInputChange(field, newList);
  };

  const validateStep = (step: number): boolean => {
    setError('');

    switch (step) {
      case 1:
        if (!profileData.dateOfBirth) {
          setError('Date of birth is required');
          return false;
        }
        if (!profileData.phoneNumber) {
          setError('Phone number is required');
          return false;
        }
        if (!profileData.address || !profileData.city || !profileData.state || !profileData.zipCode) {
          setError('Complete address is required');
          return false;
        }
        break;

      case 2:
        if (!profileData.emergencyContactName) {
          setError('Emergency contact name is required');
          return false;
        }
        if (!profileData.emergencyContactPhone) {
          setError('Emergency contact phone is required');
          return false;
        }
        if (!profileData.emergencyContactRelation) {
          setError('Emergency contact relationship is required');
          return false;
        }
        break;

      case 3:
        // Medical history is optional
        const hasAnyMedicalInfo = profileData.allergies.length > 0 || 
                                 profileData.currentMedications.length > 0 ||
                                 profileData.medicalConditions.length > 0 || 
                                 profileData.surgeries.length > 0 ||
                                 profileData.familyHistory.trim();
        
        if (!hasAnyMedicalInfo) {
          console.log('User chose to skip medical history');
        }
        break;

      case 4:
        // Insurance is optional, but if they say they have insurance, require basic info
        if (profileData.hasInsurance) {
          if (!profileData.insuranceProvider) {
            setError('Please provide your insurance provider');
            return false;
          }
          if (!profileData.policyNumber) {
            setError('Please provide your policy number');
            return false;
          }
        }
        break;

      default:
        return true;
    }

    return true;
  };

  const handleNext = () => {
    if (validateStep(currentStep)) {
      if (currentStep < SETUP_STEPS.length) {
        setCurrentStep(prev => prev + 1);
      } else {
        handleComplete();
      }
    }
  };

  const handleBack = () => {
    if (currentStep > 1) {
      setCurrentStep(prev => prev - 1);
    }
  };

  const handleSkip = () => {
    if (currentStep < SETUP_STEPS.length) {
      setCurrentStep(prev => prev + 1);
    } else {
      handleComplete();
    }
  };

  const handleComplete = async () => {
    if (!user) {
      setError('User session not found. Please log in again.');
      return;
    }

    setIsLoading(true);
    setError('');

    try {
      // Create or update user profile in Supabase
      const { error: profileError } = await supabase
        .from('profiles')
        .upsert([
          {
            user_id: user.id,
            display_name: `${user.name}`,
            bio: `Patient profile for ${user.name}`,
            updated_at: new Date().toISOString(),
          }
        ]);

      if (profileError) {
        throw profileError;
      }

      toast({
        title: "Profile Setup Complete!",
        description: "Your health profile has been successfully created.",
      });
      
      navigate('/patient/', { replace: true });
      
    } catch (error) {
      console.error('Profile setup error:', error);
      setError('Failed to save profile information. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const getProgressPercentage = () => {
    return ((currentStep - 1) / (SETUP_STEPS.length - 1)) * 100;
  };

  const renderStepContent = () => {
    switch (currentStep) {
      case 1:
        return (
          <div className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="dateOfBirth">Date of Birth</Label>
              <Popover>
                <PopoverTrigger asChild>
                  <Button
                    variant="outline"
                    className={cn(
                      "w-full justify-start text-left font-normal",
                      !profileData.dateOfBirth && "text-muted-foreground"
                    )}
                  >
                    <CalendarIcon className="mr-2 h-4 w-4" />
                    {profileData.dateOfBirth ? (
                      format(profileData.dateOfBirth, "PPP")
                    ) : (
                      <span>Pick your date of birth</span>
                    )}
                  </Button>
                </PopoverTrigger>
                <PopoverContent className="w-auto p-0" align="start">
                  <Calendar
                    mode="single"
                    selected={profileData.dateOfBirth}
                    onSelect={(date) => handleInputChange('dateOfBirth', date)}
                    disabled={(date) =>
                      date > new Date() || date < new Date("1900-01-01")
                    }
                    initialFocus
                    className={cn("p-3 pointer-events-auto")}
                  />
                </PopoverContent>
              </Popover>
            </div>

            <div className="space-y-2">
              <Label htmlFor="phoneNumber">Phone Number</Label>
              <Input
                id="phoneNumber"
                type="tel"
                placeholder="(555) 123-4567"
                value={profileData.phoneNumber}
                onChange={(e) => handleInputChange('phoneNumber', e.target.value)}
                required
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="address">Street Address</Label>
              <Input
                id="address"
                type="text"
                placeholder="123 Main Street"
                value={profileData.address}
                onChange={(e) => handleInputChange('address', e.target.value)}
                required
              />
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="city">City</Label>
                <Input
                  id="city"
                  type="text"
                  placeholder="New York"
                  value={profileData.city}
                  onChange={(e) => handleInputChange('city', e.target.value)}
                  required
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="state">State</Label>
                <Select 
                  value={profileData.state} 
                  onValueChange={(value) => handleInputChange('state', value)}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Select state" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="NY">New York</SelectItem>
                    <SelectItem value="CA">California</SelectItem>
                    <SelectItem value="TX">Texas</SelectItem>
                    <SelectItem value="FL">Florida</SelectItem>
                    <SelectItem value="IL">Illinois</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>

            <div className="space-y-2">
              <Label htmlFor="zipCode">ZIP Code</Label>
              <Input
                id="zipCode"
                type="text"
                placeholder="12345"
                value={profileData.zipCode}
                onChange={(e) => handleInputChange('zipCode', e.target.value)}
                required
              />
            </div>
          </div>
        );

      case 2:
        return (
          <div className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="emergencyContactName">Emergency Contact Name</Label>
              <Input
                id="emergencyContactName"
                type="text"
                placeholder="John Smith"
                value={profileData.emergencyContactName}
                onChange={(e) => handleInputChange('emergencyContactName', e.target.value)}
                required
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="emergencyContactPhone">Emergency Contact Phone</Label>
              <Input
                id="emergencyContactPhone"
                type="tel"
                placeholder="(555) 987-6543"
                value={profileData.emergencyContactPhone}
                onChange={(e) => handleInputChange('emergencyContactPhone', e.target.value)}
                required
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="emergencyContactRelation">Relationship</Label>
              <Select 
                value={profileData.emergencyContactRelation} 
                onValueChange={(value) => handleInputChange('emergencyContactRelation', value)}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select relationship" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="spouse">Spouse</SelectItem>
                  <SelectItem value="parent">Parent</SelectItem>
                  <SelectItem value="child">Child</SelectItem>
                  <SelectItem value="sibling">Sibling</SelectItem>
                  <SelectItem value="friend">Friend</SelectItem>
                  <SelectItem value="other">Other</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
        );

      case 3:
        return (
          <div className="space-y-6">
            <Alert className="border-blue-200 bg-blue-50">
              <Heart className="h-4 w-4 text-blue-600" />
              <AlertDescription className="text-blue-700">
                This information helps us provide better care. All fields are optional and you can update them later.
              </AlertDescription>
            </Alert>

            {/* Allergies */}
            <div className="space-y-2">
              <Label htmlFor="allergies">Known Allergies</Label>
              <div className="space-y-2">
                <div className="flex space-x-2">
                  <Input
                    placeholder="Enter an allergy (e.g., Penicillin, Peanuts)"
                    value={newAllergy}
                    onChange={(e) => setNewAllergy(e.target.value)}
                    onKeyPress={(e) => {
                      if (e.key === 'Enter') {
                        e.preventDefault();
                        addToList('allergies', newAllergy);
                        setNewAllergy('');
                      }
                    }}
                  />
                  <Button
                    type="button"
                    onClick={() => {
                      addToList('allergies', newAllergy);
                      setNewAllergy('');
                    }}
                    variant="outline"
                    size="sm"
                  >
                    Add
                  </Button>
                </div>
                {profileData.allergies.length > 0 && (
                  <div className="flex flex-wrap gap-2">
                    {profileData.allergies.map((allergy, index) => (
                      <div
                        key={index}
                        className="flex items-center bg-red-100 text-red-800 px-2 py-1 rounded-md text-sm"
                      >
                        <span>{allergy}</span>
                        <button
                          type="button"
                          onClick={() => removeFromList('allergies', index)}
                          className="ml-2 text-red-600 hover:text-red-800"
                        >
                          ×
                        </button>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>

            {/* Family History */}
            <div className="space-y-2">
              <Label htmlFor="familyHistory">Family Medical History</Label>
              <Textarea
                id="familyHistory"
                placeholder="Any relevant family medical history (e.g., diabetes in father, heart disease in grandmother)..."
                value={profileData.familyHistory}
                onChange={(e) => handleInputChange('familyHistory', e.target.value)}
                rows={3}
              />
            </div>
          </div>
        );

      case 4:
        return (
          <div className="space-y-4">
            <Alert className="border-green-200 bg-green-50">
              <Shield className="h-4 w-4 text-green-600" />
              <AlertDescription className="text-green-700">
                Insurance information helps us process payments and verify coverage. This step is optional.
              </AlertDescription>
            </Alert>

            <div className="space-y-4">
              <div className="space-y-3">
                <Label>Do you have health insurance?</Label>
                <div className="flex items-center space-x-6">
                  <div className="flex items-center space-x-2">
                    <Checkbox
                      id="hasInsurance"
                      checked={profileData.hasInsurance}
                      onCheckedChange={(checked) => handleInputChange('hasInsurance', checked as boolean)}
                    />
                    <Label htmlFor="hasInsurance">Yes, I have insurance</Label>
                  </div>
                </div>
              </div>

              {profileData.hasInsurance && (
                <div className="space-y-4 border-t pt-4">
                  <div className="space-y-2">
                    <Label htmlFor="insuranceProvider">Insurance Provider</Label>
                    <Select 
                      value={profileData.insuranceProvider} 
                      onValueChange={(value) => handleInputChange('insuranceProvider', value)}
                    >
                      <SelectTrigger>
                        <SelectValue placeholder="Select your insurance provider" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="aetna">Aetna</SelectItem>
                        <SelectItem value="anthem">Anthem</SelectItem>
                        <SelectItem value="bcbs">Blue Cross Blue Shield</SelectItem>
                        <SelectItem value="cigna">Cigna</SelectItem>
                        <SelectItem value="humana">Humana</SelectItem>
                        <SelectItem value="kaiser">Kaiser Permanente</SelectItem>
                        <SelectItem value="medicare">Medicare</SelectItem>
                        <SelectItem value="medicaid">Medicaid</SelectItem>
                        <SelectItem value="tricare">Tricare</SelectItem>
                        <SelectItem value="uhc">UnitedHealthcare</SelectItem>
                        <SelectItem value="other">Other</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="policyNumber">Policy Number</Label>
                    <Input
                      id="policyNumber"
                      type="text"
                      placeholder="Policy number"
                      value={profileData.policyNumber}
                      onChange={(e) => handleInputChange('policyNumber', e.target.value)}
                    />
                  </div>
                </div>
              )}

              {!profileData.hasInsurance && (
                <Alert className="border-yellow-200 bg-yellow-50">
                  <AlertDescription className="text-yellow-700">
                    No problem! You can add insurance information later, or we can discuss payment options during your visit.
                  </AlertDescription>
                </Alert>
              )}
            </div>
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center p-4">
      <div className="w-full max-w-2xl">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">Complete Your Profile</h1>
          <p className="text-gray-600">
            Let's set up your health profile in just a few steps
          </p>
        </div>

        {/* Progress Bar */}
        <div className="mb-8">
          <div className="flex justify-between items-center mb-4">
            {SETUP_STEPS.map((step, index) => (
              <div 
                key={step.id}
                className={`flex items-center ${index < SETUP_STEPS.length - 1 ? 'flex-1' : ''}`}
              >
                <div className={`
                  flex items-center justify-center w-10 h-10 rounded-full border-2 transition-colors
                  ${currentStep >= step.id 
                    ? 'bg-blue-600 border-blue-600 text-white' 
                    : 'bg-white border-gray-300 text-gray-500'}
                `}>
                  {currentStep > step.id ? (
                    <CheckCircle className="w-6 h-6" />
                  ) : (
                    step.icon
                  )}
                </div>
                {index < SETUP_STEPS.length - 1 && (
                  <div className={`flex-1 h-1 mx-4 rounded ${
                    currentStep > step.id ? 'bg-blue-600' : 'bg-gray-200'
                  }`} />
                )}
              </div>
            ))}
          </div>
          <Progress value={getProgressPercentage()} className="h-2" />
        </div>

        {/* Main Card */}
        <Card className="shadow-xl">
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              {SETUP_STEPS[currentStep - 1].icon}
              <span>{SETUP_STEPS[currentStep - 1].title}</span>
            </CardTitle>
            <CardDescription>
              {SETUP_STEPS[currentStep - 1].description}
            </CardDescription>
          </CardHeader>

          <CardContent className="space-y-6">
            {renderStepContent()}

            {error && (
              <Alert variant="destructive">
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}

            {/* Navigation Buttons */}
            <div className="flex justify-between pt-6">
              <Button
                onClick={handleBack}
                variant="outline"
                disabled={currentStep === 1}
                className="flex items-center space-x-2"
              >
                <ArrowLeft className="w-4 h-4" />
                <span>Back</span>
              </Button>

              <div className="flex space-x-3">
                {(currentStep === 3 || currentStep === 4) && (
                  <Button
                    onClick={handleSkip}
                    variant="ghost"
                    disabled={isLoading}
                  >
                    Skip for now
                  </Button>
                )}

                <Button
                  onClick={handleNext}
                  disabled={isLoading}
                  className="flex items-center space-x-2"
                >
                  <span>
                    {currentStep === SETUP_STEPS.length 
                      ? (isLoading ? 'Completing...' : 'Complete Setup') 
                      : 'Next'}
                  </span>
                  {currentStep < SETUP_STEPS.length && <ArrowRight className="w-4 h-4" />}
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}