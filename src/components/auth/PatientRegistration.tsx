import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Checkbox } from '@/components/ui/checkbox';
import { Eye, EyeOff, Activity, User, ArrowLeft, CheckCircle } from 'lucide-react';
import { supabase } from '@/integrations/supabase/client';

interface RegistrationFormData {
  email: string;
  password: string;
  confirmPassword: string;
  firstName: string;
  lastName: string;
  agreedToTerms: boolean;
  agreedToPrivacy: boolean;
}

export default function PatientRegistration() {
  const [formData, setFormData] = useState<RegistrationFormData>({
    email: '',
    password: '',
    confirmPassword: '',
    firstName: '',
    lastName: '',
    agreedToTerms: false,
    agreedToPrivacy: false
  });
  
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [registrationStep, setRegistrationStep] = useState<'form' | 'success'>('form');
  
  const navigate = useNavigate();

  const handleInputChange = (field: keyof RegistrationFormData) => (
    e: React.ChangeEvent<HTMLInputElement>
  ) => {
    if (field === 'agreedToTerms' || field === 'agreedToPrivacy') {
      setFormData(prev => ({ ...prev, [field]: e.target.checked }));
    } else {
      setFormData(prev => ({ ...prev, [field]: e.target.value }));
    }
    
    // Clear error when user starts typing
    if (error) setError('');
  };

  const validateForm = (): boolean => {
    // Reset error
    setError('');

    // Check required fields
    if (!formData.firstName.trim()) {
      setError('First name is required');
      return false;
    }

    if (!formData.lastName.trim()) {
      setError('Last name is required');
      return false;
    }

    if (!formData.email.trim()) {
      setError('Email is required');
      return false;
    }

    // Email validation
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(formData.email)) {
      setError('Please enter a valid email address');
      return false;
    }

    // Password validation
    if (formData.password.length < 8) {
      setError('Password must be at least 8 characters long');
      return false;
    }

    if (!/(?=.*[a-z])(?=.*[A-Z])(?=.*\d)/.test(formData.password)) {
      setError('Password must contain at least one uppercase letter, one lowercase letter, and one number');
      return false;
    }

    if (formData.password !== formData.confirmPassword) {
      setError('Passwords do not match');
      return false;
    }

    // Agreement validation
    if (!formData.agreedToTerms) {
      setError('Please agree to the Terms of Service');
      return false;
    }

    if (!formData.agreedToPrivacy) {
      setError('Please agree to the Privacy Policy');
      return false;
    }

    return true;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!validateForm()) {
      return;
    }

    setIsLoading(true);
    setError('');

    try {
      const redirectUrl = `${window.location.origin}/patient/`;
      
      const { data, error: signUpError } = await supabase.auth.signUp({
        email: formData.email,
        password: formData.password,
        options: {
          emailRedirectTo: redirectUrl,
          data: {
            first_name: formData.firstName,
            last_name: formData.lastName,
            role: 'patient'
          }
        }
      });

      if (signUpError) {
        if (signUpError.message.includes('User already registered')) {
          setError('An account with this email already exists. Please try logging in instead.');
        } else if (signUpError.message.includes('Password')) {
          setError('Password does not meet security requirements');
        } else {
          setError(signUpError.message || 'Registration failed. Please try again.');
        }
        return;
      }

      if (data.user) {
        setRegistrationStep('success');
      }

    } catch (error) {
      console.error('Registration error:', error);
      setError('An unexpected error occurred. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleBackToLogin = () => {
    navigate('/login');
  };

  const handleContinueToProfile = () => {
    navigate('/patient/profile-setup');
  };

  if (registrationStep === 'success') {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center p-4">
        <Card className="w-full max-w-md shadow-xl">
          <CardHeader className="space-y-1 text-center">
            <div className="mx-auto w-12 h-12 bg-green-600 rounded-xl flex items-center justify-center mb-4">
              <CheckCircle className="w-6 h-6 text-white" />
            </div>
            <CardTitle className="text-2xl font-bold text-gray-900">
              Registration Successful!
            </CardTitle>
            <CardDescription className="text-gray-600">
              Welcome to DoctAI Health Hub
            </CardDescription>
          </CardHeader>

          <CardContent className="space-y-4">
            <Alert className="border-green-200 bg-green-50">
              <CheckCircle className="h-4 w-4 text-green-600" />
              <AlertDescription className="text-green-700">
                Your account has been created successfully. Please check your email to verify your account, 
                then complete your profile to get started.
              </AlertDescription>
            </Alert>

            <div className="space-y-3">
              <Button 
                onClick={handleContinueToProfile} 
                className="w-full"
              >
                Complete Your Profile
              </Button>
              
              <Button 
                onClick={handleBackToLogin} 
                variant="outline" 
                className="w-full"
              >
                Back to Login
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        {/* Back to Login Button */}
        <Button
          onClick={handleBackToLogin}
          variant="ghost"
          className="mb-4 text-gray-600 hover:text-gray-900"
        >
          <ArrowLeft className="w-4 h-4 mr-2" />
          Back to Login
        </Button>

        <Card className="shadow-xl">
          <CardHeader className="space-y-1 text-center">
            <div className="mx-auto w-12 h-12 bg-blue-600 rounded-xl flex items-center justify-center mb-4">
              <User className="w-6 h-6 text-white" />
            </div>
            <CardTitle className="text-2xl font-bold text-gray-900">
              Create Your Account
            </CardTitle>
            <CardDescription className="text-gray-600">
              Join DoctAI to start your health journey
            </CardDescription>
          </CardHeader>

          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-4">
              {/* Name Fields */}
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="firstName">First Name</Label>
                  <Input
                    id="firstName"
                    type="text"
                    placeholder="John"
                    value={formData.firstName}
                    onChange={handleInputChange('firstName')}
                    required
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="lastName">Last Name</Label>
                  <Input
                    id="lastName"
                    type="text"
                    placeholder="Doe"
                    value={formData.lastName}
                    onChange={handleInputChange('lastName')}
                    required
                  />
                </div>
              </div>

              {/* Email Field */}
              <div className="space-y-2">
                <Label htmlFor="email">Email</Label>
                <Input
                  id="email"
                  type="email"
                  placeholder="john.doe@example.com"
                  value={formData.email}
                  onChange={handleInputChange('email')}
                  required
                />
              </div>

              {/* Password Field */}
              <div className="space-y-2">
                <Label htmlFor="password">Password</Label>
                <div className="relative">
                  <Input
                    id="password"
                    type={showPassword ? 'text' : 'password'}
                    placeholder="Create a strong password"
                    value={formData.password}
                    onChange={handleInputChange('password')}
                    required
                  />
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    className="absolute right-0 top-0 h-full px-3 py-2 hover:bg-transparent"
                    onClick={() => setShowPassword(!showPassword)}
                  >
                    {showPassword ? (
                      <EyeOff className="h-4 w-4" />
                    ) : (
                      <Eye className="h-4 w-4" />
                    )}
                  </Button>
                </div>
                <p className="text-xs text-gray-500">
                  Must be at least 8 characters with uppercase, lowercase, and number
                </p>
              </div>

              {/* Confirm Password Field */}
              <div className="space-y-2">
                <Label htmlFor="confirmPassword">Confirm Password</Label>
                <div className="relative">
                  <Input
                    id="confirmPassword"
                    type={showConfirmPassword ? 'text' : 'password'}
                    placeholder="Confirm your password"
                    value={formData.confirmPassword}
                    onChange={handleInputChange('confirmPassword')}
                    required
                  />
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    className="absolute right-0 top-0 h-full px-3 py-2 hover:bg-transparent"
                    onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                  >
                    {showConfirmPassword ? (
                      <EyeOff className="h-4 w-4" />
                    ) : (
                      <Eye className="h-4 w-4" />
                    )}
                  </Button>
                </div>
              </div>

              {/* Agreement Checkboxes */}
              <div className="space-y-3">
                <div className="flex items-start space-x-2">
                  <Checkbox
                    id="terms"
                    checked={formData.agreedToTerms}
                    onCheckedChange={(checked) => 
                      setFormData(prev => ({ ...prev, agreedToTerms: checked as boolean }))
                    }
                  />
                  <Label htmlFor="terms" className="text-sm leading-5 cursor-pointer">
                    I agree to the{' '}
                    <a href="#" className="text-blue-600 hover:underline">
                      Terms of Service
                    </a>
                  </Label>
                </div>

                <div className="flex items-start space-x-2">
                  <Checkbox
                    id="privacy"
                    checked={formData.agreedToPrivacy}
                    onCheckedChange={(checked) => 
                      setFormData(prev => ({ ...prev, agreedToPrivacy: checked as boolean }))
                    }
                  />
                  <Label htmlFor="privacy" className="text-sm leading-5 cursor-pointer">
                    I agree to the{' '}
                    <a href="#" className="text-blue-600 hover:underline">
                      Privacy Policy
                    </a>
                  </Label>
                </div>
              </div>

              {error && (
                <Alert variant="destructive">
                  <AlertDescription>{error}</AlertDescription>
                </Alert>
              )}

              <Button
                type="submit"
                className="w-full"
                disabled={isLoading}
              >
                {isLoading ? 'Creating Account...' : 'Create Account'}
              </Button>
            </form>

            <div className="mt-4 text-center">
              <p className="text-sm text-gray-600">
                Already have an account?{' '}
                <button
                  onClick={handleBackToLogin}
                  className="text-blue-600 hover:underline font-medium"
                >
                  Sign in here
                </button>
              </p>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}