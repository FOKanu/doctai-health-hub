import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Loader2, Bell, MessageSquare, Mail, Smartphone, AlertTriangle } from 'lucide-react';
import { apiServiceManager } from '@/services/api/apiServiceManager';

interface NotificationManagerProps {
  className?: string;
}

export const NotificationManager: React.FC<NotificationManagerProps> = ({ className }) => {
  const [notificationType, setNotificationType] = useState<'appointment' | 'medication' | 'emergency'>('appointment');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<unknown>(null);
  const [error, setError] = useState<string | null>(null);

  // Form data
  const [formData, setFormData] = useState({
    userId: 'user123',
    phone: '',
    email: '',
    // Appointment data
    appointmentDate: '',
    appointmentTime: '',
    doctor: '',
    specialty: '',
    // Medication data
    medicationName: '',
    dosage: '',
    frequency: '',
    medicationTime: '',
    // Emergency data
    emergencyType: 'health_emergency',
    severity: 'medium',
    emergencyMessage: ''
  });

  const handleInputChange = (field: string, value: string) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  const handleSubmit = async () => {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const contactInfo = {
        phone: formData.phone || undefined,
        email: formData.email || undefined
      };

      let response;

      switch (notificationType) {
        case 'appointment':
          response = await apiServiceManager.sendAppointmentReminder(
            formData.userId,
            {
              date: formData.appointmentDate,
              time: formData.appointmentTime,
              doctor: formData.doctor,
              specialty: formData.specialty
            },
            contactInfo
          );
          break;

        case 'medication':
          response = await apiServiceManager.sendMedicationReminder(
            formData.userId,
            formData.medicationName,
            contactInfo
          );
          break;

        case 'emergency':
          response = await apiServiceManager.sendEmergencyAlert(
            formData.userId,
            formData.emergencyMessage,
            contactInfo
          );
          break;
      }

      if (response.success) {
        setResult(response.data);
      } else {
        setError(response.error || 'Failed to send notification');
      }
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : 'An error occurred';
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  const renderForm = () => {
    switch (notificationType) {
      case 'appointment':
        return (
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <Label htmlFor="appointmentDate">Date</Label>
                <Input
                  id="appointmentDate"
                  type="date"
                  value={formData.appointmentDate}
                  onChange={(e) => handleInputChange('appointmentDate', e.target.value)}
                />
              </div>
              <div>
                <Label htmlFor="appointmentTime">Time</Label>
                <Input
                  id="appointmentTime"
                  type="time"
                  value={formData.appointmentTime}
                  onChange={(e) => handleInputChange('appointmentTime', e.target.value)}
                />
              </div>
            </div>
            <div>
              <Label htmlFor="doctor">Doctor Name</Label>
              <Input
                id="doctor"
                placeholder="Dr. Smith"
                value={formData.doctor}
                onChange={(e) => handleInputChange('doctor', e.target.value)}
              />
            </div>
            <div>
              <Label htmlFor="specialty">Specialty</Label>
              <Input
                id="specialty"
                placeholder="Cardiology"
                value={formData.specialty}
                onChange={(e) => handleInputChange('specialty', e.target.value)}
              />
            </div>
          </div>
        );

      case 'medication':
        return (
          <div className="space-y-4">
            <div>
              <Label htmlFor="medicationName">Medication Name</Label>
              <Input
                id="medicationName"
                placeholder="Aspirin"
                value={formData.medicationName}
                onChange={(e) => handleInputChange('medicationName', e.target.value)}
              />
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <Label htmlFor="dosage">Dosage</Label>
                <Input
                  id="dosage"
                  placeholder="100mg"
                  value={formData.dosage}
                  onChange={(e) => handleInputChange('dosage', e.target.value)}
                />
              </div>
              <div>
                <Label htmlFor="frequency">Frequency</Label>
                <Input
                  id="frequency"
                  placeholder="twice daily"
                  value={formData.frequency}
                  onChange={(e) => handleInputChange('frequency', e.target.value)}
                />
              </div>
            </div>
            <div>
              <Label htmlFor="medicationTime">Time</Label>
              <Input
                id="medicationTime"
                type="time"
                value={formData.medicationTime}
                onChange={(e) => handleInputChange('medicationTime', e.target.value)}
              />
            </div>
          </div>
        );

      case 'emergency':
        return (
          <div className="space-y-4">
            <div>
              <Label htmlFor="emergencyType">Emergency Type</Label>
              <Select
                value={formData.emergencyType}
                onValueChange={(value) => handleInputChange('emergencyType', value)}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="health_emergency">Health Emergency</SelectItem>
                  <SelectItem value="medication_overdue">Medication Overdue</SelectItem>
                  <SelectItem value="appointment_missed">Appointment Missed</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div>
              <Label htmlFor="severity">Severity</Label>
              <Select
                value={formData.severity}
                onValueChange={(value) => handleInputChange('severity', value)}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="low">Low</SelectItem>
                  <SelectItem value="medium">Medium</SelectItem>
                  <SelectItem value="high">High</SelectItem>
                  <SelectItem value="critical">Critical</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div>
              <Label htmlFor="emergencyMessage">Message</Label>
              <Input
                id="emergencyMessage"
                placeholder="Enter emergency message"
                value={formData.emergencyMessage}
                onChange={(e) => handleInputChange('emergencyMessage', e.target.value)}
              />
            </div>
          </div>
        );

      default:
        return null;
    }
  };

  const renderResult = () => {
    if (!result) return null;

    return (
      <Card className="mt-4">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Bell className="h-5 w-5" />
            Notification Results
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {result.map((notification: Notification, index: number) => (
              <div key={index} className="flex items-center justify-between p-3 border rounded-lg">
                <div className="flex items-center gap-3">
                  <Bell className="h-4 w-4 text-blue-500" />

                  <div>
                    <p className="font-medium">{notification.type || 'notification'}</p>
                    <p className="text-sm text-muted-foreground">{notification.message}</p>
                  </div>
                </div>

                <Badge
                  variant={notification.status === 'sent' ? 'default' : 'destructive'}
                >
                  {notification.status || 'pending'}
                </Badge>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    );
  };

  return (
    <div className={className}>
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Bell className="h-6 w-6" />
            Notification Manager
          </CardTitle>
          <CardDescription>
            Test and manage different types of notifications
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Notification Type Selection */}
          <div>
            <Label>Notification Type</Label>
            <div className="flex gap-2 mt-2">
              <Button
                variant={notificationType === 'appointment' ? 'default' : 'outline'}
                size="sm"
                onClick={() => setNotificationType('appointment')}
              >
                <MessageSquare className="h-4 w-4 mr-2" />
                Appointment
              </Button>
              <Button
                variant={notificationType === 'medication' ? 'default' : 'outline'}
                size="sm"
                onClick={() => setNotificationType('medication')}
              >
                <AlertTriangle className="h-4 w-4 mr-2" />
                Medication
              </Button>
              <Button
                variant={notificationType === 'emergency' ? 'default' : 'outline'}
                size="sm"
                onClick={() => setNotificationType('emergency')}
              >
                <AlertTriangle className="h-4 w-4 mr-2" />
                Emergency
              </Button>
            </div>
          </div>

          {/* Contact Information */}
          <div className="space-y-4">
            <h4 className="font-semibold">Contact Information</h4>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <Label htmlFor="phone">Phone Number (optional)</Label>
                <Input
                  id="phone"
                  type="tel"
                  placeholder="+1234567890"
                  value={formData.phone}
                  onChange={(e) => handleInputChange('phone', e.target.value)}
                />
              </div>
              <div>
                <Label htmlFor="email">Email (optional)</Label>
                <Input
                  id="email"
                  type="email"
                  placeholder="user@example.com"
                  value={formData.email}
                  onChange={(e) => handleInputChange('email', e.target.value)}
                />
              </div>
            </div>
          </div>

          {/* Dynamic Form */}
          <div className="space-y-4">
            <h4 className="font-semibold capitalize">{notificationType} Details</h4>
            {renderForm()}
          </div>

          {/* Submit Button */}
          <Button
            onClick={handleSubmit}
            disabled={loading}
            className="w-full"
          >
            {loading ? (
              <>
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                Sending...
              </>
            ) : (
              <>
                <Bell className="h-4 w-4 mr-2" />
                Send Notification
              </>
            )}
          </Button>

          {/* Error */}
          {error && (
            <Alert variant="destructive">
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          {/* Result */}
          {renderResult()}
        </CardContent>
      </Card>
    </div>
  );
};
