import React, { useState } from 'react';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardContent } from '@/components/ui/card';
import { Smartphone, Mail, Shield } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';

interface TwoFactorModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export const TwoFactorModal = ({ open, onOpenChange }: TwoFactorModalProps) => {
  const [is2FAEnabled, setIs2FAEnabled] = useState(false);
  const [selectedMethod, setSelectedMethod] = useState<'phone' | 'email' | null>(null);
  const [phoneNumber, setPhoneNumber] = useState('');
  const [email, setEmail] = useState('');
  const [showQR, setShowQR] = useState(false);
  const { toast } = useToast();

  const handleEnable2FA = () => {
    if (!selectedMethod) {
      toast({
        title: "Error",
        description: "Please select a verification method",
        variant: "destructive"
      });
      return;
    }

    if (selectedMethod === 'phone' && !phoneNumber) {
      toast({
        title: "Error", 
        description: "Please enter your phone number",
        variant: "destructive"
      });
      return;
    }

    if (selectedMethod === 'email' && !email) {
      toast({
        title: "Error",
        description: "Please enter your email address", 
        variant: "destructive"
      });
      return;
    }

    setShowQR(true);
  };

  const handleConfirm2FA = () => {
    setIs2FAEnabled(true);
    toast({
      title: "Success",
      description: "Two-factor authentication enabled successfully"
    });
    onOpenChange(false);
  };

  const handleDisable2FA = () => {
    setIs2FAEnabled(false);
    setSelectedMethod(null);
    setShowQR(false);
    toast({
      title: "Success",
      description: "Two-factor authentication disabled"
    });
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Shield className="h-5 w-5" />
            Two-Factor Authentication
          </DialogTitle>
        </DialogHeader>

        {!is2FAEnabled ? (
          <div className="space-y-4">
            <p className="text-sm text-muted-foreground">
              Add an extra layer of security to your account by enabling two-factor authentication.
            </p>

            {!showQR ? (
              <>
                <div className="space-y-3">
                  <Label>Choose verification method:</Label>
                  
                  <Card 
                    className={`cursor-pointer transition-colors ${selectedMethod === 'phone' ? 'ring-2 ring-medical-accent' : ''}`}
                    onClick={() => setSelectedMethod('phone')}
                  >
                    <CardContent className="flex items-center space-x-3 p-3">
                      <Smartphone className="h-5 w-5 text-medical-accent" />
                      <div>
                        <p className="font-medium">SMS</p>
                        <p className="text-xs text-muted-foreground">Receive codes via text message</p>
                      </div>
                    </CardContent>
                  </Card>

                  <Card 
                    className={`cursor-pointer transition-colors ${selectedMethod === 'email' ? 'ring-2 ring-medical-accent' : ''}`}
                    onClick={() => setSelectedMethod('email')}
                  >
                    <CardContent className="flex items-center space-x-3 p-3">
                      <Mail className="h-5 w-5 text-medical-accent" />
                      <div>
                        <p className="font-medium">Email</p>
                        <p className="text-xs text-muted-foreground">Receive codes via email</p>
                      </div>
                    </CardContent>
                  </Card>
                </div>

                {selectedMethod === 'phone' && (
                  <div className="space-y-2">
                    <Label htmlFor="phone">Phone Number</Label>
                    <Input
                      id="phone"
                      type="tel"
                      placeholder="+1 (555) 123-4567"
                      value={phoneNumber}
                      onChange={(e) => setPhoneNumber(e.target.value)}
                    />
                  </div>
                )}

                {selectedMethod === 'email' && (
                  <div className="space-y-2">
                    <Label htmlFor="email">Email Address</Label>
                    <Input
                      id="email"
                      type="email"
                      placeholder="your@email.com"
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                    />
                  </div>
                )}

                <div className="flex justify-end space-x-2 pt-4">
                  <Button variant="outline" onClick={() => onOpenChange(false)}>
                    Cancel
                  </Button>
                  <Button onClick={handleEnable2FA}>
                    Continue
                  </Button>
                </div>
              </>
            ) : (
              <div className="space-y-4 text-center">
                <div className="mx-auto w-32 h-32 bg-medical-accent-light rounded-lg flex items-center justify-center">
                  <div className="text-xs font-mono">QR CODE</div>
                </div>
                <p className="text-sm text-muted-foreground">
                  Scan this QR code with your authenticator app or use the verification code sent to your {selectedMethod}.
                </p>
                <div className="flex justify-end space-x-2">
                  <Button variant="outline" onClick={() => setShowQR(false)}>
                    Back
                  </Button>
                  <Button onClick={handleConfirm2FA}>
                    Confirm Setup
                  </Button>
                </div>
              </div>
            )}
          </div>
        ) : (
          <div className="space-y-4">
            <div className="flex items-center justify-center p-4 bg-medical-accent-light rounded-lg">
              <Shield className="h-8 w-8 text-medical-accent mr-2" />
              <span className="text-medical-accent font-medium">2FA Enabled</span>
            </div>
            <p className="text-sm text-muted-foreground text-center">
              Your account is protected with two-factor authentication.
            </p>
            <div className="flex justify-end space-x-2">
              <Button variant="outline" onClick={() => onOpenChange(false)}>
                Close
              </Button>
              <Button variant="destructive" onClick={handleDisable2FA}>
                Disable 2FA
              </Button>
            </div>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
};