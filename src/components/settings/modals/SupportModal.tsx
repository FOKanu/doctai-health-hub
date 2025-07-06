import React, { useState } from 'react';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { ScrollArea } from '@/components/ui/scroll-area';
import { HelpCircle, MessageCircle, Bug, Shield, FileText, ExternalLink } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';

interface SupportModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  type: 'help' | 'contact' | 'bug' | 'privacy' | 'terms';
}

export const SupportModal = ({ open, onOpenChange, type }: SupportModalProps) => {
  const [subject, setSubject] = useState('');
  const [message, setMessage] = useState('');
  const [category, setCategory] = useState('');
  const [email, setEmail] = useState('');
  const { toast } = useToast();

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    toast({
      title: "Message Sent",
      description: "We've received your message and will respond within 24 hours."
    });
    onOpenChange(false);
    setSubject('');
    setMessage('');
    setCategory('');
  };

  const getTitle = () => {
    switch (type) {
      case 'help': return 'Help Center';
      case 'contact': return 'Contact Support';
      case 'bug': return 'Report a Bug';
      case 'privacy': return 'Privacy Policy';
      case 'terms': return 'Terms of Service';
      default: return 'Support';
    }
  };

  const renderContent = () => {
    switch (type) {
      case 'help':
        return (
          <ScrollArea className="h-96">
            <div className="space-y-6">
              <div className="flex items-center gap-3 p-4 bg-medical-accent-light rounded-lg">
                <HelpCircle className="h-6 w-6 text-medical-accent" />
                <h3 className="font-medium">Frequently Asked Questions</h3>
              </div>

              <div className="space-y-4">
                <div className="border-l-4 border-medical-accent pl-4">
                  <h4 className="font-medium mb-2">How do I upload medical images?</h4>
                  <p className="text-sm text-muted-foreground">
                    Go to the Scan screen and tap the upload button. You can upload images from your camera roll or take a new photo directly.
                  </p>
                </div>

                <div className="border-l-4 border-medical-accent pl-4">
                  <h4 className="font-medium mb-2">How accurate are the AI predictions?</h4>
                  <p className="text-sm text-muted-foreground">
                    Our AI models are trained on medical datasets but should not replace professional medical advice. Always consult with healthcare providers for diagnosis and treatment.
                  </p>
                </div>

                <div className="border-l-4 border-medical-accent pl-4">
                  <h4 className="font-medium mb-2">Is my health data secure?</h4>
                  <p className="text-sm text-muted-foreground">
                    Yes, we use bank-level encryption and comply with HIPAA standards to protect your health information.
                  </p>
                </div>

                <div className="border-l-4 border-medical-accent pl-4">
                  <h4 className="font-medium mb-2">Can I export my health data?</h4>
                  <p className="text-sm text-muted-foreground">
                    Yes, you can export all your health data from Settings → Data & Storage → Export Data.
                  </p>
                </div>

                <div className="border-l-4 border-medical-accent pl-4">
                  <h4 className="font-medium mb-2">How do I schedule appointments?</h4>
                  <p className="text-sm text-muted-foreground">
                    Use the Appointments screen to view available specialists and book appointments directly through the app.
                  </p>
                </div>
              </div>

              <div className="flex justify-end">
                <Button onClick={() => onOpenChange(false)}>
                  Close
                </Button>
              </div>
            </div>
          </ScrollArea>
        );

      case 'contact':
        return (
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="flex items-center gap-3 p-4 bg-medical-accent-light rounded-lg">
              <MessageCircle className="h-6 w-6 text-medical-accent" />
              <div>
                <h3 className="font-medium">Get in Touch</h3>
                <p className="text-sm text-muted-foreground">We typically respond within 24 hours</p>
              </div>
            </div>

            <div className="space-y-2">
              <Label htmlFor="contact-email">Email Address</Label>
              <Input
                id="contact-email"
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="your@email.com"
                required
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="contact-category">Category</Label>
              <Select value={category} onValueChange={setCategory} required>
                <SelectTrigger>
                  <SelectValue placeholder="Select a category" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="general">General Inquiry</SelectItem>
                  <SelectItem value="technical">Technical Issue</SelectItem>
                  <SelectItem value="billing">Billing Question</SelectItem>
                  <SelectItem value="feature">Feature Request</SelectItem>
                  <SelectItem value="privacy">Privacy Concern</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label htmlFor="contact-subject">Subject</Label>
              <Input
                id="contact-subject"
                value={subject}
                onChange={(e) => setSubject(e.target.value)}
                placeholder="Brief description of your inquiry"
                required
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="contact-message">Message</Label>
              <Textarea
                id="contact-message"
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                placeholder="Please provide details about your inquiry..."
                rows={4}
                required
              />
            </div>

            <div className="flex justify-end space-x-2">
              <Button type="button" variant="outline" onClick={() => onOpenChange(false)}>
                Cancel
              </Button>
              <Button type="submit">
                Send Message
              </Button>
            </div>
          </form>
        );

      case 'bug':
        return (
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="flex items-center gap-3 p-4 bg-red-50 border border-red-200 rounded-lg">
              <Bug className="h-6 w-6 text-red-600" />
              <div>
                <h3 className="font-medium text-red-800">Report a Bug</h3>
                <p className="text-sm text-red-600">Help us improve the app by reporting issues</p>
              </div>
            </div>

            <div className="space-y-2">
              <Label htmlFor="bug-subject">What went wrong?</Label>
              <Input
                id="bug-subject"
                value={subject}
                onChange={(e) => setSubject(e.target.value)}
                placeholder="Brief description of the bug"
                required
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="bug-category">Bug Category</Label>
              <Select value={category} onValueChange={setCategory} required>
                <SelectTrigger>
                  <SelectValue placeholder="Select bug type" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="crash">App Crash</SelectItem>
                  <SelectItem value="ui">UI Issue</SelectItem>
                  <SelectItem value="performance">Performance Problem</SelectItem>
                  <SelectItem value="sync">Data Sync Issue</SelectItem>
                  <SelectItem value="login">Login Problem</SelectItem>
                  <SelectItem value="other">Other</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label htmlFor="bug-details">Steps to Reproduce</Label>
              <Textarea
                id="bug-details"
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                placeholder="1. Go to...&#10;2. Click on...&#10;3. The bug occurs when..."
                rows={5}
                required
              />
            </div>

            <div className="flex justify-end space-x-2">
              <Button type="button" variant="outline" onClick={() => onOpenChange(false)}>
                Cancel
              </Button>
              <Button type="submit">
                Report Bug
              </Button>
            </div>
          </form>
        );

      case 'privacy':
        return (
          <ScrollArea className="h-96">
            <div className="space-y-4">
              <div className="flex items-center gap-3 p-4 bg-medical-accent-light rounded-lg">
                <Shield className="h-6 w-6 text-medical-accent" />
                <h3 className="font-medium">Privacy Policy</h3>
              </div>

              <div className="space-y-6 text-sm">
                <section>
                  <h4 className="font-medium mb-2">Data Collection</h4>
                  <p className="text-muted-foreground">
                    We collect only the health data you provide and technical information necessary to operate the service. This includes medical images, health records, and usage analytics.
                  </p>
                </section>

                <section>
                  <h4 className="font-medium mb-2">Data Usage</h4>
                  <p className="text-muted-foreground">
                    Your health data is used solely for providing medical insights and improving our AI models. We do not sell or share your personal health information with third parties.
                  </p>
                </section>

                <section>
                  <h4 className="font-medium mb-2">Data Security</h4>
                  <p className="text-muted-foreground">
                    All data is encrypted in transit and at rest. We comply with HIPAA standards and use bank-level security measures to protect your information.
                  </p>
                </section>

                <section>
                  <h4 className="font-medium mb-2">Your Rights</h4>
                  <p className="text-muted-foreground">
                    You have the right to access, modify, or delete your health data at any time. You can export your data or request account deletion through the app settings.
                  </p>
                </section>

                <section>
                  <h4 className="font-medium mb-2">Contact Us</h4>
                  <p className="text-muted-foreground">
                    If you have questions about our privacy practices, please contact our privacy team at privacy@doctai.com.
                  </p>
                </section>
              </div>

              <div className="flex justify-between items-center pt-4 border-t">
                <Button variant="outline" onClick={() => window.open('https://doctai.com/privacy', '_blank')}>
                  <ExternalLink className="h-4 w-4 mr-2" />
                  Full Policy
                </Button>
                <Button onClick={() => onOpenChange(false)}>
                  Close
                </Button>
              </div>
            </div>
          </ScrollArea>
        );

      case 'terms':
        return (
          <ScrollArea className="h-96">
            <div className="space-y-4">
              <div className="flex items-center gap-3 p-4 bg-medical-accent-light rounded-lg">
                <FileText className="h-6 w-6 text-medical-accent" />
                <h3 className="font-medium">Terms of Service</h3>
              </div>

              <div className="space-y-6 text-sm">
                <section>
                  <h4 className="font-medium mb-2">Acceptance of Terms</h4>
                  <p className="text-muted-foreground">
                    By using DoctAI, you agree to these terms of service and our privacy policy. These terms may be updated from time to time.
                  </p>
                </section>

                <section>
                  <h4 className="font-medium mb-2">Medical Disclaimer</h4>
                  <p className="text-muted-foreground">
                    DoctAI provides health information and AI-powered insights but is not a substitute for professional medical advice, diagnosis, or treatment. Always consult qualified healthcare providers.
                  </p>
                </section>

                <section>
                  <h4 className="font-medium mb-2">User Responsibilities</h4>
                  <p className="text-muted-foreground">
                    You are responsible for the accuracy of health information you provide and for maintaining the security of your account credentials.
                  </p>
                </section>

                <section>
                  <h4 className="font-medium mb-2">Service Availability</h4>
                  <p className="text-muted-foreground">
                    We strive to maintain high service availability but cannot guarantee uninterrupted access. Scheduled maintenance will be communicated in advance.
                  </p>
                </section>

                <section>
                  <h4 className="font-medium mb-2">Intellectual Property</h4>
                  <p className="text-muted-foreground">
                    DoctAI and its content are protected by intellectual property laws. You retain ownership of your health data but grant us license to process it for service provision.
                  </p>
                </section>
              </div>

              <div className="flex justify-between items-center pt-4 border-t">
                <Button variant="outline" onClick={() => window.open('https://doctai.com/terms', '_blank')}>
                  <ExternalLink className="h-4 w-4 mr-2" />
                  Full Terms
                </Button>
                <Button onClick={() => onOpenChange(false)}>
                  Close
                </Button>
              </div>
            </div>
          </ScrollArea>
        );

      default:
        return null;
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>{getTitle()}</DialogTitle>
        </DialogHeader>
        {renderContent()}
      </DialogContent>
    </Dialog>
  );
};