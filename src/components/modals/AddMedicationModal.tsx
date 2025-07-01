import React, { useState } from 'react';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Switch } from '@/components/ui/switch';
import { Plus, Pill } from 'lucide-react';

interface AddMedicationModalProps {
  trigger?: React.ReactNode;
  onAddMedication?: (medication: any) => void;
}

export function AddMedicationModal({ trigger, onAddMedication }: AddMedicationModalProps) {
  const [open, setOpen] = useState(false);
  const [formData, setFormData] = useState({
    name: '',
    dosage: '',
    frequency: '',
    timing: '',
    reminderEnabled: true,
    treatmentPlan: '',
    refillDate: ''
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const newMedication = {
      id: Date.now(),
      ...formData,
      nextRefill: formData.refillDate,
      pillsRemaining: 30 // Default value
    };
    
    onAddMedication?.(newMedication);
    setOpen(false);
    setFormData({
      name: '',
      dosage: '',
      frequency: '',
      timing: '',
      reminderEnabled: true,
      treatmentPlan: '',
      refillDate: ''
    });
  };

  const defaultTrigger = (
    <Button className="flex items-center gap-2">
      <Plus className="w-4 h-4" />
      Add Medication
    </Button>
  );

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        {trigger || defaultTrigger}
      </DialogTrigger>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Pill className="w-5 h-5 text-blue-600" />
            Add New Medication
          </DialogTitle>
        </DialogHeader>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="name">Medication Name</Label>
              <Input
                id="name"
                value={formData.name}
                onChange={(e) => setFormData(prev => ({ ...prev, name: e.target.value }))}
                placeholder="e.g., Aspirin"
                required
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="dosage">Dosage</Label>
              <Input
                id="dosage"
                value={formData.dosage}
                onChange={(e) => setFormData(prev => ({ ...prev, dosage: e.target.value }))}
                placeholder="e.g., 100mg"
                required
              />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="frequency">Frequency</Label>
              <Select value={formData.frequency} onValueChange={(value) => setFormData(prev => ({ ...prev, frequency: value }))}>
                <SelectTrigger>
                  <SelectValue placeholder="Select frequency" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="Once daily">Once daily</SelectItem>
                  <SelectItem value="Twice daily">Twice daily</SelectItem>
                  <SelectItem value="Three times daily">Three times daily</SelectItem>
                  <SelectItem value="As needed">As needed</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label htmlFor="timing">Time</Label>
              <Input
                id="timing"
                type="time"
                value={formData.timing}
                onChange={(e) => setFormData(prev => ({ ...prev, timing: e.target.value }))}
                required
              />
            </div>
          </div>

          <div className="space-y-2">
            <Label htmlFor="treatmentPlan">Treatment Plan</Label>
            <Input
              id="treatmentPlan"
              value={formData.treatmentPlan}
              onChange={(e) => setFormData(prev => ({ ...prev, treatmentPlan: e.target.value }))}
              placeholder="e.g., Cardiovascular Prevention"
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="refillDate">Next Refill Date</Label>
            <Input
              id="refillDate"
              type="date"
              value={formData.refillDate}
              onChange={(e) => setFormData(prev => ({ ...prev, refillDate: e.target.value }))}
              required
            />
          </div>

          <div className="flex items-center justify-between">
            <Label htmlFor="reminder" className="text-sm font-medium">
              Enable Reminders
            </Label>
            <Switch
              id="reminder"
              checked={formData.reminderEnabled}
              onCheckedChange={(checked) => setFormData(prev => ({ ...prev, reminderEnabled: checked }))}
            />
          </div>

          <div className="flex gap-2 pt-4">
            <Button type="button" variant="outline" onClick={() => setOpen(false)} className="flex-1">
              Cancel
            </Button>
            <Button type="submit" className="flex-1">
              Add Medication
            </Button>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  );
}