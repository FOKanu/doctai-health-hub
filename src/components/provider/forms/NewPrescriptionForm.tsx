import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { DialogClose, DialogFooter } from '@/components/ui/dialog';
import { useProviderStore } from '@/stores/providerStore';

interface NewPrescriptionFormProps {
  onSuccess: () => void;
}

export function NewPrescriptionForm({ onSuccess }: NewPrescriptionFormProps) {
  const { patients, addPrescription } = useProviderStore();
  const [formData, setFormData] = useState({
    patientId: '',
    medicationName: '',
    dosage: '',
    frequency: '',
    quantity: '',
    notes: ''
  });

  const commonMedications = [
    'Lisinopril', 'Metformin', 'Atorvastatin', 'Albuterol', 'Omeprazole', 'Amlodipine',
    'Hydrochlorothiazide', 'Levothyroxine', 'Simvastatin', 'Metoprolol'
  ];

  const frequencies = [
    'Once daily', 'Twice daily', 'Three times daily', 'Four times daily', 'As needed'
  ];

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    const selectedPatient = patients.find(p => p.id === formData.patientId);
    if (!selectedPatient) return;

    const today = new Date();
    const expirationDate = new Date(today.getTime() + (365 * 24 * 60 * 60 * 1000));
    const renewalDate = new Date(today.getTime() + (30 * 24 * 60 * 60 * 1000));

    addPrescription({
      patientId: formData.patientId,
      patientName: selectedPatient.name,
      medicationName: formData.medicationName,
      dosage: formData.dosage,
      frequency: formData.frequency,
      quantity: parseInt(formData.quantity),
      refillsRemaining: 5,
      prescribedDate: today.toISOString().split('T')[0],
      expirationDate: expirationDate.toISOString().split('T')[0],
      renewalDate: renewalDate.toISOString().split('T')[0],
      status: 'Active',
      prescribedBy: 'Dr. Smith',
      notes: formData.notes || undefined
    });

    onSuccess();
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div className="space-y-2">
        <Label htmlFor="patient">Patient</Label>
        <Select value={formData.patientId} onValueChange={(value) => setFormData(prev => ({ ...prev, patientId: value }))}>
          <SelectTrigger>
            <SelectValue placeholder="Select a patient" />
          </SelectTrigger>
          <SelectContent>
            {patients.map((patient) => (
              <SelectItem key={patient.id} value={patient.id}>
                {patient.name} - {patient.mrn}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      <div className="space-y-2">
        <Label htmlFor="medicationName">Medication</Label>
        <Select value={formData.medicationName} onValueChange={(value) => setFormData(prev => ({ ...prev, medicationName: value }))}>
          <SelectTrigger>
            <SelectValue placeholder="Select medication" />
          </SelectTrigger>
          <SelectContent>
            {commonMedications.map((medication) => (
              <SelectItem key={medication} value={medication}>
                {medication}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div className="space-y-2">
          <Label htmlFor="dosage">Dosage</Label>
          <Input
            id="dosage"
            placeholder="e.g., 10mg"
            value={formData.dosage}
            onChange={(e) => setFormData(prev => ({ ...prev, dosage: e.target.value }))}
          />
        </div>

        <div className="space-y-2">
          <Label htmlFor="quantity">Quantity</Label>
          <Input
            id="quantity"
            type="number"
            placeholder="30"
            value={formData.quantity}
            onChange={(e) => setFormData(prev => ({ ...prev, quantity: e.target.value }))}
          />
        </div>
      </div>

      <div className="space-y-2">
        <Label htmlFor="frequency">Frequency</Label>
        <Select value={formData.frequency} onValueChange={(value) => setFormData(prev => ({ ...prev, frequency: value }))}>
          <SelectTrigger>
            <SelectValue placeholder="Select frequency" />
          </SelectTrigger>
          <SelectContent>
            {frequencies.map((frequency) => (
              <SelectItem key={frequency} value={frequency}>
                {frequency}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      <div className="space-y-2">
        <Label htmlFor="notes">Notes (Optional)</Label>
        <Textarea
          id="notes"
          placeholder="Instructions for patient..."
          value={formData.notes}
          onChange={(e) => setFormData(prev => ({ ...prev, notes: e.target.value }))}
        />
      </div>

      <DialogFooter>
        <DialogClose asChild>
          <Button type="button" variant="outline">Cancel</Button>
        </DialogClose>
        <DialogClose asChild>
          <Button 
            type="submit"
            disabled={!formData.patientId || !formData.medicationName || !formData.dosage || !formData.frequency || !formData.quantity}
          >
            Create Prescription
          </Button>
        </DialogClose>
      </DialogFooter>
    </form>
  );
}