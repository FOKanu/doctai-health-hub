import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { DialogClose, DialogFooter } from '@/components/ui/dialog';
import { useProviderStore } from '@/stores/providerStore';

interface NewOrderFormProps {
  onSuccess: () => void;
}

export function NewOrderForm({ onSuccess }: NewOrderFormProps) {
  const { patients, addOrder } = useProviderStore();
  const [formData, setFormData] = useState({
    patientId: '',
    type: '',
    notes: ''
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    const selectedPatient = patients.find(p => p.id === formData.patientId);
    if (!selectedPatient) return;

    addOrder({
      patientId: formData.patientId,
      patientName: selectedPatient.name,
      type: formData.type as 'Lab' | 'Imaging' | 'Medication' | 'Procedure',
      status: 'Pending',
      orderedBy: 'Dr. Smith',
      orderedDate: new Date().toISOString().split('T')[0],
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
        <Label htmlFor="type">Order Type</Label>
        <Select value={formData.type} onValueChange={(value) => setFormData(prev => ({ ...prev, type: value }))}>
          <SelectTrigger>
            <SelectValue placeholder="Select order type" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="Lab">Lab</SelectItem>
            <SelectItem value="Imaging">Imaging</SelectItem>
            <SelectItem value="Medication">Medication</SelectItem>
            <SelectItem value="Procedure">Procedure</SelectItem>
          </SelectContent>
        </Select>
      </div>

      <div className="space-y-2">
        <Label htmlFor="notes">Notes (Optional)</Label>
        <Textarea
          id="notes"
          placeholder="Additional notes or instructions..."
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
            disabled={!formData.patientId || !formData.type}
          >
            Create Order
          </Button>
        </DialogClose>
      </DialogFooter>
    </form>
  );
}