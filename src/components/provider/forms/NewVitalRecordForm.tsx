import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { DialogClose, DialogFooter } from '@/components/ui/dialog';
import { useProviderStore } from '@/stores/providerStore';

interface NewVitalRecordFormProps {
  onSuccess: () => void;
}

export function NewVitalRecordForm({ onSuccess }: NewVitalRecordFormProps) {
  const { patients, addVitalRecord } = useProviderStore();
  const [formData, setFormData] = useState({
    patientId: '',
    bloodPressureSystolic: '',
    bloodPressureDiastolic: '',
    heartRate: '',
    temperature: '',
    respiratoryRate: '',
    oxygenSaturation: '',
    weight: '',
    height: '',
    notes: ''
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    const selectedPatient = patients.find(p => p.id === formData.patientId);
    if (!selectedPatient) return;

    const bmi = formData.weight && formData.height 
      ? (parseFloat(formData.weight) / Math.pow(parseFloat(formData.height) / 12, 2) * 703) 
      : undefined;

    addVitalRecord({
      patientId: formData.patientId,
      patientName: selectedPatient.name,
      recordedDate: new Date().toISOString().split('T')[0],
      recordedBy: 'Nurse Johnson',
      bloodPressureSystolic: formData.bloodPressureSystolic ? parseInt(formData.bloodPressureSystolic) : undefined,
      bloodPressureDiastolic: formData.bloodPressureDiastolic ? parseInt(formData.bloodPressureDiastolic) : undefined,
      heartRate: formData.heartRate ? parseInt(formData.heartRate) : undefined,
      temperature: formData.temperature ? parseFloat(formData.temperature) : undefined,
      respiratoryRate: formData.respiratoryRate ? parseInt(formData.respiratoryRate) : undefined,
      oxygenSaturation: formData.oxygenSaturation ? parseInt(formData.oxygenSaturation) : undefined,
      weight: formData.weight ? parseInt(formData.weight) : undefined,
      height: formData.height ? parseInt(formData.height) : undefined,
      bmi: bmi ? Math.round(bmi * 10) / 10 : undefined,
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

      <div className="grid grid-cols-2 gap-4">
        <div className="space-y-2">
          <Label htmlFor="systolic">Blood Pressure (Systolic)</Label>
          <Input
            id="systolic"
            type="number"
            placeholder="120"
            value={formData.bloodPressureSystolic}
            onChange={(e) => setFormData(prev => ({ ...prev, bloodPressureSystolic: e.target.value }))}
          />
        </div>

        <div className="space-y-2">
          <Label htmlFor="diastolic">Blood Pressure (Diastolic)</Label>
          <Input
            id="diastolic"
            type="number"
            placeholder="80"
            value={formData.bloodPressureDiastolic}
            onChange={(e) => setFormData(prev => ({ ...prev, bloodPressureDiastolic: e.target.value }))}
          />
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div className="space-y-2">
          <Label htmlFor="heartRate">Heart Rate (bpm)</Label>
          <Input
            id="heartRate"
            type="number"
            placeholder="72"
            value={formData.heartRate}
            onChange={(e) => setFormData(prev => ({ ...prev, heartRate: e.target.value }))}
          />
        </div>

        <div className="space-y-2">
          <Label htmlFor="temperature">Temperature (°F)</Label>
          <Input
            id="temperature"
            type="number"
            step="0.1"
            placeholder="98.6"
            value={formData.temperature}
            onChange={(e) => setFormData(prev => ({ ...prev, temperature: e.target.value }))}
          />
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div className="space-y-2">
          <Label htmlFor="respiratoryRate">Respiratory Rate (/min)</Label>
          <Input
            id="respiratoryRate"
            type="number"
            placeholder="16"
            value={formData.respiratoryRate}
            onChange={(e) => setFormData(prev => ({ ...prev, respiratoryRate: e.target.value }))}
          />
        </div>

        <div className="space-y-2">
          <Label htmlFor="oxygenSaturation">Oxygen Saturation (%)</Label>
          <Input
            id="oxygenSaturation"
            type="number"
            placeholder="98"
            value={formData.oxygenSaturation}
            onChange={(e) => setFormData(prev => ({ ...prev, oxygenSaturation: e.target.value }))}
          />
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div className="space-y-2">
          <Label htmlFor="weight">Weight (lbs)</Label>
          <Input
            id="weight"
            type="number"
            placeholder="150"
            value={formData.weight}
            onChange={(e) => setFormData(prev => ({ ...prev, weight: e.target.value }))}
          />
        </div>

        <div className="space-y-2">
          <Label htmlFor="height">Height (inches)</Label>
          <Input
            id="height"
            type="number"
            placeholder="68"
            value={formData.height}
            onChange={(e) => setFormData(prev => ({ ...prev, height: e.target.value }))}
          />
        </div>
      </div>

      <div className="space-y-2">
        <Label htmlFor="notes">Notes (Optional)</Label>
        <Textarea
          id="notes"
          placeholder="Additional observations..."
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
            disabled={!formData.patientId}
          >
            Record Vitals
          </Button>
        </DialogClose>
      </DialogFooter>
    </form>
  );
}