import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { DialogClose, DialogFooter } from '@/components/ui/dialog';
import { useProviderStore } from '@/stores/providerStore';

interface NewLabTestFormProps {
  onSuccess: () => void;
}

export function NewLabTestForm({ onSuccess }: NewLabTestFormProps) {
  const { patients, addLabTest } = useProviderStore();
  const [formData, setFormData] = useState({
    patientId: '',
    testType: '',
    testName: '',
    priority: '',
    notes: ''
  });

  const labTests = {
    Blood: [
      'Complete Blood Count (CBC)',
      'Lipid Panel',
      'Comprehensive Metabolic Panel',
      'Hemoglobin A1C',
      'Thyroid Function Tests'
    ],
    Urine: ['Urinalysis', 'Urine Culture'],
    Other: ['Blood Culture', 'Stool Sample']
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    const selectedPatient = patients.find(p => p.id === formData.patientId);
    if (!selectedPatient) return;

    addLabTest({
      patientId: formData.patientId,
      patientName: selectedPatient.name,
      testType: formData.testType,
      testName: formData.testName,
      orderedBy: 'Dr. Smith',
      orderedDate: new Date().toISOString().split('T')[0],
      status: 'Pending',
      priority: formData.priority as 'Low' | 'Medium' | 'High' | 'Urgent',
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
        <Label htmlFor="testType">Test Type</Label>
        <Select value={formData.testType} onValueChange={(value) => setFormData(prev => ({ ...prev, testType: value, testName: '' }))}>
          <SelectTrigger>
            <SelectValue placeholder="Select test type" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="Blood">Blood</SelectItem>
            <SelectItem value="Urine">Urine</SelectItem>
            <SelectItem value="Other">Other</SelectItem>
          </SelectContent>
        </Select>
      </div>

      {formData.testType && (
        <div className="space-y-2">
          <Label htmlFor="testName">Test Name</Label>
          <Select value={formData.testName} onValueChange={(value) => setFormData(prev => ({ ...prev, testName: value }))}>
            <SelectTrigger>
              <SelectValue placeholder="Select test name" />
            </SelectTrigger>
            <SelectContent>
              {labTests[formData.testType as keyof typeof labTests]?.map((test) => (
                <SelectItem key={test} value={test}>
                  {test}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      )}

      <div className="space-y-2">
        <Label htmlFor="priority">Priority</Label>
        <Select value={formData.priority} onValueChange={(value) => setFormData(prev => ({ ...prev, priority: value }))}>
          <SelectTrigger>
            <SelectValue placeholder="Select priority" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="Low">Low</SelectItem>
            <SelectItem value="Medium">Medium</SelectItem>
            <SelectItem value="High">High</SelectItem>
            <SelectItem value="Urgent">Urgent</SelectItem>
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
            disabled={!formData.patientId || !formData.testType || !formData.testName || !formData.priority}
          >
            Order Lab Test
          </Button>
        </DialogClose>
      </DialogFooter>
    </form>
  );
}