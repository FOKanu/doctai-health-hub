
import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Bell, BellOff, Plus } from 'lucide-react';
import TreatmentOverview from './treatments/TreatmentOverview';
import MedicationsList from './treatments/MedicationsList';
import ProceduresList from './treatments/ProceduresList';
import NotesInstructions from './treatments/NotesInstructions';
import FollowUpAppointments from './treatments/FollowUpAppointments';
import DocumentsUploads from './treatments/DocumentsUploads';
import { treatmentData } from './treatments/treatmentData';

const TreatmentsScreen = () => {
  const [notifications, setNotifications] = useState(true);
  const [data, setData] = useState(treatmentData);

  const toggleNotifications = () => {
    setNotifications(!notifications);
  };

  const handleAddTreatment = () => {
    // TODO: Open add treatment dialog/modal
    console.log('Add treatment clicked');
  };

  const updateMedicationStatus = (medicationId: string, taken: boolean) => {
    setData(prev => ({
      ...prev,
      medications: prev.medications.map(med => 
        med.id === medicationId ? { ...med, takenToday: taken } : med
      )
    }));
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Treatment Plans</h1>
          <p className="text-gray-600">Manage your medications, procedures, and follow-up care</p>
        </div>
        <div className="flex items-center gap-3">
          <Button
            variant="outline"
            size="sm"
            onClick={toggleNotifications}
            className="flex items-center gap-2"
          >
            {notifications ? <Bell className="w-4 h-4" /> : <BellOff className="w-4 h-4" />}
            {notifications ? 'Notifications On' : 'Notifications Off'}
          </Button>
          <Button size="sm" className="flex items-center gap-2" onClick={handleAddTreatment}>
            <Plus className="w-4 h-4" />
            Add Treatment
          </Button>
        </div>
      </div>

      {/* Treatment Overview */}
      <TreatmentOverview overview={data.overview} />

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left Column */}
        <div className="space-y-6">
          <MedicationsList 
            medications={data.medications}
            onUpdateStatus={updateMedicationStatus}
          />
          <ProceduresList procedures={data.procedures} />
          <NotesInstructions notes={data.notes} instructions={data.instructions} />
        </div>

        {/* Right Column */}
        <div className="space-y-6">
          <FollowUpAppointments appointments={data.appointments} />
          <DocumentsUploads documents={data.documents} />
        </div>
      </div>
    </div>
  );
};

export default TreatmentsScreen;
