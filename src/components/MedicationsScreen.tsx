
import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowLeft, Plus, Edit, Trash2, Bell, BellOff, Pill } from 'lucide-react';

const MedicationsScreen = () => {
  const navigate = useNavigate();
  const [medications, setMedications] = useState([
    {
      id: 1,
      name: 'Aspirin',
      dosage: '100mg',
      frequency: 'Once daily',
      timing: '08:00',
      reminderEnabled: true,
      treatmentPlan: 'Cardiovascular Prevention',
      nextRefill: '2024-07-15',
      pillsRemaining: 15
    },
    {
      id: 2,
      name: 'Vitamin D3',
      dosage: '1000 IU',
      frequency: 'Daily',
      timing: '09:00',
      reminderEnabled: true,
      treatmentPlan: 'General Health',
      nextRefill: '2024-08-01',
      pillsRemaining: 45
    },
    {
      id: 3,
      name: 'Lisinopril',
      dosage: '10mg',
      frequency: 'Once daily',
      timing: '20:00',
      reminderEnabled: false,
      treatmentPlan: 'Hypertension Management',
      nextRefill: '2024-06-20',
      pillsRemaining: 3
    }
  ]);

  const toggleReminder = (id: number) => {
    setMedications(meds => 
      meds.map(med => 
        med.id === id ? { ...med, reminderEnabled: !med.reminderEnabled } : med
      )
    );
  };

  const deleteMedication = (id: number) => {
    if (confirm('Are you sure you want to delete this medication?')) {
      setMedications(meds => meds.filter(med => med.id !== id));
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Medications</h1>
          <p className="text-gray-600">Manage your medications and reminders</p>
        </div>
        <button className="bg-blue-600 text-white px-4 py-2 rounded-lg font-semibold hover:bg-blue-700 transition-colors flex items-center gap-2">
          <Plus className="w-5 h-5" />
          Add Medication
        </button>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-white rounded-lg p-6 shadow-sm border">
          <div className="text-2xl font-bold text-blue-600 mb-2">{medications.length}</div>
          <div className="text-sm text-gray-500">Active Medications</div>
        </div>
        <div className="bg-white rounded-lg p-6 shadow-sm border">
          <div className="text-2xl font-bold text-green-600 mb-2">
            {medications.filter(m => m.reminderEnabled).length}
          </div>
          <div className="text-sm text-gray-500">Reminders Enabled</div>
        </div>
        <div className="bg-white rounded-lg p-6 shadow-sm border">
          <div className="text-2xl font-bold text-orange-600 mb-2">
            {medications.filter(m => m.pillsRemaining < 10).length}
          </div>
          <div className="text-sm text-gray-500">Low Stock Alerts</div>
        </div>
      </div>

      {/* Medications List */}
      <div className="space-y-4">
        <h2 className="text-lg font-semibold text-gray-800">Your Medications</h2>
        
        {medications.map((medication) => (
          <div key={medication.id} className="bg-white rounded-lg shadow-sm p-6 border">
            <div className="flex items-start justify-between mb-4">
              <div className="flex items-start space-x-4">
                <div className="p-3 bg-blue-100 rounded-lg">
                  <Pill className="w-6 h-6 text-blue-600" />
                </div>
                <div>
                  <h3 className="font-semibold text-gray-800 text-lg">{medication.name}</h3>
                  <p className="text-gray-600">{medication.dosage} • {medication.frequency}</p>
                  <p className="text-sm text-blue-600 mt-1">{medication.treatmentPlan}</p>
                </div>
              </div>
              
              <div className="flex items-center space-x-2">
                <button
                  onClick={() => toggleReminder(medication.id)}
                  className={`p-2 rounded-full transition-colors ${
                    medication.reminderEnabled 
                      ? 'bg-blue-100 text-blue-600 hover:bg-blue-200' 
                      : 'bg-gray-100 text-gray-400 hover:bg-gray-200'
                  }`}
                >
                  {medication.reminderEnabled ? <Bell className="w-5 h-5" /> : <BellOff className="w-5 h-5" />}
                </button>
                <button className="p-2 rounded-full bg-gray-100 text-gray-600 hover:bg-gray-200 transition-colors">
                  <Edit className="w-5 h-5" />
                </button>
                <button 
                  onClick={() => deleteMedication(medication.id)}
                  className="p-2 rounded-full bg-red-100 text-red-600 hover:bg-red-200 transition-colors"
                >
                  <Trash2 className="w-5 h-5" />
                </button>
              </div>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div>
                <span className="text-gray-500 block">Next dose:</span>
                <p className="font-medium">{medication.timing}</p>
              </div>
              <div>
                <span className="text-gray-500 block">Pills remaining:</span>
                <p className={`font-medium ${
                  medication.pillsRemaining < 10 ? 'text-red-600' : 'text-gray-800'
                }`}>
                  {medication.pillsRemaining}
                </p>
              </div>
              <div>
                <span className="text-gray-500 block">Next refill:</span>
                <p className="font-medium">{medication.nextRefill}</p>
              </div>
              <div>
                <span className="text-gray-500 block">Reminder:</span>
                <p className={`font-medium ${
                  medication.reminderEnabled ? 'text-green-600' : 'text-gray-400'
                }`}>
                  {medication.reminderEnabled ? 'Enabled' : 'Disabled'}
                </p>
              </div>
            </div>

            {medication.pillsRemaining < 10 && (
              <div className="mt-4 p-3 bg-red-50 border-l-4 border-red-400 rounded-r">
                <p className="text-sm text-red-700">
                  Low stock! Consider ordering a refill soon.
                </p>
                <button className="text-sm text-red-600 hover:underline mt-1">
                  Renew Prescription
                </button>
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Treatment Plans Link */}
      <div className="bg-white rounded-lg p-6 shadow-sm border">
        <h3 className="font-semibold text-gray-800 mb-3">Treatment Plans</h3>
        <p className="text-sm text-gray-600 mb-4">
          Link your medications to specific treatment plans for better organization and tracking.
        </p>
        <button 
          onClick={() => navigate('/treatments')}
          className="text-blue-600 text-sm hover:underline font-medium"
        >
          View All Treatment Plans →
        </button>
      </div>
    </div>
  );
};

export default MedicationsScreen;
