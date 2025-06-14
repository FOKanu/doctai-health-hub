
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

  const toggleReminder = (id) => {
    setMedications(meds => 
      meds.map(med => 
        med.id === id ? { ...med, reminderEnabled: !med.reminderEnabled } : med
      )
    );
  };

  const deleteMedication = (id) => {
    if (confirm('Are you sure you want to delete this medication?')) {
      setMedications(meds => meds.filter(med => med.id !== id));
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 pb-20">
      {/* Header */}
      <div className="bg-white shadow-sm">
        <div className="flex items-center justify-between p-4">
          <div className="flex items-center">
            <button
              onClick={() => navigate('/')}
              className="p-2 -ml-2 rounded-full hover:bg-gray-100"
            >
              <ArrowLeft className="w-6 h-6" />
            </button>
            <h1 className="text-xl font-semibold ml-2">Medications</h1>
          </div>
          <button className="p-2 bg-blue-600 text-white rounded-full hover:bg-blue-700 transition-colors">
            <Plus className="w-5 h-5" />
          </button>
        </div>
      </div>

      <div className="p-4">
        {/* Quick Stats */}
        <div className="grid grid-cols-3 gap-4 mb-6">
          <div className="bg-white rounded-lg p-4 text-center shadow-sm">
            <div className="text-2xl font-bold text-blue-600 mb-1">{medications.length}</div>
            <div className="text-xs text-gray-500">Active Meds</div>
          </div>
          <div className="bg-white rounded-lg p-4 text-center shadow-sm">
            <div className="text-2xl font-bold text-green-600 mb-1">
              {medications.filter(m => m.reminderEnabled).length}
            </div>
            <div className="text-xs text-gray-500">Reminders On</div>
          </div>
          <div className="bg-white rounded-lg p-4 text-center shadow-sm">
            <div className="text-2xl font-bold text-orange-600 mb-1">
              {medications.filter(m => m.pillsRemaining < 10).length}
            </div>
            <div className="text-xs text-gray-500">Low Stock</div>
          </div>
        </div>

        {/* Medications List */}
        <div className="space-y-4">
          <h2 className="text-lg font-semibold text-gray-800">Your Medications</h2>
          
          {medications.map((medication) => (
            <div key={medication.id} className="bg-white rounded-lg shadow-sm p-4">
              <div className="flex items-start justify-between mb-3">
                <div className="flex items-start space-x-3">
                  <div className="p-2 bg-blue-100 rounded-lg">
                    <Pill className="w-5 h-5 text-blue-600" />
                  </div>
                  <div>
                    <h3 className="font-semibold text-gray-800">{medication.name}</h3>
                    <p className="text-sm text-gray-600">{medication.dosage} • {medication.frequency}</p>
                    <p className="text-xs text-blue-600 mt-1">{medication.treatmentPlan}</p>
                  </div>
                </div>
                
                <div className="flex items-center space-x-2">
                  <button
                    onClick={() => toggleReminder(medication.id)}
                    className={`p-2 rounded-full ${
                      medication.reminderEnabled 
                        ? 'bg-blue-100 text-blue-600' 
                        : 'bg-gray-100 text-gray-400'
                    }`}
                  >
                    {medication.reminderEnabled ? <Bell className="w-4 h-4" /> : <BellOff className="w-4 h-4" />}
                  </button>
                  <button className="p-2 rounded-full bg-gray-100 text-gray-600 hover:bg-gray-200">
                    <Edit className="w-4 h-4" />
                  </button>
                  <button 
                    onClick={() => deleteMedication(medication.id)}
                    className="p-2 rounded-full bg-red-100 text-red-600 hover:bg-red-200"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <span className="text-gray-500">Next dose:</span>
                  <p className="font-medium">{medication.timing}</p>
                </div>
                <div>
                  <span className="text-gray-500">Pills remaining:</span>
                  <p className={`font-medium ${
                    medication.pillsRemaining < 10 ? 'text-red-600' : 'text-gray-800'
                  }`}>
                    {medication.pillsRemaining}
                  </p>
                </div>
                <div>
                  <span className="text-gray-500">Next refill:</span>
                  <p className="font-medium">{medication.nextRefill}</p>
                </div>
                <div>
                  <span className="text-gray-500">Reminder:</span>
                  <p className={`font-medium ${
                    medication.reminderEnabled ? 'text-green-600' : 'text-gray-400'
                  }`}>
                    {medication.reminderEnabled ? 'Enabled' : 'Disabled'}
                  </p>
                </div>
              </div>

              {medication.pillsRemaining < 10 && (
                <div className="mt-3 p-2 bg-red-50 border-l-4 border-red-400 rounded-r">
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

        {/* Add Medication Button */}
        <button className="w-full mt-6 bg-blue-600 text-white py-4 rounded-lg font-semibold hover:bg-blue-700 transition-colors flex items-center justify-center">
          <Plus className="w-5 h-5 mr-2" />
          Add New Medication
        </button>

        {/* Treatment Plans Link */}
        <div className="mt-4 bg-white rounded-lg p-4 shadow-sm">
          <h3 className="font-semibold text-gray-800 mb-2">Treatment Plans</h3>
          <p className="text-sm text-gray-600 mb-3">
            Link your medications to specific treatment plans for better organization.
          </p>
          <button className="text-blue-600 text-sm hover:underline">
            View All Treatment Plans →
          </button>
        </div>
      </div>
    </div>
  );
};

export default MedicationsScreen;
