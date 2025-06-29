
import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Pill, CheckCircle2, Circle } from 'lucide-react';

interface Medication {
  id: string;
  name: string;
  dosage: string;
  frequency: string;
  duration: string;
  color: string;
  takenToday: boolean;
  adherenceRate: number;
}

interface MedicationsListProps {
  medications: Medication[];
  onUpdateStatus: (medicationId: string, taken: boolean) => void;
}

const MedicationsList: React.FC<MedicationsListProps> = ({ medications, onUpdateStatus }) => {
  const getColorClass = (color: string) => {
    const colors: { [key: string]: string } = {
      blue: 'bg-blue-100 text-blue-800',
      green: 'bg-green-100 text-green-800',
      red: 'bg-red-100 text-red-800',
      yellow: 'bg-yellow-100 text-yellow-800',
      purple: 'bg-purple-100 text-purple-800'
    };
    return colors[color] || 'bg-gray-100 text-gray-800';
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Pill className="w-5 h-5 text-green-600" />
          Prescribed Medications
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {medications.map((medication) => (
            <div key={medication.id} className="border rounded-lg p-4">
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center gap-3 mb-2">
                    <h3 className="font-semibold">{medication.name}</h3>
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${getColorClass(medication.color)}`}>
                      {medication.dosage}
                    </span>
                  </div>
                  <div className="text-sm text-gray-600 space-y-1">
                    <p><span className="font-medium">Frequency:</span> {medication.frequency}</p>
                    <p><span className="font-medium">Duration:</span> {medication.duration}</p>
                    <div className="flex items-center gap-2">
                      <span className="font-medium">Adherence:</span>
                      <div className="flex-1 bg-gray-200 rounded-full h-2 max-w-32">
                        <div 
                          className="bg-green-500 h-2 rounded-full" 
                          style={{ width: `${medication.adherenceRate}%` }}
                        ></div>
                      </div>
                      <span className="text-xs text-gray-500">{medication.adherenceRate}%</span>
                    </div>
                  </div>
                </div>
                <button
                  onClick={() => onUpdateStatus(medication.id, !medication.takenToday)}
                  className="flex items-center gap-2 text-sm hover:bg-gray-50 p-2 rounded"
                >
                  {medication.takenToday ? (
                    <CheckCircle2 className="w-5 h-5 text-green-500" />
                  ) : (
                    <Circle className="w-5 h-5 text-gray-400" />
                  )}
                  <span className={medication.takenToday ? 'text-green-600' : 'text-gray-500'}>
                    {medication.takenToday ? 'Taken today' : 'Mark as taken'}
                  </span>
                </button>
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
};

export default MedicationsList;
