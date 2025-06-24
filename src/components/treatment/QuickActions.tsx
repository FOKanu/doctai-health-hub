
import React from 'react';
import { Button } from '@/components/ui/button';
import { 
  Calendar, 
  Pharmacy, 
  Pill, 
  Plus,
  Phone,
  MapPin
} from 'lucide-react';

interface QuickActionsProps {
  onPharmacySearch: () => void;
}

const QuickActions: React.FC<QuickActionsProps> = ({ onPharmacySearch }) => {
  const handleBookAppointment = () => {
    // This would typically navigate to the appointments screen or open a booking modal
    console.log('Book appointment clicked');
  };

  const handleUpdateMedication = () => {
    // This would open a medication logging modal
    console.log('Update medication log clicked');
  };

  const handleEmergencyCall = () => {
    // This would dial emergency services
    window.open('tel:112', '_self');
  };

  return (
    <div className="fixed bottom-6 right-6 z-40">
      <div className="flex flex-col gap-3">
        {/* Emergency Call Button */}
        <Button
          onClick={handleEmergencyCall}
          className="bg-red-600 hover:bg-red-700 text-white rounded-full w-12 h-12 p-0 shadow-lg"
          title="Emergency Call"
        >
          <Phone className="w-5 h-5" />
        </Button>

        {/* Main Action Buttons */}
        <div className="bg-white rounded-2xl shadow-lg border p-2 space-y-2">
          <Button
            onClick={handleBookAppointment}
            className="flex items-center gap-2 w-full justify-start bg-blue-600 hover:bg-blue-700 text-white"
            size="sm"
          >
            <Calendar className="w-4 h-4" />
            Book Appointment
          </Button>

          <Button
            onClick={onPharmacySearch}
            className="flex items-center gap-2 w-full justify-start bg-green-600 hover:bg-green-700 text-white"
            size="sm"
          >
            <Pharmacy className="w-4 h-4" />
            Find Pharmacy
          </Button>

          <Button
            onClick={handleUpdateMedication}
            className="flex items-center gap-2 w-full justify-start bg-purple-600 hover:bg-purple-700 text-white"
            size="sm"
          >
            <Pill className="w-4 h-4" />
            Log Medication
          </Button>
        </div>

        {/* Floating Action Button */}
        <Button
          onClick={() => document.querySelector('input')?.focus()}
          className="bg-blue-600 hover:bg-blue-700 text-white rounded-full w-14 h-14 p-0 shadow-lg"
          title="Quick Search"
        >
          <Plus className="w-6 h-6" />
        </Button>
      </div>
    </div>
  );
};

export default QuickActions;
