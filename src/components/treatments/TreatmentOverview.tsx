
import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Calendar, User, MapPin } from 'lucide-react';

interface TreatmentOverviewProps {
  overview: {
    patientName: string;
    diagnosis: string;
    startDate: string;
    endDate: string;
    assignedDoctor: string;
    clinic: string;
  };
}

const TreatmentOverview: React.FC<TreatmentOverviewProps> = ({ overview }) => {
  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric'
    });
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <User className="w-5 h-5 text-blue-600" />
          Treatment Overview
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          <div>
            <p className="text-sm text-gray-500">Patient</p>
            <p className="font-medium">{overview.patientName}</p>
          </div>
          <div>
            <p className="text-sm text-gray-500">Diagnosis</p>
            <p className="font-medium">{overview.diagnosis}</p>
          </div>
          <div>
            <p className="text-sm text-gray-500">Duration</p>
            <div className="flex items-center gap-1 text-sm">
              <Calendar className="w-4 h-4 text-gray-400" />
              <span>{formatDate(overview.startDate)} - {formatDate(overview.endDate)}</span>
            </div>
          </div>
          <div>
            <p className="text-sm text-gray-500">Assigned Doctor</p>
            <p className="font-medium">{overview.assignedDoctor}</p>
          </div>
          <div>
            <p className="text-sm text-gray-500">Clinic</p>
            <div className="flex items-center gap-1">
              <MapPin className="w-4 h-4 text-gray-400" />
              <span className="font-medium">{overview.clinic}</span>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default TreatmentOverview;
