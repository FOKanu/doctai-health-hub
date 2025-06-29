
import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Stethoscope, Calendar, MapPin, Clock } from 'lucide-react';

interface Procedure {
  id: string;
  name: string;
  date: string;
  time: string;
  specialist: string;
  hospital: string;
  status: 'confirmed' | 'pending' | 'cancelled';
}

interface ProceduresListProps {
  procedures: Procedure[];
}

const ProceduresList: React.FC<ProceduresListProps> = ({ procedures }) => {
  const getStatusColor = (status: string) => {
    const colors = {
      confirmed: 'bg-green-100 text-green-800',
      pending: 'bg-yellow-100 text-yellow-800',
      cancelled: 'bg-red-100 text-red-800'
    };
    return colors[status as keyof typeof colors] || 'bg-gray-100 text-gray-800';
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      weekday: 'short',
      year: 'numeric',
      month: 'short',
      day: 'numeric'
    });
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Stethoscope className="w-5 h-5 text-purple-600" />
          Scheduled Procedures
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {procedures.map((procedure) => (
            <div key={procedure.id} className="border rounded-lg p-4">
              <div className="flex items-start justify-between mb-3">
                <h3 className="font-semibold">{procedure.name}</h3>
                <span className={`px-2 py-1 rounded-full text-xs font-medium capitalize ${getStatusColor(procedure.status)}`}>
                  {procedure.status}
                </span>
              </div>
              <div className="space-y-2 text-sm text-gray-600">
                <div className="flex items-center gap-2">
                  <Calendar className="w-4 h-4" />
                  <span>{formatDate(procedure.date)}</span>
                </div>
                <div className="flex items-center gap-2">
                  <Clock className="w-4 h-4" />
                  <span>{procedure.time}</span>
                </div>
                <div className="flex items-center gap-2">
                  <MapPin className="w-4 h-4" />
                  <span>{procedure.hospital}</span>
                </div>
                <p><span className="font-medium">Specialist:</span> {procedure.specialist}</p>
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
};

export default ProceduresList;
