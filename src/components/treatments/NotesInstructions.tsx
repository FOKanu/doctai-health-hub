
import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { FileText, AlertCircle } from 'lucide-react';

interface NotesInstructionsProps {
  notes: string[];
  instructions: string[];
}

const NotesInstructions: React.FC<NotesInstructionsProps> = ({ notes, instructions }) => {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <FileText className="w-5 h-5 text-blue-600" />
          Notes & Instructions
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Doctor's Notes */}
        <div>
          <h3 className="font-medium mb-3 text-gray-900">Doctor's Notes</h3>
          <div className="space-y-2">
            {notes.map((note, index) => (
              <div key={index} className="flex items-start gap-2 p-3 bg-blue-50 rounded-lg">
                <FileText className="w-4 h-4 text-blue-600 mt-0.5 flex-shrink-0" />
                <p className="text-sm text-gray-700">{note}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Patient Instructions */}
        <div>
          <h3 className="font-medium mb-3 text-gray-900">Patient Instructions</h3>
          <div className="space-y-2">
            {instructions.map((instruction, index) => (
              <div key={index} className="flex items-start gap-2 p-3 bg-amber-50 rounded-lg">
                <AlertCircle className="w-4 h-4 text-amber-600 mt-0.5 flex-shrink-0" />
                <p className="text-sm text-gray-700">{instruction}</p>
              </div>
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default NotesInstructions;
