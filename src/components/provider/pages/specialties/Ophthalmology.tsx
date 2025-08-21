import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Eye } from 'lucide-react';

export function Ophthalmology() {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Ophthalmology</h1>
          <p className="text-gray-600 mt-1">Vision and eye care specialty tools</p>
        </div>
      </div>

      <Card className="card-glass rounded-2xl">
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Eye className="w-5 h-5 text-blue-600" />
            <span>Ophthalmology Dashboard</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-12">
            <Eye className="w-12 h-12 text-blue-600 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">Ophthalmology Tools</h3>
            <p className="text-gray-600">Eye care specialty tools coming soon.</p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}