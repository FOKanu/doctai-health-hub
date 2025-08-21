import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Brain } from 'lucide-react';

export function Neurology() {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Neurology</h1>
          <p className="text-gray-600 mt-1">Neurological specialty tools and insights</p>
        </div>
      </div>

      <Card className="card-glass rounded-2xl">
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Brain className="w-5 h-5 text-purple-600" />
            <span>Neurology Dashboard</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-12">
            <Brain className="w-12 h-12 text-purple-600 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">Neurology Tools</h3>
            <p className="text-gray-600">Neurological specialty tools coming soon.</p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}