import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Clipboard } from 'lucide-react';

export function ClinicalWorkflow() {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Clinical Workflow</h1>
          <p className="text-gray-600 mt-1">Labs, prescriptions, and vitals management</p>
        </div>
      </div>

      <Card className="card-glass rounded-2xl">
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Clipboard className="w-5 h-5" />
            <span>Clinical Tasks</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-12">
            <Clipboard className="w-12 h-12 text-gray-400 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">Clinical Workflow</h3>
            <p className="text-gray-600">Clinical task management coming soon.</p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}