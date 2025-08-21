import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Bone } from 'lucide-react';

export function Orthopedics() {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Orthopedics</h1>
          <p className="text-gray-600 mt-1">Musculoskeletal specialty tools and insights</p>
        </div>
      </div>

      <Card className="card-glass rounded-2xl">
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Bone className="w-5 h-5 text-orange-600" />
            <span>Orthopedics Dashboard</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-12">
            <Bone className="w-12 h-12 text-orange-600 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">Orthopedics Tools</h3>
            <p className="text-gray-600">Musculoskeletal specialty tools coming soon.</p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}