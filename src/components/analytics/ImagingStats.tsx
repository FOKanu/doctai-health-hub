
import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Camera, FileImage, Clock, CheckCircle, AlertCircle } from 'lucide-react';

interface ImagingStatsProps {
  dateRange: string;
}

export const ImagingStats: React.FC<ImagingStatsProps> = ({ dateRange }) => {
  const uploadData = [
    { type: 'Skin Scan', count: 15, analyzed: 13, pending: 2 },
    { type: 'CT Scan', count: 3, analyzed: 3, pending: 0 },
    { type: 'MRI', count: 2, analyzed: 2, pending: 0 },
    { type: 'EEG', count: 1, analyzed: 1, pending: 0 },
    { type: 'X-Ray', count: 4, analyzed: 4, pending: 0 },
  ];

  const statusData = [
    { status: 'Analyzed', count: 23, color: '#10b981' },
    { status: 'Under Review', count: 2, color: '#f59e0b' },
    { status: 'Pending', count: 0, color: '#6b7280' },
  ];

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Upload Statistics */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <FileImage className="w-5 h-5 text-blue-600" />
              Upload Statistics
            </CardTitle>
            <CardDescription>
              Medical imaging uploads by type over the last {dateRange === '7d' ? 'week' : 'month'}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={uploadData} layout="horizontal">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis dataKey="type" type="category" width={80} />
                <Tooltip 
                  formatter={(value, name) => [value, name === 'count' ? 'Total Uploads' : 'Analyzed']}
                />
                <Bar dataKey="count" fill="#3b82f6" name="count" />
                <Bar dataKey="analyzed" fill="#10b981" name="analyzed" />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Analysis Status */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Camera className="w-5 h-5 text-green-600" />
              Analysis Status
            </CardTitle>
            <CardDescription>
              Current status of uploaded medical images
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {statusData.map((item, index) => (
                <div key={index} className="flex items-center justify-between p-3 rounded-lg bg-gray-50">
                  <div className="flex items-center gap-3">
                    <div 
                      className="w-4 h-4 rounded-full"
                      style={{ backgroundColor: item.color }}
                    />
                    <span className="font-medium">{item.status}</span>
                  </div>
                  <span className="text-lg font-bold">{item.count}</span>
                </div>
              ))}
            </div>
            
            <div className="mt-6 p-4 bg-green-50 rounded-lg border border-green-200">
              <div className="flex items-center gap-2">
                <CheckCircle className="w-5 h-5 text-green-600" />
                <span className="font-medium text-green-900">Analysis Complete</span>
              </div>
              <p className="text-sm text-green-700 mt-1">
                92% of uploads processed successfully within 24 hours
              </p>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Recent Uploads */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Clock className="w-5 h-5 text-purple-600" />
            Recent Upload Activity
          </CardTitle>
          <CardDescription>
            Latest medical imaging uploads and their analysis status
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {[
              { id: 'IMG-2024-015', type: 'Skin Scan', date: '2024-02-14', status: 'analyzed', result: 'No anomalies detected' },
              { id: 'IMG-2024-014', type: 'CT Scan', date: '2024-02-13', status: 'analyzed', result: 'Normal findings' },
              { id: 'IMG-2024-013', type: 'Skin Scan', date: '2024-02-12', status: 'review', result: 'Requires specialist review' },
              { id: 'IMG-2024-012', type: 'MRI', date: '2024-02-11', status: 'analyzed', result: 'Within normal limits' },
              { id: 'IMG-2024-011', type: 'Skin Scan', date: '2024-02-10', status: 'analyzed', result: 'Monitor for changes' },
            ].map((upload) => (
              <div key={upload.id} className="flex items-center justify-between p-3 border rounded-lg">
                <div className="flex items-center gap-3">
                  <div className={`p-2 rounded-full ${
                    upload.status === 'analyzed' ? 'bg-green-100' : 'bg-orange-100'
                  }`}>
                    {upload.status === 'analyzed' ? (
                      <CheckCircle className="w-4 h-4 text-green-600" />
                    ) : (
                      <AlertCircle className="w-4 h-4 text-orange-600" />
                    )}
                  </div>
                  <div>
                    <div className="font-medium">{upload.id}</div>
                    <div className="text-sm text-gray-600">{upload.type} • {upload.date}</div>
                  </div>
                </div>
                <div className="text-right">
                  <div className={`text-sm font-medium ${
                    upload.status === 'analyzed' ? 'text-green-600' : 'text-orange-600'
                  }`}>
                    {upload.status === 'analyzed' ? 'Analyzed' : 'Under Review'}
                  </div>
                  <div className="text-xs text-gray-500">{upload.result}</div>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
