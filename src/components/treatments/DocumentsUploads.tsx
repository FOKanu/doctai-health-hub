
import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { FileText, Upload, Download, Eye } from 'lucide-react';
import { Button } from '@/components/ui/button';

interface Document {
  id: string;
  name: string;
  type: string;
  uploadDate: string;
  size: string;
}

interface DocumentsUploadsProps {
  documents: Document[];
}

const DocumentsUploads: React.FC<DocumentsUploadsProps> = ({ documents }) => {
  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric'
    });
  };

  const getFileIcon = (type: string) => {
    return <FileText className="w-5 h-5 text-blue-600" />;
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <FileText className="w-5 h-5 text-green-600" />
          Documents & Uploads
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {/* Upload Button */}
          <Button className="w-full flex items-center gap-2" variant="outline">
            <Upload className="w-4 h-4" />
            Upload New Document
          </Button>

          {/* Documents List */}
          <div className="space-y-3">
            {documents.map((document) => (
              <div key={document.id} className="flex items-center justify-between p-3 border rounded-lg">
                <div className="flex items-center gap-3">
                  {getFileIcon(document.type)}
                  <div>
                    <h4 className="font-medium text-sm">{document.name}</h4>
                    <p className="text-xs text-gray-500">
                      {document.type} • {document.size} • {formatDate(document.uploadDate)}
                    </p>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <Button variant="ghost" size="sm">
                    <Eye className="w-4 h-4" />
                  </Button>
                  <Button variant="ghost" size="sm">
                    <Download className="w-4 h-4" />
                  </Button>
                </div>
              </div>
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default DocumentsUploads;
