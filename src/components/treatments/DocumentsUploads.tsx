
import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { FileText, Upload, Download, Eye, Calendar } from 'lucide-react';

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
  const [uploadedFiles, setUploadedFiles] = useState(documents);

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric'
    });
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (files) {
      // In a real app, you'd upload to a server here
      Array.from(files).forEach((file, index) => {
        const newDoc = {
          id: `new-${Date.now()}-${index}`,
          name: file.name,
          type: file.type.includes('pdf') ? 'PDF' : 'Image',
          uploadDate: new Date().toISOString().split('T')[0],
          size: `${Math.round(file.size / 1024)} KB`
        };
        setUploadedFiles(prev => [...prev, newDoc]);
      });
    }
  };

  const getFileIcon = (type: string) => {
    return <FileText className="w-5 h-5 text-red-600" />;
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <FileText className="w-5 h-5 text-orange-600" />
          Documents & Uploads
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="mb-4">
          <label htmlFor="file-upload" className="cursor-pointer">
            <Button variant="outline" className="w-full flex items-center gap-2" asChild>
              <span>
                <Upload className="w-4 h-4" />
                Upload New Document
              </span>
            </Button>
          </label>
          <input
            id="file-upload"
            type="file"
            multiple
            accept=".pdf,.jpg,.jpeg,.png,.doc,.docx"
            onChange={handleFileUpload}
            className="hidden"
          />
          <p className="text-xs text-gray-500 mt-2 text-center">
            Support: PDF, Images, Word documents (Max 10MB each)
          </p>
        </div>

        <div className="space-y-3">
          {uploadedFiles.map((document) => (
            <div key={document.id} className="border rounded-lg p-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3 flex-1 min-w-0">
                  {getFileIcon(document.type)}
                  <div className="flex-1 min-w-0">
                    <p className="font-medium text-sm truncate">{document.name}</p>
                    <div className="flex items-center gap-3 text-xs text-gray-500">
                      <span>{document.type}</span>
                      <span>{document.size}</span>
                      <div className="flex items-center gap-1">
                        <Calendar className="w-3 h-3" />
                        <span>{formatDate(document.uploadDate)}</span>
                      </div>
                    </div>
                  </div>
                </div>
                <div className="flex items-center gap-1">
                  <Button variant="ghost" size="sm" className="p-1 h-8 w-8">
                    <Eye className="w-4 h-4" />
                  </Button>
                  <Button variant="ghost" size="sm" className="p-1 h-8 w-8">
                    <Download className="w-4 h-4" />
                  </Button>
                </div>
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
};

export default DocumentsUploads;
