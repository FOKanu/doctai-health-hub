
import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowLeft, Upload, FileText, Calendar } from 'lucide-react';

const UploadScreen = () => {
  const navigate = useNavigate();
  const [selectedFile, setSelectedFile] = useState(null);
  const [scanType, setScanType] = useState('');
  const [scanDate, setScanDate] = useState('');
  const [notes, setNotes] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState(null);

  const scanTypes = ['CT Scan', 'MRI', 'X-Ray', 'EEG', 'Blood Test', 'Other'];

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    setSelectedFile(file);
  };

  const handleAnalyze = () => {
    if (!selectedFile || !scanType) {
      alert('Please select a file and scan type');
      return;
    }

    setIsAnalyzing(true);
    // Simulate AI analysis
    setTimeout(() => {
      setResult({
        summary: 'Normal findings with no significant abnormalities detected.',
        recommendation: 'Continue routine monitoring. Follow up in 6 months.',
        riskLevel: 'Low',
        confidence: 92
      });
      setIsAnalyzing(false);
    }, 3000);
  };

  return (
    <div className="min-h-screen bg-gray-50 pb-20">
      {/* Header */}
      <div className="bg-white shadow-sm">
        <div className="flex items-center p-4">
          <button
            onClick={() => navigate('/patient/')}
            className="p-2 -ml-2 rounded-full hover:bg-gray-100"
          >
            <ArrowLeft className="w-6 h-6" />
          </button>
          <h1 className="text-xl font-semibold ml-2">Upload Medical Image</h1>
        </div>
      </div>

      <div className="p-4 space-y-6">
        {/* File Upload */}
        <div className="bg-white rounded-lg p-6 shadow-sm">
          <h2 className="text-lg font-semibold mb-4">Select Medical Image</h2>

          <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-blue-400 transition-colors">
            <Upload className="w-12 h-12 text-gray-400 mx-auto mb-4" />
            <p className="text-gray-600 mb-2">Upload CT, MRI, EEG, PDF, JPG, or PNG</p>
            <input
              type="file"
              accept=".ct,.mri,.eeg,.pdf,.jpg,.jpeg,.png,.dicom"
              onChange={handleFileSelect}
              className="hidden"
              id="file-upload"
            />
            <label
              htmlFor="file-upload"
              className="inline-block bg-blue-600 text-white px-6 py-2 rounded-lg cursor-pointer hover:bg-blue-700 transition-colors"
            >
              Choose File
            </label>
          </div>

          {selectedFile && (
            <div className="mt-4 p-3 bg-blue-50 rounded-lg">
              <p className="text-sm text-blue-800">
                <FileText className="w-4 h-4 inline mr-2" />
                {selectedFile.name} ({(selectedFile.size / 1024 / 1024).toFixed(2)} MB)
              </p>
            </div>
          )}
        </div>

        {/* Metadata Form */}
        <div className="bg-white rounded-lg p-6 shadow-sm">
          <h2 className="text-lg font-semibold mb-4">Image Details</h2>

          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Scan Type *
              </label>
              <select
                value={scanType}
                onChange={(e) => setScanType(e.target.value)}
                className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                <option value="">Select scan type</option>
                {scanTypes.map((type) => (
                  <option key={type} value={type}>{type}</option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                <Calendar className="w-4 h-4 inline mr-1" />
                Scan Date
              </label>
              <input
                type="date"
                value={scanDate}
                onChange={(e) => setScanDate(e.target.value)}
                className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Notes (Optional)
              </label>
              <textarea
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                placeholder="Any additional information about the scan..."
                rows={3}
                className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>
          </div>
        </div>

        {/* Analyze Button */}
        <button
          onClick={handleAnalyze}
          disabled={isAnalyzing || !selectedFile || !scanType}
          className="w-full bg-blue-600 text-white py-4 rounded-lg font-semibold hover:bg-blue-700 transition-colors disabled:bg-gray-300 disabled:cursor-not-allowed"
        >
          {isAnalyzing ? 'Analyzing...' : 'Analyze with AI'}
        </button>

        {/* Results */}
        {result && (
          <div className="bg-white rounded-lg p-6 shadow-sm">
            <h2 className="text-lg font-semibold mb-4 text-green-600">Analysis Complete</h2>

            <div className="space-y-4">
              <div>
                <h3 className="font-medium text-gray-700 mb-2">Summary:</h3>
                <p className="text-gray-600 text-sm">{result.summary}</p>
              </div>

              <div>
                <h3 className="font-medium text-gray-700 mb-2">Recommendation:</h3>
                <p className="text-gray-600 text-sm">{result.recommendation}</p>
              </div>

              <div className="flex justify-between items-center p-3 bg-green-50 rounded-lg">
                <span className="font-medium">Risk Level:</span>
                <span className="text-green-600 font-bold">{result.riskLevel}</span>
              </div>

              <div className="flex space-x-3">
                <button
                  onClick={() => navigate('/patient/specialists')}
                  className="flex-1 bg-blue-600 text-white py-3 rounded-lg font-medium hover:bg-blue-700 transition-colors"
                >
                  Get Specialist
                </button>
                <button
                  onClick={() => navigate('/patient/history')}
                  className="flex-1 bg-gray-100 text-gray-700 py-3 rounded-lg font-medium hover:bg-gray-200 transition-colors"
                >
                  Save to History
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default UploadScreen;
