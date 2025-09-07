
import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowLeft, Upload, FileText, Calendar } from 'lucide-react';
import { analyzeImage, savePredictionToSupabase, PredictionResult, HybridAnalysisResult } from '../services/predictionService';
import { useAuth } from '../contexts/AuthContext';
import { useToast } from '@/hooks/use-toast';

const UploadScreen = () => {
  const navigate = useNavigate();
  const { user } = useAuth();
  const { toast } = useToast();
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [scanType, setScanType] = useState('');
  const [scanDate, setScanDate] = useState('');
  const [notes, setNotes] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<any>(null);

  const scanTypes = ['CT Scan', 'MRI', 'X-Ray', 'EEG', 'Blood Test', 'Other'];

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedFile(file);
    }
  };

  const handleAnalyze = async () => {
    if (!selectedFile || !scanType) {
      toast({
        title: "Missing Information",
        description: "Please select a file and scan type before analyzing.",
        variant: "destructive"
      });
      return;
    }

    if (!user) {
      toast({
        title: "Authentication Required",
        description: "Please log in to analyze images.",
        variant: "destructive"
      });
      return;
    }

    setIsAnalyzing(true);

    try {
      // Map scan types to image types for AI analysis
      const imageTypeMap: Record<string, string> = {
        'CT Scan': 'ct_scan',
        'MRI': 'mri',
        'X-Ray': 'xray',
        'EEG': 'eeg',
        'Blood Test': 'skin_lesion', // Default fallback
        'Other': 'skin_lesion'
      };

      const imageType = imageTypeMap[scanType] || 'skin_lesion';

      // Perform real AI analysis
      const analysisResult = await analyzeImage(selectedFile, imageType as any);

      // Save to database - handle both result types
      const imageUrl = URL.createObjectURL(selectedFile);
      
      // Type guard to check if it's a PredictionResult
      const isPredictionResult = (result: PredictionResult | HybridAnalysisResult): result is PredictionResult => {
        return result && 'prediction' in result && 'confidence' in result;
      };

      if (isPredictionResult(analysisResult)) {
        await savePredictionToSupabase(analysisResult, imageUrl);
      }

      // Format result for display - handle both result types
      let formattedResult;
      
      if (isPredictionResult(analysisResult)) {
        // Handle PredictionResult format
        formattedResult = {
          summary: analysisResult.metadata?.findings || ['Analysis completed successfully.'],
          recommendation: analysisResult.metadata?.recommendations || ['Please consult with your healthcare provider for detailed interpretation.'],
          riskLevel: analysisResult.prediction === 'malignant' ? 'High' : 'Low',
          confidence: Math.round(analysisResult.confidence * 100),
          prediction: analysisResult.prediction,
          probabilities: analysisResult.probabilities,
          timestamp: analysisResult.timestamp
        };
      } else {
        // Handle HybridAnalysisResult format
        const hybridResult = analysisResult as any;
        formattedResult = {
          summary: hybridResult.findings || ['Analysis completed successfully.'],
          recommendation: hybridResult.recommendations || ['Please consult with your healthcare provider for detailed interpretation.'],
          riskLevel: hybridResult.riskLevel || 'Medium',
          confidence: Math.round((hybridResult.confidence || 0.8) * 100),
          prediction: hybridResult.overallAssessment || 'Analysis complete',
          probabilities: { benign: 0.5, malignant: 0.5 }, // Default for hybrid results
          timestamp: new Date().toISOString()
        };
      }

      setResult(formattedResult);

      toast({
        title: "Analysis Complete",
        description: "Your medical image has been analyzed successfully.",
      });

    } catch (error) {
      console.error('Error analyzing image:', error);
      toast({
        title: "Analysis Failed",
        description: "Failed to analyze the image. Please try again.",
        variant: "destructive"
      });
    } finally {
      setIsAnalyzing(false);
    }
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

              {/* AI Prediction Details */}
              <div className="grid grid-cols-2 gap-4">
                <div className="p-3 bg-blue-50 rounded-lg">
                  <span className="font-medium text-blue-800">AI Prediction:</span>
                  <p className="text-blue-600 font-bold capitalize">{result.prediction}</p>
                </div>
                <div className="p-3 bg-green-50 rounded-lg">
                  <span className="font-medium text-green-800">Confidence:</span>
                  <p className="text-green-600 font-bold">{result.confidence}%</p>
                </div>
              </div>

              {/* Risk Level */}
              <div className={`flex justify-between items-center p-3 rounded-lg ${
                result.riskLevel === 'High' ? 'bg-red-50' : 'bg-green-50'
              }`}>
                <span className="font-medium">Risk Level:</span>
                <span className={`font-bold ${
                  result.riskLevel === 'High' ? 'text-red-600' : 'text-green-600'
                }`}>
                  {result.riskLevel}
                </span>
              </div>

              {/* Probability Breakdown */}
              {result.probabilities && (
                <div className="p-3 bg-gray-50 rounded-lg">
                  <h4 className="font-medium text-gray-700 mb-2">Probability Breakdown:</h4>
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-sm">Benign:</span>
                      <span className="text-sm font-medium">
                        {Math.round(result.probabilities.benign * 100)}%
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm">Malignant:</span>
                      <span className="text-sm font-medium">
                        {Math.round(result.probabilities.malignant * 100)}%
                      </span>
                    </div>
                  </div>
                </div>
              )}

              {/* Timestamp */}
              {result.timestamp && (
                <div className="text-xs text-gray-500">
                  Analysis completed: {new Date(result.timestamp).toLocaleString()}
                </div>
              )}

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
