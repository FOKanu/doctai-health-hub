
import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Camera, ArrowLeft, Flashlight, RefreshCw } from 'lucide-react';

const ScanScreen = () => {
  const navigate = useNavigate();
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [flashlightOn, setFlashlightOn] = useState(false);

  const handleCapture = () => {
    setIsAnalyzing(true);
    // Simulate AI analysis
    setTimeout(() => {
      setResult({
        riskLevel: 'Low',
        confidence: 87,
        notes: 'Benign-appearing lesion. Regular monitoring recommended.',
        color: 'text-green-600',
        bgColor: 'bg-green-50'
      });
      setIsAnalyzing(false);
    }, 3000);
  };

  const handleAddToMonitoring = () => {
    // This would typically save to a monitoring list
    alert('Added to weekly monitoring reminders!');
    navigate('/');
  };

  return (
    <div className="min-h-screen bg-black relative">
      {/* Header */}
      <div className="absolute top-0 left-0 right-0 z-10 flex justify-between items-center p-4">
        <button
          onClick={() => navigate('/')}
          className="p-2 bg-black bg-opacity-50 rounded-full text-white"
        >
          <ArrowLeft className="w-6 h-6" />
        </button>
        <h1 className="text-white font-semibold">Skin Lesion Scan</h1>
        <button
          onClick={() => setFlashlightOn(!flashlightOn)}
          className={`p-2 rounded-full ${flashlightOn ? 'bg-yellow-500' : 'bg-black bg-opacity-50'} text-white`}
        >
          <Flashlight className="w-6 h-6" />
        </button>
      </div>

      {/* Camera View Simulation */}
      <div className="relative h-screen">
        <div className="absolute inset-0 bg-gradient-to-br from-gray-800 to-gray-900 flex items-center justify-center">
          <div className="text-white text-center">
            <Camera className="w-16 h-16 mx-auto mb-4 opacity-50" />
            <p className="text-lg mb-2">Camera Preview</p>
            <p className="text-sm opacity-75">Position lesion in center frame</p>
          </div>
        </div>

        {/* Viewfinder */}
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="w-64 h-64 border-2 border-white border-dashed rounded-full opacity-75"></div>
        </div>

        {/* Capture Button */}
        <div className="absolute bottom-8 left-1/2 transform -translate-x-1/2">
          <button
            onClick={handleCapture}
            disabled={isAnalyzing}
            className="w-20 h-20 bg-white rounded-full shadow-lg flex items-center justify-center hover:scale-105 transition-transform duration-200 disabled:opacity-50"
          >
            {isAnalyzing ? (
              <RefreshCw className="w-8 h-8 text-blue-600 animate-spin" />
            ) : (
              <div className="w-16 h-16 bg-blue-600 rounded-full"></div>
            )}
          </button>
        </div>
      </div>

      {/* Analysis Result Overlay */}
      {result && (
        <div className="absolute inset-0 bg-black bg-opacity-75 flex items-end">
          <div className="w-full bg-white rounded-t-3xl p-6">
            <h2 className="text-xl font-bold mb-4">Analysis Complete</h2>
            
            <div className={`p-4 rounded-lg mb-4 ${result.bgColor}`}>
              <div className="flex justify-between items-center mb-2">
                <span className="font-semibold">Risk Level:</span>
                <span className={`font-bold ${result.color}`}>{result.riskLevel}</span>
              </div>
              <div className="flex justify-between items-center mb-2">
                <span className="font-semibold">Confidence:</span>
                <span className="font-bold">{result.confidence}%</span>
              </div>
            </div>

            <div className="mb-4">
              <h3 className="font-semibold mb-2">AI Notes:</h3>
              <p className="text-gray-700 text-sm">{result.notes}</p>
            </div>

            <div className="space-y-3">
              <button
                onClick={handleAddToMonitoring}
                className="w-full bg-blue-600 text-white py-3 rounded-lg font-semibold hover:bg-blue-700 transition-colors"
              >
                Add to Monitoring
              </button>
              <button
                onClick={() => navigate('/specialists')}
                className="w-full bg-gray-100 text-gray-700 py-3 rounded-lg font-semibold hover:bg-gray-200 transition-colors"
              >
                Get Specialist Recommendation
              </button>
              <button
                onClick={() => navigate('/')}
                className="w-full text-gray-500 py-2 font-medium"
              >
                Done
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ScanScreen;
