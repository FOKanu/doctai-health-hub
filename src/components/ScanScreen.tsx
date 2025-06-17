import React, { useState, useRef, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { Camera, ArrowLeft, Flashlight, RefreshCw, Upload, ZoomIn, ZoomOut, RotateCcw } from 'lucide-react';
import { analyzePrediction, savePredictionToSupabase, PredictionResult } from '../services/predictionService';
import { BodyPart } from './BodyPartSelectionDialog';

interface ScanMetaData {
  bodyPart: BodyPart;
}

const ScanScreen = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const scanMetaData = location.state?.scanMetaData as ScanMetaData | undefined;
  
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [flashlightOn, setFlashlightOn] = useState(false);
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [facingMode, setFacingMode] = useState<'environment' | 'user'>('environment');
  const [zoom, setZoom] = useState(1);
  const [isZooming, setIsZooming] = useState(false);
  const [touchStartDistance, setTouchStartDistance] = useState<number | null>(null);
  const [isCapturing, setIsCapturing] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const zoomTimeoutRef = useRef<NodeJS.Timeout>();
  const captureTimeoutRef = useRef<NodeJS.Timeout>();

  // Redirect if no body part is selected
  useEffect(() => {
    if (!scanMetaData?.bodyPart) {
      navigate('/');
    }
  }, [scanMetaData, navigate]);

  useEffect(() => {
    startCamera();
    return () => {
      stopCamera();
    };
  }, [facingMode]);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode,
          width: { ideal: 1920 },
          height: { ideal: 1080 }
        }
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        streamRef.current = stream;
      }
    } catch (error) {
      console.error('Error accessing camera:', error);
      setCameraError('Unable to access camera. Please ensure camera permissions are granted.');
    }
  };

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
  };

  const handleSwitchCamera = () => {
    setFacingMode(prev => prev === 'environment' ? 'user' : 'environment');
  };

  const handleZoom = (direction: 'in' | 'out' | number) => {
    if (isZooming) return;

    setIsZooming(true);
    let newZoom: number;

    if (typeof direction === 'number') {
      newZoom = Math.max(1, Math.min(4, direction));
    } else {
      newZoom = direction === 'in' ? Math.min(zoom + 0.5, 4) : Math.max(zoom - 0.5, 1);
    }

    setZoom(newZoom);

    if (videoRef.current) {
      videoRef.current.style.transform = `scale(${newZoom})`;
    }

    if (zoomTimeoutRef.current) {
      clearTimeout(zoomTimeoutRef.current);
    }
    zoomTimeoutRef.current = setTimeout(() => {
      setIsZooming(false);
    }, 300);
  };

  const calculateTouchDistance = (touch1: Touch, touch2: Touch) => {
    const dx = touch1.clientX - touch2.clientX;
    const dy = touch1.clientY - touch2.clientY;
    return Math.sqrt(dx * dx + dy * dy);
  };

  const handleTouchStart = (e: React.TouchEvent) => {
    if (e.touches.length === 2) {
      const nativeEvent = e.nativeEvent;
      setTouchStartDistance(calculateTouchDistance(nativeEvent.touches[0], nativeEvent.touches[1]));
    }
  };

  const handleTouchMove = (e: React.TouchEvent) => {
    if (e.touches.length === 2 && touchStartDistance !== null) {
      const nativeEvent = e.nativeEvent;
      const currentDistance = calculateTouchDistance(nativeEvent.touches[0], nativeEvent.touches[1]);
      const zoomFactor = currentDistance / touchStartDistance;
      const newZoom = zoom * zoomFactor;
      handleZoom(newZoom);
      setTouchStartDistance(currentDistance);
    }
  };

  const handleTouchEnd = () => {
    setTouchStartDistance(null);
  };

  const handleCapture = async () => {
    if (!videoRef.current || isCapturing) return;

    setIsCapturing(true);

    if (videoRef.current) {
      videoRef.current.style.filter = 'brightness(0.8)';
    }

    captureTimeoutRef.current = setTimeout(async () => {
      const canvas = document.createElement('canvas');
      canvas.width = videoRef.current!.videoWidth;
      canvas.height = videoRef.current!.videoHeight;
      const ctx = canvas.getContext('2d');

      if (ctx) {
        ctx.drawImage(videoRef.current!, 0, 0);
        const imageUrl = canvas.toDataURL('image/jpeg', 1.0);
        setSelectedImage(imageUrl);
        await analyzeImage(imageUrl);
      }

      if (videoRef.current) {
        videoRef.current.style.filter = '';
      }
      setIsCapturing(false);
    }, 100);
  };

  const handleFileSelect = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const imageUrl = URL.createObjectURL(file);
    setSelectedImage(imageUrl);
    await analyzeImage(imageUrl);
  };

  const analyzeImage = async (imageUri: string) => {
    setIsAnalyzing(true);
    try {
      const predictionResult = await analyzePrediction(imageUri);
      
      // Include body part metadata in the result
      const enhancedResult = {
        ...predictionResult,
        metadata: {
          bodyPart: scanMetaData?.bodyPart
        }
      };
      
      await savePredictionToSupabase(enhancedResult, imageUri);
      setResult(enhancedResult);
      
      console.log('Scan completed for body part:', scanMetaData?.bodyPart);
    } catch (error) {
      console.error('Error analyzing image:', error);
      alert('Failed to analyze image. Please try again.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleAddToMonitoring = () => {
    alert('Added to weekly monitoring reminders!');
    navigate('/');
  };

  const handleRetake = () => {
    setSelectedImage(null);
    setResult(null);
    startCamera();
  };

  // Don't render if no body part selected
  if (!scanMetaData?.bodyPart) {
    return null;
  }

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
        <div className="text-center">
          <h1 className="text-white font-semibold">Skin Lesion Scan</h1>
          <p className="text-white text-sm opacity-75">Scanning: {scanMetaData.bodyPart}</p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={handleSwitchCamera}
            className="p-2 bg-black bg-opacity-50 rounded-full text-white"
          >
            <RotateCcw className="w-6 h-6" />
          </button>
          <button
            onClick={() => setFlashlightOn(!flashlightOn)}
            className={`p-2 rounded-full ${flashlightOn ? 'bg-yellow-500' : 'bg-black bg-opacity-50'} text-white`}
          >
            <Flashlight className="w-6 h-6" />
          </button>
        </div>
      </div>

      {/* Camera View or Selected Image */}
      <div className="relative h-screen">
        {selectedImage ? (
          <div className="absolute inset-0 flex items-center justify-center">
            <img src={selectedImage} alt="Selected" className="max-w-full max-h-full object-contain" />
          </div>
        ) : (
          <div
            className="absolute inset-0 bg-gradient-to-br from-gray-800 to-gray-900"
            onTouchStart={handleTouchStart}
            onTouchMove={handleTouchMove}
            onTouchEnd={handleTouchEnd}
          >
            {cameraError ? (
              <div className="h-full flex flex-col items-center justify-center text-white p-4 text-center">
                <Camera className="w-16 h-16 mx-auto mb-4 opacity-50" />
                <p className="text-lg mb-2">Camera Access Error</p>
                <p className="text-sm opacity-75 mb-4">{cameraError}</p>
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="flex items-center gap-2 bg-blue-600 text-white px-4 py-2 rounded-lg"
                >
                  <Upload className="w-4 h-4" />
                  Upload Image Instead
                </button>
              </div>
            ) : (
              <video
                ref={videoRef}
                autoPlay
                playsInline
                className="w-full h-full object-cover transition-all duration-100"
                style={{ transformOrigin: 'center' }}
              />
            )}
          </div>
        )}

        {/* Viewfinder and Zoom Controls */}
        {!selectedImage && !cameraError && (
          <>
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="w-64 h-64 border-2 border-white border-dashed rounded-full opacity-75"></div>
            </div>

            {/* Zoom Controls */}
            <div className="absolute right-4 top-1/2 transform -translate-y-1/2 flex flex-col gap-4">
              <button
                onClick={() => handleZoom('in')}
                disabled={isZooming || zoom >= 4}
                className="p-2 bg-black bg-opacity-50 rounded-full text-white disabled:opacity-50"
              >
                <ZoomIn className="w-6 h-6" />
              </button>
              <button
                onClick={() => handleZoom('out')}
                disabled={isZooming || zoom <= 1}
                className="p-2 bg-black bg-opacity-50 rounded-full text-white disabled:opacity-50"
              >
                <ZoomOut className="w-6 h-6" />
              </button>
            </div>

            {/* Zoom Level Indicator */}
            <div className="absolute top-20 left-1/2 transform -translate-x-1/2 bg-black bg-opacity-50 text-white px-3 py-1 rounded-full text-sm">
              {Math.round(zoom * 100)}%
            </div>

            {/* Touch Instructions */}
            <div className="absolute bottom-32 left-1/2 transform -translate-x-1/2 text-white text-center bg-black bg-opacity-50 px-4 py-2 rounded-full text-sm">
              Pinch to zoom • Tap to capture
            </div>
          </>
        )}

        {/* Capture/Retake Button */}
        <div className="absolute bottom-8 left-1/2 transform -translate-x-1/2">
          {selectedImage ? (
            <button
              onClick={handleRetake}
              className="w-20 h-20 bg-white rounded-full shadow-lg flex items-center justify-center hover:scale-105 transition-transform duration-200"
            >
              <RefreshCw className="w-8 h-8 text-blue-600" />
            </button>
          ) : (
            <button
              onClick={handleCapture}
              disabled={isAnalyzing || !!cameraError || isCapturing}
              className="w-20 h-20 bg-white rounded-full shadow-lg flex items-center justify-center hover:scale-105 transition-transform duration-200 disabled:opacity-50"
            >
              {isAnalyzing ? (
                <RefreshCw className="w-8 h-8 text-blue-600 animate-spin" />
              ) : (
                <div className="w-16 h-16 bg-blue-600 rounded-full"></div>
              )}
            </button>
          )}
        </div>

        {/* Hidden file input */}
        <input
          type="file"
          ref={fileInputRef}
          onChange={handleFileSelect}
          accept="image/*"
          className="hidden"
        />
      </div>

      {/* Analysis Result Overlay */}
      {result && (
        <div className="absolute inset-0 bg-black bg-opacity-75 flex items-end">
          <div className="w-full bg-white rounded-t-3xl p-6">
            <h2 className="text-xl font-bold mb-4">Analysis Complete</h2>

            <div className={`p-4 rounded-lg mb-4 ${result.prediction === 'benign' ? 'bg-green-50' : 'bg-red-50'}`}>
              <div className="flex justify-between items-center mb-2">
                <span className="font-semibold">Body Part:</span>
                <span className="font-bold text-blue-600">{scanMetaData.bodyPart}</span>
              </div>
              <div className="flex justify-between items-center mb-2">
                <span className="font-semibold">Prediction:</span>
                <span className={`font-bold ${result.prediction === 'benign' ? 'text-green-600' : 'text-red-600'}`}>
                  {result.prediction.toUpperCase()}
                </span>
              </div>
              <div className="flex justify-between items-center mb-2">
                <span className="font-semibold">Confidence:</span>
                <span className="font-bold">{(result.confidence * 100).toFixed(2)}%</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="font-semibold">Benign Probability:</span>
                <span className="font-bold">{(result.probabilities.benign * 100).toFixed(2)}%</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="font-semibold">Malignant Probability:</span>
                <span className="font-bold">{(result.probabilities.malignant * 100).toFixed(2)}%</span>
              </div>
            </div>

            <div className="flex space-x-3">
              <button
                onClick={() => navigate('/specialists')}
                className="flex-1 bg-blue-600 text-white py-3 rounded-lg font-medium hover:bg-blue-700 transition-colors"
              >
                Get Specialist
              </button>
              <button
                onClick={handleAddToMonitoring}
                className="flex-1 bg-gray-100 text-gray-700 py-3 rounded-lg font-medium hover:bg-gray-200 transition-colors"
              >
                Add to Monitoring
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ScanScreen;
