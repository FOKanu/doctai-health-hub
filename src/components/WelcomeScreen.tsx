
import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Upload, Search, History, ArrowRight, Shield, Heart } from 'lucide-react';

const WelcomeScreen = () => {
  const navigate = useNavigate();

  const tutorialSteps = [
    {
      icon: Upload,
      title: 'Upload Medical Scans',
      description: 'Easily upload X-rays, MRIs, CT scans, and other medical images. Our AI will analyze them for potential insights.',
      steps: [
        'Tap the "Scan" button on your dashboard',
        'Choose to take a photo or upload from your device',
        'Wait for AI analysis (usually 30-60 seconds)',
        'Review results and recommendations'
      ]
    },
    {
      icon: Search,
      title: 'View Diagnostic Results',
      description: 'Access detailed analysis reports with clear explanations and visual highlights of areas of interest.',
      steps: [
        'Navigate to "Results" from the main menu',
        'Select any scan to view detailed analysis',
        'See highlighted areas and confidence scores',
        'Download or share reports with your doctor'
      ]
    },
    {
      icon: History,
      title: 'Track Health History',
      description: 'Monitor your health journey over time with comprehensive tracking and trend analysis.',
      steps: [
        'Visit "History" to see all your past scans',
        'Compare results across different time periods',
        'Set reminders for follow-up scans',
        'Export data for medical appointments'
      ]
    }
  ];

  const handleProceed = () => {
    navigate('/');
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 to-blue-50 p-4">
      <div className="max-w-4xl mx-auto">
        {/* Welcome Header */}
        <div className="text-center mb-8 pt-8">
          <div className="flex justify-center mb-4">
            <div className="bg-white rounded-full p-4 shadow-lg">
              <Heart className="h-12 w-12 text-blue-600" />
            </div>
          </div>
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            Welcome to DoctAI! 🎉
          </h1>
          <p className="text-xl text-gray-600 max-w-2xl mx-auto">
            Your personal AI-powered health assistant is ready to help you analyze medical scans and track your health journey.
          </p>
        </div>

        {/* Tutorial Section */}
        <div className="mb-8">
          <h2 className="text-2xl font-bold text-gray-900 text-center mb-6">
            How to Get Started
          </h2>
          <div className="grid md:grid-cols-3 gap-6">
            {tutorialSteps.map((step, index) => (
              <Card key={index} className="shadow-lg hover:shadow-xl transition-shadow">
                <CardHeader className="text-center">
                  <div className="flex justify-center mb-3">
                    <div className="bg-blue-100 rounded-full p-3">
                      <step.icon className="h-8 w-8 text-blue-600" />
                    </div>
                  </div>
                  <CardTitle className="text-lg font-semibold text-gray-900">
                    {step.title}
                  </CardTitle>
                  <CardDescription className="text-gray-600">
                    {step.description}
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <ul className="space-y-2">
                    {step.steps.map((stepText, stepIndex) => (
                      <li key={stepIndex} className="flex items-start">
                        <span className="bg-blue-600 text-white text-xs rounded-full w-5 h-5 flex items-center justify-center mr-3 mt-0.5 font-medium">
                          {stepIndex + 1}
                        </span>
                        <span className="text-sm text-gray-700">{stepText}</span>
                      </li>
                    ))}
                  </ul>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>

        {/* Legal Disclaimer */}
        <Alert className="mb-8 border-amber-200 bg-amber-50">
          <Shield className="h-5 w-5 text-amber-600" />
          <AlertDescription className="text-amber-800">
            <strong className="font-semibold">Important Medical Disclaimer:</strong>{' '}
            This app is not a replacement for professional medical care or specialist opinions. 
            The AI analysis is for informational purposes only and should not be used as a substitute 
            for professional medical diagnosis, treatment, or advice. Always consult a qualified 
            healthcare provider for medical concerns and before making any medical decisions.
          </AlertDescription>
        </Alert>

        {/* Quick Tips */}
        <Card className="mb-8 bg-blue-50 border-blue-200">
          <CardHeader>
            <CardTitle className="text-lg font-semibold text-blue-900 flex items-center">
              <Heart className="h-5 w-5 mr-2" />
              Quick Tips for Best Results
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid md:grid-cols-2 gap-4 text-sm text-blue-800">
              <ul className="space-y-2">
                <li className="flex items-start">
                  <span className="text-blue-600 mr-2">•</span>
                  Ensure images are clear and well-lit
                </li>
                <li className="flex items-start">
                  <span className="text-blue-600 mr-2">•</span>
                  Include patient information when uploading
                </li>
              </ul>
              <ul className="space-y-2">
                <li className="flex items-start">
                  <span className="text-blue-600 mr-2">•</span>
                  Save or screenshot results for your records
                </li>
                <li className="flex items-start">
                  <span className="text-blue-600 mr-2">•</span>
                  Share findings with your healthcare provider
                </li>
              </ul>
            </div>
          </CardContent>
        </Card>

        {/* Proceed Button */}
        <div className="text-center">
          <Button
            onClick={handleProceed}
            size="lg"
            className="bg-blue-600 hover:bg-blue-700 text-white font-medium px-8 py-3 rounded-lg shadow-lg hover:shadow-xl transition-all duration-200 transform hover:scale-105"
          >
            Get Started with DoctAI
            <ArrowRight className="ml-2 h-5 w-5" />
          </Button>
          <p className="text-sm text-gray-500 mt-4">
            Ready to begin your health journey? Let's go!
          </p>
        </div>
      </div>
    </div>
  );
};

export default WelcomeScreen;
