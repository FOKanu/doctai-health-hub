import React, { useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { 
  Shield, 
  Upload, 
  Search, 
  TrendingUp, 
  Lock, 
  ArrowRight, 
  ArrowLeft,
  ChevronRight,
  X,
  Heart
} from 'lucide-react';
import {
  Carousel,
  CarouselContent,
  CarouselItem,
  CarouselNext,
  CarouselPrevious,
  type CarouselApi,
} from '@/components/ui/carousel';

interface OnboardingCard {
  id: string;
  title: string;
  content: React.ReactNode;
  icon: React.ComponentType<{ className?: string }>;
}

const OnboardingCarousel = () => {
  const navigate = useNavigate();
  const [api, setApi] = useState<CarouselApi>();
  const [current, setCurrent] = useState(0);
  const [count, setCount] = useState(0);

  // Define handlers before they're used
  const handleSkip = () => {
    navigate('/');
  };

  const handleGetStarted = () => {
    navigate('/');
  };

  const onboardingCards: OnboardingCard[] = [
    {
      id: 'disclaimer',
      title: 'Important Medical Disclaimer',
      icon: Shield,
      content: (
        <div className="space-y-4">
          <Alert className="border-amber-200 bg-amber-50">
            <Shield className="h-5 w-5 text-amber-600" />
            <AlertDescription className="text-amber-800">
              <strong className="font-semibold">Important:</strong> This app is not a replacement for professional medical care or specialist opinions.
            </AlertDescription>
          </Alert>
          <div className="space-y-3 text-foreground">
            <p>• The AI analysis is for informational purposes only</p>
            <p>• Should not be used as a substitute for professional medical diagnosis</p>
            <p>• Always consult a qualified healthcare provider before making any medical decisions</p>
            <p>• Seek immediate medical attention for emergencies</p>
          </div>
        </div>
      )
    },
    {
      id: 'upload',
      title: 'Upload Medical Scans',
      icon: Upload,
      content: (
        <div className="space-y-4">
          <p className="text-muted-foreground">
            Easily upload X-rays, MRIs, CT scans, and other medical images for AI analysis.
          </p>
          <div className="space-y-3">
            <div className="flex items-start space-x-3">
              <span className="bg-primary text-primary-foreground text-xs rounded-full w-6 h-6 flex items-center justify-center font-medium">1</span>
              <span>Tap "Scan" button on dashboard</span>
            </div>
            <div className="flex items-start space-x-3">
              <span className="bg-primary text-primary-foreground text-xs rounded-full w-6 h-6 flex items-center justify-center font-medium">2</span>
              <span>Take photo or upload from device</span>
            </div>
            <div className="flex items-start space-x-3">
              <span className="bg-primary text-primary-foreground text-xs rounded-full w-6 h-6 flex items-center justify-center font-medium">3</span>
              <span>Wait for AI analysis (30–60s)</span>
            </div>
            <div className="flex items-start space-x-3">
              <span className="bg-primary text-primary-foreground text-xs rounded-full w-6 h-6 flex items-center justify-center font-medium">4</span>
              <span>Review results and recommendations</span>
            </div>
          </div>
        </div>
      )
    },
    {
      id: 'results',
      title: 'View Diagnostic Results',
      icon: Search,
      content: (
        <div className="space-y-4">
          <p className="text-muted-foreground">
            Access detailed analysis reports with clear explanations and visual highlights.
          </p>
          <div className="space-y-3">
            <div className="flex items-start space-x-3">
              <span className="bg-primary text-primary-foreground text-xs rounded-full w-6 h-6 flex items-center justify-center font-medium">1</span>
              <span>Navigate to "Results" from the main menu</span>
            </div>
            <div className="flex items-start space-x-3">
              <span className="bg-primary text-primary-foreground text-xs rounded-full w-6 h-6 flex items-center justify-center font-medium">2</span>
              <span>Select any scan to view detailed analysis</span>
            </div>
            <div className="flex items-start space-x-3">
              <span className="bg-primary text-primary-foreground text-xs rounded-full w-6 h-6 flex items-center justify-center font-medium">3</span>
              <span>See confidence score and highlighted areas</span>
            </div>
            <div className="flex items-start space-x-3">
              <span className="bg-primary text-primary-foreground text-xs rounded-full w-6 h-6 flex items-center justify-center font-medium">4</span>
              <span>Save or share results with your doctor</span>
            </div>
          </div>
        </div>
      )
    },
    {
      id: 'tracking',
      title: 'Track Your Health',
      icon: TrendingUp,
      content: (
        <div className="space-y-4">
          <p className="text-muted-foreground">
            Monitor your health journey over time with comprehensive tracking and insights.
          </p>
          <div className="space-y-3">
            <div className="flex items-start space-x-3">
              <span className="bg-primary text-primary-foreground text-xs rounded-full w-6 h-6 flex items-center justify-center font-medium">1</span>
              <span>View health trends and patterns over time</span>
            </div>
            <div className="flex items-start space-x-3">
              <span className="bg-primary text-primary-foreground text-xs rounded-full w-6 h-6 flex items-center justify-center font-medium">2</span>
              <span>Set reminders for follow-up scans</span>
            </div>
            <div className="flex items-start space-x-3">
              <span className="bg-primary text-primary-foreground text-xs rounded-full w-6 h-6 flex items-center justify-center font-medium">3</span>
              <span>Compare results across different periods</span>
            </div>
            <div className="flex items-start space-x-3">
              <span className="bg-primary text-primary-foreground text-xs rounded-full w-6 h-6 flex items-center justify-center font-medium">4</span>
              <span>Export data for medical appointments</span>
            </div>
          </div>
        </div>
      )
    },
    {
      id: 'privacy',
      title: 'Your Data is Safe with Us',
      icon: Lock,
      content: (
        <div className="space-y-4">
          <p className="text-muted-foreground">
            We take your privacy and data security seriously with enterprise-grade protection.
          </p>
          <div className="space-y-3">
            <div className="flex items-start space-x-3">
              <Lock className="h-5 w-5 text-green-600 mt-0.5" />
              <span>End-to-end encrypted storage</span>
            </div>
            <div className="flex items-start space-x-3">
              <Lock className="h-5 w-5 text-green-600 mt-0.5" />
              <span>Fully anonymized data processing</span>
            </div>
            <div className="flex items-start space-x-3">
              <Lock className="h-5 w-5 text-green-600 mt-0.5" />
              <span>HIPAA & GDPR compliant</span>
            </div>
            <div className="flex items-start space-x-3">
              <Lock className="h-5 w-5 text-green-600 mt-0.5" />
              <span>No data sharing with third parties</span>
            </div>
          </div>
          <div className="pt-2">
            <Button variant="link" className="p-0 h-auto text-sm text-primary">
              Read our Privacy Policy
            </Button>
          </div>
        </div>
      )
    },
    {
      id: 'begin',
      title: "Let's Begin!",
      icon: Heart,
      content: (
        <div className="space-y-6 text-center">
          <div className="flex justify-center">
            <div className="bg-primary/10 rounded-full p-6">
              <Heart className="h-16 w-16 text-primary" />
            </div>
          </div>
          <div className="space-y-3">
            <p className="text-lg font-medium">Ready to begin your health journey?</p>
            <p className="text-muted-foreground">
              DoctAI is here to help you monitor your health with AI-powered insights and recommendations.
            </p>
          </div>
          <Button 
            onClick={handleGetStarted}
            size="lg"
            className="w-full"
          >
            Get Started with DoctAI
            <ArrowRight className="ml-2 h-5 w-5" />
          </Button>
        </div>
      )
    }
  ];

  React.useEffect(() => {
    if (!api) {
      return;
    }

    setCount(api.scrollSnapList().length);
    setCurrent(api.selectedScrollSnap() + 1);

    api.on("select", () => {
      setCurrent(api.selectedScrollSnap() + 1);
    });
  }, [api]);

  const handleNext = useCallback(() => {
    api?.scrollNext();
  }, [api]);

  const handlePrevious = useCallback(() => {
    api?.scrollPrev();
  }, [api]);

  const goToSlide = (index: number) => {
    api?.scrollTo(index);
  };

  const isLastSlide = current === count;
  const isFirstSlide = current === 1;

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-muted/20 p-4">
      <div className="max-w-2xl mx-auto">
        {/* Header with Skip Button */}
        <div className="flex justify-between items-center mb-8 pt-8">
          <div className="flex justify-center">
            <div className="bg-card rounded-full p-3 shadow-lg">
              <Heart className="h-8 w-8 text-primary" />
            </div>
          </div>
          <Button 
            variant="ghost" 
            onClick={handleSkip}
            className="text-muted-foreground hover:text-foreground"
            aria-label="Skip tutorial"
          >
            Skip
            <X className="ml-1 h-4 w-4" />
          </Button>
        </div>

        {/* Welcome Text */}
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold mb-2">
            Welcome to DoctAI! 🎉
          </h1>
          <p className="text-muted-foreground">
            Let's take a quick tour to get you started
          </p>
        </div>

        {/* Carousel */}
        <Carousel setApi={setApi} className="w-full">
          <CarouselContent>
            {onboardingCards.map((card, index) => (
              <CarouselItem key={card.id}>
                <Card className="min-h-[400px] flex flex-col">
                  <CardHeader className="text-center pb-4">
                    <div className="flex justify-center mb-4">
                      <div className="bg-primary/10 rounded-full p-4">
                        <card.icon className="h-10 w-10 text-primary" />
                      </div>
                    </div>
                    <CardTitle className="text-xl font-bold">
                      {card.title}
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="flex-1 flex flex-col justify-center">
                    {card.content}
                  </CardContent>
                </Card>
              </CarouselItem>
            ))}
          </CarouselContent>
        </Carousel>

        {/* Dot Indicators */}
        <div className="flex justify-center space-x-2 mt-6 mb-6">
          {Array.from({ length: count }, (_, index) => (
            <button
              key={index}
              onClick={() => goToSlide(index)}
              className={`w-3 h-3 rounded-full transition-all duration-200 ${
                index + 1 === current 
                  ? 'bg-primary scale-110' 
                  : 'bg-muted-foreground/30 hover:bg-muted-foreground/50'
              }`}
              aria-label={`Go to slide ${index + 1}`}
            />
          ))}
        </div>

        {/* Navigation Buttons */}
        {!isLastSlide && (
          <div className="flex justify-between items-center">
            <Button
              variant="outline"
              onClick={handlePrevious}
              disabled={isFirstSlide}
              aria-label="Previous slide"
            >
              <ArrowLeft className="mr-2 h-4 w-4" />
              Back
            </Button>

            <div className="text-sm text-muted-foreground">
              {current} of {count}
            </div>

            <Button
              onClick={handleNext}
              aria-label="Next slide"
            >
              Next
              <ChevronRight className="ml-2 h-4 w-4" />
            </Button>
          </div>
        )}
      </div>
    </div>
  );
};

export default OnboardingCarousel;