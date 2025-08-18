import React, { useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Upload, Check, RotateCcw } from 'lucide-react';
import { useBackgroundImages } from '@/hooks/useBackgroundImages';
import { useRouteBackgrounds } from '@/hooks/useRouteBackgrounds';
import { getAllRouteBackgrounds } from '@/config/routeBackgrounds';

export const BackgroundSelector = () => {
  const { backgroundImages, currentBackground, setBackground, addCustomBackground } = useBackgroundImages();
  const { 
    getCurrentBackground, 
    setCustomBackground, 
    clearCustomBackground, 
    isCustomBackgroundActive 
  } = useRouteBackgrounds();
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Combine route backgrounds with user custom backgrounds
  const allBackgrounds = [
    ...getAllRouteBackgrounds(),
    ...backgroundImages.filter(img => img.id.startsWith('custom-'))
  ];

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file && file.type.startsWith('image/')) {
      try {
        const newBackground = await addCustomBackground(file);
        setCustomBackground(newBackground.url);
      } catch (error) {
        console.error('Error uploading background:', error);
      }
    }
  };

  const handleBackgroundSelect = (url: string) => {
    setCustomBackground(url);
  };

  const handleResetToRoute = () => {
    clearCustomBackground();
  };

  const isGradient = (url: string) => url.startsWith('linear-gradient');
  const isSelected = (url: string) => {
    const currentBg = getCurrentBackground();
    return currentBg?.url === url;
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-lg">Background Images</CardTitle>
        <p className="text-sm text-muted-foreground">
          Choose a custom background or use automatic route-based backgrounds
        </p>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Control Buttons */}
        <div className="flex gap-2">
          <Button
            variant="outline"
            onClick={() => fileInputRef.current?.click()}
            className="flex-1"
          >
            <Upload className="w-4 h-4 mr-2" />
            Upload Custom
          </Button>
          {isCustomBackgroundActive() && (
            <Button
              variant="outline"
              onClick={handleResetToRoute}
              className="flex-1"
            >
              <RotateCcw className="w-4 h-4 mr-2" />
              Reset to Route
            </Button>
          )}
          <Input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleFileUpload}
            className="hidden"
          />
        </div>

        {/* Background Options */}
        <div className="grid grid-cols-2 gap-3">
          {allBackgrounds.map((image) => (
            <div
              key={image.id}
              className={`relative rounded-lg border-2 cursor-pointer transition-all duration-200 ${
                isSelected(image.url)
                  ? 'border-primary shadow-lg'
                  : 'border-border hover:border-primary/50'
              }`}
              onClick={() => handleBackgroundSelect(image.url)}
            >
              <div
                className="h-20 rounded-md"
                style={{
                  background: isGradient(image.url) ? image.url : `url(${image.url})`,
                  backgroundSize: 'cover',
                  backgroundPosition: 'center'
                }}
              />
              <div className="p-2 text-center">
                <p className="text-xs font-medium truncate">
                  {image.name}
                </p>
                {image.description && (
                  <p className="text-xs text-muted-foreground truncate">
                    {image.description}
                  </p>
                )}
              </div>
              {isSelected(image.url) && (
                <div className="absolute top-1 right-1 bg-primary text-primary-foreground rounded-full p-1">
                  <Check className="w-3 h-3" />
                </div>
              )}
            </div>
          ))}
        </div>

        {/* Current Background Info */}
        {getCurrentBackground() && (
          <div className="mt-4 p-3 bg-muted rounded-lg">
            <p className="text-sm font-medium">
              Current: {getCurrentBackground()?.name}
            </p>
            <p className="text-xs text-muted-foreground">
              {isCustomBackgroundActive() ? 'Custom background active' : 'Route-based background'}
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  );
};