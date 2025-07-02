import React, { useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Upload, Check } from 'lucide-react';
import { useBackgroundImages } from '@/hooks/useBackgroundImages';

export const BackgroundSelector = () => {
  const { backgroundImages, currentBackground, setBackground, addCustomBackground } = useBackgroundImages();
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file && file.type.startsWith('image/')) {
      try {
        const newBackground = await addCustomBackground(file);
        setBackground(newBackground.url);
      } catch (error) {
        console.error('Error uploading background:', error);
      }
    }
  };

  const isGradient = (url: string) => url.startsWith('linear-gradient');
  const isSelected = (url: string) => url === currentBackground;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-lg">Background Images</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Upload Button */}
        <div>
          <Button
            variant="outline"
            onClick={() => fileInputRef.current?.click()}
            className="w-full"
          >
            <Upload className="w-4 h-4 mr-2" />
            Upload Custom Background
          </Button>
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
          {backgroundImages.map((image) => (
            <div
              key={image.id}
              className={`relative rounded-lg border-2 cursor-pointer transition-all duration-200 ${
                isSelected(image.url)
                  ? 'border-medical-accent shadow-lg'
                  : 'border-gray-200 hover:border-gray-300'
              }`}
              onClick={() => setBackground(image.url)}
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
                <p className="text-xs font-medium text-medical-text truncate">
                  {image.name}
                </p>
              </div>
              {isSelected(image.url) && (
                <div className="absolute top-1 right-1 bg-medical-accent text-white rounded-full p-1">
                  <Check className="w-3 h-3" />
                </div>
              )}
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
};