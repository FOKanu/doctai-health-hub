import { useState, useEffect, useMemo } from 'react';

interface BackgroundImage {
  id: string;
  name: string;
  url: string;
  description?: string;
}

export const useBackgroundImages = () => {
  const [backgroundImages, setBackgroundImages] = useState<BackgroundImage[]>([]);
  const [currentBackground, setCurrentBackground] = useState<string>('');

  // Placeholder images from your system that can be used as backgrounds
  const placeholderImages = useMemo((): BackgroundImage[] => [
    {
      id: 'medical-gradient',
      name: 'Medical Gradient',
      url: 'linear-gradient(135deg, hsl(var(--medical-accent-light)), hsl(var(--medical-bg)))',
      description: 'Clean medical gradient background'
    },
    {
      id: 'tech-pattern',
      name: 'Technology Pattern',
      url: 'https://images.unsplash.com/photo-1518770660439-4636190af475?auto=format&fit=crop&w=1920&q=80',
      description: 'Technology circuit board pattern'
    },
    {
      id: 'clean-minimal',
      name: 'Clean Minimal',
      url: 'https://images.unsplash.com/photo-1470813740244-df37b8c1edcb?auto=format&fit=crop&w=1920&q=80',
      description: 'Clean minimal background'
    }
  ], []);

  useEffect(() => {
    // Initialize with placeholder images
    setBackgroundImages(placeholderImages);

    // Set default background
    const savedBackground = localStorage.getItem('selected-background');
    if (savedBackground) {
      setCurrentBackground(savedBackground);
    } else {
      setCurrentBackground(placeholderImages[0].url);
    }
  }, [placeholderImages]);

  const setBackground = (imageUrl: string) => {
    setCurrentBackground(imageUrl);
    localStorage.setItem('selected-background', imageUrl);
  };

  const addCustomBackground = (file: File) => {
    return new Promise<BackgroundImage>((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = (e) => {
        const url = e.target?.result as string;
        const newImage: BackgroundImage = {
          id: `custom-${Date.now()}`,
          name: file.name,
          url,
          description: 'Custom uploaded background'
        };

        setBackgroundImages(prev => [...prev, newImage]);
        resolve(newImage);
      };
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  };

  return {
    backgroundImages,
    currentBackground,
    setBackground,
    addCustomBackground
  };
};
