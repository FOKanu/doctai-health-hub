import { useState, useEffect } from 'react';
import { useLocation } from 'react-router-dom';

interface BackgroundImage {
  id: string;
  name: string;
  url: string;
  description?: string;
  pageSpecific?: string[];
  blendMode?: string;
  overlay?: string;
}

export const useBackgroundImages = () => {
  const [backgroundImages, setBackgroundImages] = useState<BackgroundImage[]>([]);
  const [currentBackground, setCurrentBackground] = useState<string>('');
  const location = useLocation();

  // Enhanced backgrounds with UI mockup integration and page-specific styling
  const placeholderImages: BackgroundImage[] = [
    {
      id: 'medical-gradient',
      name: 'Medical Gradient',
      url: 'linear-gradient(135deg, hsl(var(--medical-accent-light)), hsl(var(--medical-bg)))',
      description: 'Clean medical gradient background',
      overlay: 'rgba(255, 255, 255, 0.1)'
    },
    {
      id: 'ui-mockup-lavender',
      name: 'Lavender UI Blend',
      url: 'linear-gradient(135deg, #E8D5FF 0%, #D1C4E9 30%, #B39DDB 60%), url("/lovable-uploads/febc780b-8928-468d-bb33-6be080080d26.png")',
      description: 'Blended lavender gradient with UI mockup overlay',
      blendMode: 'overlay',
      overlay: 'rgba(232, 213, 255, 0.8)',
      pageSpecific: ['/', '/welcome', '/profile']
    },
    {
      id: 'ui-mockup-teal',
      name: 'Teal Medical Blend',
      url: 'linear-gradient(135deg, #B2DFDB 0%, #80CBC4 30%, #4DB6AC 60%), url("/lovable-uploads/a138d765-4920-4591-9bea-8df52c474c6c.png")',
      description: 'Teal medical gradient with scan UI overlay',
      blendMode: 'soft-light',
      overlay: 'rgba(178, 223, 219, 0.7)',
      pageSpecific: ['/scan', '/upload', '/results']
    },
    {
      id: 'soft-blue-ui',
      name: 'Soft Blue Healthcare',
      url: 'linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 30%, #90CAF9 60%), url("/lovable-uploads/febc780b-8928-468d-bb33-6be080080d26.png")',
      description: 'Soft blue with healthcare UI elements',
      blendMode: 'multiply',
      overlay: 'rgba(227, 242, 253, 0.6)',
      pageSpecific: ['/history', '/medical-records', '/appointments']
    },
    {
      id: 'rose-quartz-blend',
      name: 'Rose Quartz Wellness',
      url: 'linear-gradient(135deg, #FCE4EC 0%, #F8BBD9 30%, #F48FB1 60%), url("/lovable-uploads/a138d765-4920-4591-9bea-8df52c474c6c.png")',
      description: 'Rose quartz wellness theme with UI accents',
      blendMode: 'screen',
      overlay: 'rgba(252, 228, 236, 0.5)',
      pageSpecific: ['/fitness', '/diet', '/treatments']
    },
    {
      id: 'tech-pattern',
      name: 'Technology Pattern',
      url: 'https://images.unsplash.com/photo-1518770660439-4636190af475?auto=format&fit=crop&w=1920&q=80',
      description: 'Technology circuit board pattern',
      overlay: 'rgba(0, 0, 0, 0.2)'
    },
    {
      id: 'clean-minimal',
      name: 'Clean Minimal',
      url: 'https://images.unsplash.com/photo-1470813740244-df37b8c1edcb?auto=format&fit=crop&w=1920&q=80',
      description: 'Clean minimal background',
      overlay: 'rgba(255, 255, 255, 0.3)'
    }
  ];

  useEffect(() => {
    // Initialize with placeholder images
    setBackgroundImages(placeholderImages);
    
    // Set page-specific background or saved background
    const savedBackground = localStorage.getItem('selected-background');
    const autoSelectEnabled = localStorage.getItem('auto-select-background') !== 'false';
    
    if (autoSelectEnabled) {
      // Find page-specific background
      const pageSpecificBg = placeholderImages.find(bg => 
        bg.pageSpecific?.includes(location.pathname)
      );
      
      if (pageSpecificBg) {
        setCurrentBackground(pageSpecificBg.url);
        return;
      }
    }
    
    if (savedBackground) {
      setCurrentBackground(savedBackground);
    } else {
      setCurrentBackground(placeholderImages[0].url);
    }
  }, [location.pathname]);

  const setBackground = (imageUrl: string) => {
    setCurrentBackground(imageUrl);
    localStorage.setItem('selected-background', imageUrl);
    // Disable auto-select when manually setting background
    localStorage.setItem('auto-select-background', 'false');
  };

  const enableAutoSelect = () => {
    localStorage.setItem('auto-select-background', 'true');
    // Trigger re-evaluation
    const pageSpecificBg = backgroundImages.find(bg => 
      bg.pageSpecific?.includes(location.pathname)
    );
    if (pageSpecificBg) {
      setCurrentBackground(pageSpecificBg.url);
    }
  };

  const getCurrentBackgroundData = () => {
    return backgroundImages.find(bg => bg.url === currentBackground);
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
          description: 'Custom uploaded background',
          overlay: 'rgba(255, 255, 255, 0.2)'
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
    addCustomBackground,
    enableAutoSelect,
    getCurrentBackgroundData,
    currentPath: location.pathname
  };
};