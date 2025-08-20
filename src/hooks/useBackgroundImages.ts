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

  // Vibrant health-related background images for enhanced user experience
  const placeholderImages = useMemo((): BackgroundImage[] => [
    {
      id: 'dna-helix',
      name: 'DNA Double Helix',
      url: 'https://images.unsplash.com/photo-1559757148-5c350d0d3c56?auto=format&fit=crop&w=1920&q=80',
      description: 'Vibrant DNA double helix structure'
    },
    {
      id: 'molecular-structure',
      name: 'Molecular Network',
      url: 'https://images.unsplash.com/photo-1576086213369-97a306d36557?auto=format&fit=crop&w=1920&q=80',
      description: 'Dynamic molecular structure network'
    },
    {
      id: 'medical-research',
      name: 'Medical Research',
      url: 'https://images.unsplash.com/photo-1582719471384-894fbb16e074?auto=format&fit=crop&w=1920&q=80',
      description: 'Modern medical research laboratory'
    },
    {
      id: 'neural-network',
      name: 'Neural Pathways',
      url: 'https://images.unsplash.com/photo-1559757175-0eb30cd8c063?auto=format&fit=crop&w=1920&q=80',
      description: 'Neural network and brain pathways'
    },
    {
      id: 'cellular-biology',
      name: 'Cellular Structure',
      url: 'https://images.unsplash.com/photo-1628595351029-c2bf17511435?auto=format&fit=crop&w=1920&q=80',
      description: 'Microscopic cellular biology view'
    },
    {
      id: 'medical-gradient-enhanced',
      name: 'Health Gradient',
      url: 'linear-gradient(135deg, #00d4aa 0%, #00a3cc 25%, #0066ff 50%, #6366f1 75%, #8b5cf6 100%)',
      description: 'Dynamic health-themed gradient'
    },
    {
      id: 'heartbeat-wave',
      name: 'Vital Signs',
      url: 'https://images.unsplash.com/photo-1559757146-8c3d6e7a3e2b?auto=format&fit=crop&w=1920&q=80',
      description: 'Heartbeat and vital signs visualization'
    },
    {
      id: 'modern-healthcare',
      name: 'Digital Health',
      url: 'https://images.unsplash.com/photo-1576091160399-112ba8d25d1f?auto=format&fit=crop&w=1920&q=80',
      description: 'Modern digital healthcare technology'
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
