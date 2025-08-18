import { useState, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import { getRouteBackground, RouteBackground } from '@/config/routeBackgrounds';
import { useTheme } from 'next-themes';

export const useRouteBackgrounds = () => {
  const location = useLocation();
  const { theme } = useTheme();
  const [currentRouteBackground, setCurrentRouteBackground] = useState<RouteBackground>();
  const [userCustomBackground, setUserCustomBackground] = useState<string | null>(null);

  useEffect(() => {
    // Get the background for current route
    const routeBackground = getRouteBackground(location.pathname, theme === 'dark');
    setCurrentRouteBackground(routeBackground);

    // Check for user's custom background override
    const savedBackground = localStorage.getItem('selected-background');
    setUserCustomBackground(savedBackground);
  }, [location.pathname, theme]);

  const setCustomBackground = (backgroundUrl: string) => {
    setUserCustomBackground(backgroundUrl);
    localStorage.setItem('selected-background', backgroundUrl);
  };

  const clearCustomBackground = () => {
    setUserCustomBackground(null);
    localStorage.removeItem('selected-background');
  };

  const getCurrentBackground = (): RouteBackground | null => {
    // If user has set a custom background, use that
    if (userCustomBackground) {
      return {
        id: 'custom',
        name: 'Custom Background',
        url: userCustomBackground,
        overlay: currentRouteBackground?.overlay || 'linear-gradient(135deg, rgba(255, 255, 255, 0.4), rgba(99, 102, 241, 0.3))',
        darkOverlay: currentRouteBackground?.darkOverlay,
        description: 'User selected custom background'
      };
    }

    return currentRouteBackground || null;
  };

  const isCustomBackgroundActive = (): boolean => {
    return !!userCustomBackground;
  };

  return {
    currentRouteBackground,
    userCustomBackground,
    getCurrentBackground,
    setCustomBackground,
    clearCustomBackground,
    isCustomBackgroundActive
  };
};