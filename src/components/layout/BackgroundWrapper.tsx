import React from 'react';
import { useRouteBackgrounds } from '@/hooks/useRouteBackgrounds';
import { useTheme } from 'next-themes';

interface BackgroundWrapperProps {
  children: React.ReactNode;
  className?: string;
}

export const BackgroundWrapper = ({ children, className = '' }: BackgroundWrapperProps) => {
  const { getCurrentBackground } = useRouteBackgrounds();
  const { theme } = useTheme();
  const background = getCurrentBackground();

  if (!background) {
    return (
      <div className={`relative min-h-screen ${className}`}>
        {children}
      </div>
    );
  }

  const isGradient = background.url.startsWith('linear-gradient');
  const isDarkMode = theme === 'dark';

  return (
    <div
      className={`relative min-h-screen ${className}`}
      style={{
        backgroundImage: isGradient ? background.url : `url(${background.url})`,
        backgroundSize: 'cover',
        backgroundPosition: 'center',
        backgroundAttachment: 'fixed'
      }}
    >
      {/* Smart overlay for better content readability */}
      <div 
        className="absolute inset-0 pointer-events-none" 
        style={{
          background: isDarkMode ? background.darkOverlay || background.overlay : background.overlay
        }}
      />

      {/* Additional blur overlay for image backgrounds */}
      {!isGradient && (
        <div className="absolute inset-0 backdrop-blur-[0.5px] pointer-events-none" />
      )}

      {/* Content */}
      <div className="relative z-10">
        {children}
      </div>
    </div>
  );
};
