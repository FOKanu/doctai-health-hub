import React from 'react';
import { useBackgroundImages } from '@/hooks/useBackgroundImages';

interface BackgroundWrapperProps {
  children: React.ReactNode;
  className?: string;
}

export const BackgroundWrapper = ({ children, className = '' }: BackgroundWrapperProps) => {
  const { currentBackground } = useBackgroundImages();

  const isGradient = currentBackground.startsWith('linear-gradient');

  return (
    <div
      className={`relative min-h-screen ${className}`}
      style={{
        backgroundImage: isGradient ? currentBackground : `url(${currentBackground})`,
        backgroundSize: 'cover',
        backgroundPosition: 'center',
        backgroundAttachment: 'fixed'
      }}
    >
      {/* Overlay for better content readability */}
      <div className="absolute inset-0 bg-white/5 backdrop-blur-[1px] pointer-events-none" />

      {/* Content */}
      <div className="relative z-10">
        {children}
      </div>
    </div>
  );
};
