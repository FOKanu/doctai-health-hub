import React from 'react';
import { useBackgroundImages } from '@/hooks/useBackgroundImages';

interface BackgroundWrapperProps {
  children: React.ReactNode;
  className?: string;
}

export const BackgroundWrapper = ({ children, className = '' }: BackgroundWrapperProps) => {
  const { currentBackground, getCurrentBackgroundData } = useBackgroundImages();
  const backgroundData = getCurrentBackgroundData();

  const isGradient = currentBackground.startsWith('linear-gradient');
  const isMultipleBackground = currentBackground.includes(',');
  
  return (
    <div 
      className={`relative min-h-screen ${className}`}
      style={{
        background: isGradient && !isMultipleBackground ? currentBackground : undefined,
        backgroundImage: !isGradient ? `url(${currentBackground})` : 
                        isMultipleBackground ? currentBackground : undefined,
        backgroundSize: 'cover, cover',
        backgroundPosition: 'center, center',
        backgroundAttachment: 'fixed',
        backgroundBlendMode: backgroundData?.blendMode || 'normal',
        filter: 'blur(0.5px) brightness(1.1)'
      }}
    >
      {/* Dynamic overlay based on background data */}
      <div 
        className="absolute inset-0 backdrop-blur-[2px]" 
        style={{
          background: backgroundData?.overlay || 'rgba(255, 255, 255, 0.05)',
          mixBlendMode: 'overlay'
        }}
      />
      
      {/* Secondary blur layer for UI mockup backgrounds */}
      {isMultipleBackground && (
        <div className="absolute inset-0 bg-gradient-to-br from-white/20 via-transparent to-white/10 backdrop-blur-[1px]" />
      )}
      
      {/* Content with glassmorphic effect */}
      <div className="relative z-10 min-h-screen">
        <div className="backdrop-blur-[0.5px] min-h-screen">
          {children}
        </div>
      </div>
      
      {/* Subtle animated pulse for premium feel */}
      <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/5 to-transparent animate-pulse opacity-30 pointer-events-none" 
           style={{ animationDuration: '4s' }} />
    </div>
  );
};