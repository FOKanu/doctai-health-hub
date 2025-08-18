
import React from 'react';
import { TrendingUp } from 'lucide-react';

interface WelcomeSectionProps {
  healthScore: number;
}

export const WelcomeSection: React.FC<WelcomeSectionProps> = ({ healthScore }) => {
  return (
    <div className="relative overflow-hidden bg-gradient-to-br from-cyan-500 via-blue-500 to-indigo-600 rounded-xl p-6 text-white card-glass theme-dashboard">
      {/* Enhanced gradient overlay for depth */}
      <div className="absolute inset-0 bg-gradient-to-tr from-transparent via-white/5 to-white/10"></div>
      
      {/* Animated background elements */}
      <div className="absolute top-0 right-0 w-32 h-32 bg-white/10 rounded-full blur-xl -translate-y-8 translate-x-8"></div>
      <div className="absolute bottom-0 left-0 w-24 h-24 bg-cyan-300/20 rounded-full blur-lg translate-y-4 -translate-x-4"></div>
      
      {/* Content wrapper with relative positioning */}
      <div className="relative z-10">
        <div className="flex justify-between items-start">
          <div>
            <h1 className="text-2xl md:text-3xl font-bold mb-2 text-white drop-shadow-sm">Welcome back!</h1>
            <p className="text-white/90 mb-4 drop-shadow-sm">Your AI-powered health monitoring is active</p>
            <div className="flex items-center space-x-2 text-sm text-white/95 drop-shadow-sm">
              <TrendingUp className="w-4 h-4" />
              <span>Health score improving by 12% this month</span>
            </div>
          </div>
          <div className="text-center">
            <div className="w-20 h-20 bg-white/20 backdrop-blur-sm rounded-full flex items-center justify-center mb-2 border border-white/30 shadow-lg">
              <span className="text-2xl font-bold text-white drop-shadow-sm">{healthScore}</span>
            </div>
            <p className="text-xs text-white/90 font-medium drop-shadow-sm">Health Score</p>
          </div>
        </div>
      </div>
    </div>
  );
};
