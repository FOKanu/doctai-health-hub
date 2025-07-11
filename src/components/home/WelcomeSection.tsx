
import React from 'react';
import { TrendingUp } from 'lucide-react';

interface WelcomeSectionProps {
  healthScore: number;
}

export const WelcomeSection: React.FC<WelcomeSectionProps> = ({ healthScore }) => {
  return (
    <div className="bg-gradient-to-r from-primary to-primary-glow rounded-xl p-6 text-primary-foreground card-glass theme-dashboard">
      <div className="flex justify-between items-start">
        <div>
          <h1 className="text-2xl md:text-3xl font-bold mb-2">Welcome back!</h1>
          <p className="text-primary-foreground/80 mb-4">Your AI-powered health monitoring is active</p>
          <div className="flex items-center space-x-2 text-sm text-primary-foreground/90">
            <TrendingUp className="w-4 h-4" />
            <span>Health score improving by 12% this month</span>
          </div>
        </div>
        <div className="text-center">
          <div className="w-20 h-20 bg-primary-foreground/20 rounded-full flex items-center justify-center mb-2 health-score-glow">
            <span className="text-2xl font-bold">{healthScore}</span>
          </div>
          <p className="text-sm text-primary-foreground/80">Health Score</p>
        </div>
      </div>
    </div>
  );
};
