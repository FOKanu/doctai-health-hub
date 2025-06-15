
import React from 'react';
import { AlertTriangle, TrendingUp, Shield, Eye } from 'lucide-react';

export const getRiskColor = (risk: string) => {
  switch (risk) {
    case 'high': return 'text-red-600 bg-red-50 border-red-200';
    case 'medium': return 'text-orange-600 bg-orange-50 border-orange-200';
    case 'low': return 'text-green-600 bg-green-50 border-green-200';
    default: return 'text-gray-600 bg-gray-50 border-gray-200';
  }
};

export const getRiskIcon = (risk: string) => {
  switch (risk) {
    case 'high': return <AlertTriangle className="w-4 h-4" />;
    case 'medium': return <TrendingUp className="w-4 h-4" />;
    case 'low': return <Shield className="w-4 h-4" />;
    default: return <Eye className="w-4 h-4" />;
  }
};
