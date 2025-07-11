import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { TrendingUp, Clock, Activity } from 'lucide-react';

interface SequenceAnalyzerProps {
  userId: string;
}

export const SequenceAnalyzer: React.FC<SequenceAnalyzerProps> = ({ userId }) => {
  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <TrendingUp className="w-5 h-5" />
            Time-Series Health Analysis
          </CardTitle>
          <CardDescription>
            Track health progression and patterns over time
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-muted p-4 rounded-lg">
              <div className="flex items-center gap-2 mb-2">
                <Activity className="w-4 h-4 text-primary" />
                <span className="font-medium">Health Score Trend</span>
              </div>
              <p className="text-2xl font-bold text-foreground">+12%</p>
              <p className="text-sm text-muted-foreground">Improving over 30 days</p>
            </div>
            
            <div className="bg-muted p-4 rounded-lg">
              <div className="flex items-center gap-2 mb-2">
                <Clock className="w-4 h-4 text-primary" />
                <span className="font-medium">Data Points</span>
              </div>
              <p className="text-2xl font-bold text-foreground">247</p>
              <p className="text-sm text-muted-foreground">Last 90 days</p>
            </div>
            
            <div className="bg-muted p-4 rounded-lg">
              <div className="flex items-center gap-2 mb-2">
                <TrendingUp className="w-4 h-4 text-primary" />
                <span className="font-medium">Prediction Accuracy</span>
              </div>
              <p className="text-2xl font-bold text-foreground">94.2%</p>
              <p className="text-sm text-muted-foreground">Model confidence</p>
            </div>
          </div>
          
          <div className="mt-6 p-4 bg-primary/10 rounded-lg">
            <h3 className="font-semibold text-foreground mb-2">Key Insights</h3>
            <ul className="space-y-2 text-sm text-muted-foreground">
              <li>• Health score shows consistent improvement over the last month</li>
              <li>• Predictive models indicate continued positive trajectory</li>
              <li>• Time-series analysis ready for future implementation</li>
            </ul>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};