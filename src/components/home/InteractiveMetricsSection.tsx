import React, { useState } from 'react';
import { Plus, Edit3, TrendingUp, TrendingDown, Minus } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { useHealthData } from '@/contexts/HealthDataContext';
import { toast } from '@/hooks/use-toast';

export const InteractiveMetricsSection: React.FC = () => {
  const { metrics, updateMetric } = useHealthData();
  const [editingMetric, setEditingMetric] = useState<string | null>(null);
  const [newValue, setNewValue] = useState<string>('');

  const handleStartEdit = (metricId: string, currentValue: number) => {
    setEditingMetric(metricId);
    setNewValue(currentValue.toString());
  };

  const handleSaveValue = (metricId: string) => {
    const value = parseFloat(newValue);
    if (isNaN(value) || value < 0) {
      toast({
        title: "Invalid Value",
        description: "Please enter a valid positive number.",
        variant: "destructive",
      });
      return;
    }

    updateMetric(metricId, value);
    setEditingMetric(null);
    setNewValue('');
    
    toast({
      title: "Metric Updated",
      description: "Your health metric has been updated successfully.",
    });
  };

  const handleCancelEdit = () => {
    setEditingMetric(null);
    setNewValue('');
  };

  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case 'up': return <TrendingUp className="w-4 h-4 text-green-500" />;
      case 'down': return <TrendingDown className="w-4 h-4 text-red-500" />;
      default: return <Minus className="w-4 h-4 text-gray-500" />;
    }
  };

  const getProgressColor = (progress: number) => {
    if (progress >= 100) return 'bg-green-500';
    if (progress >= 80) return 'bg-blue-500';
    if (progress >= 60) return 'bg-yellow-500';
    return 'bg-red-500';
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>Health Metrics</span>
          <Button variant="outline" size="sm">
            <Plus className="w-4 h-4 mr-2" />
            Add Metric
          </Button>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {metrics.map((metric) => {
            const progress = Math.min((metric.value / metric.target) * 100, 100);
            const isEditing = editingMetric === metric.id;

            return (
              <div key={metric.id} className="bg-muted rounded-lg p-4 space-y-3">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-medium text-foreground">{metric.name}</h3>
                  <div className="flex items-center space-x-2">
                    {getTrendIcon(metric.trend)}
                    {!isEditing && (
                      <Button 
                        variant="ghost" 
                        size="sm"
                        onClick={() => handleStartEdit(metric.id, metric.value)}
                        className="h-6 w-6 p-0"
                      >
                        <Edit3 className="w-3 h-3" />
                      </Button>
                    )}
                  </div>
                </div>

                <div className="space-y-2">
                  {isEditing ? (
                    <div className="flex items-center space-x-2">
                      <Input
                        type="number"
                        value={newValue}
                        onChange={(e) => setNewValue(e.target.value)}
                        className="h-8 text-sm"
                        placeholder={`Enter ${metric.unit}`}
                      />
                      <Button 
                        size="sm" 
                        onClick={() => handleSaveValue(metric.id)}
                        className="h-8 px-2"
                      >
                        Save
                      </Button>
                      <Button 
                        variant="outline" 
                        size="sm" 
                        onClick={handleCancelEdit}
                        className="h-8 px-2"
                      >
                        Cancel
                      </Button>
                    </div>
                  ) : (
                    <div className="text-lg font-bold text-foreground">
                      {metric.value} {metric.unit}
                    </div>
                  )}

                  <div className="text-xs text-muted-foreground">
                    Target: {metric.target} {metric.unit}
                  </div>

                  <div className="w-full bg-border rounded-full h-2">
                    <div
                      className={`h-2 rounded-full transition-all duration-300 ${getProgressColor(progress)}`}
                      style={{ width: `${Math.min(progress, 100)}%` }}
                    />
                  </div>

                  <div className="flex justify-between text-xs">
                    <span className="font-medium text-foreground">
                      {Math.round(progress)}% of goal
                    </span>
                    <span className="text-muted-foreground">
                      {progress >= 100 ? 'Goal achieved!' : progress >= 80 ? 'On track' : 'Needs attention'}
                    </span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </CardContent>
    </Card>
  );
};