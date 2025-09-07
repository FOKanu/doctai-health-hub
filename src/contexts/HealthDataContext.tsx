import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';

export interface HealthMetric {
  id: string;
  name: string;
  value: number;
  target: number;
  unit: string;
  icon: string;
  category: 'fitness' | 'nutrition' | 'sleep' | 'vitals';
  lastUpdated: Date;
  trend: 'up' | 'down' | 'stable';
}

export interface HealthGoal {
  id: string;
  title: string;
  description: string;
  targetValue: number;
  currentValue: number;
  unit: string;
  deadline: Date;
  category: string;
  priority: 'low' | 'medium' | 'high';
  achieved: boolean;
}

export interface HealthAlert {
  id: string;
  type: 'info' | 'warning' | 'critical';
  title: string;
  message: string;
  timestamp: Date;
  read: boolean;
  actionRequired: boolean;
}

interface HealthDataContextType {
  metrics: HealthMetric[];
  goals: HealthGoal[];
  alerts: HealthAlert[];
  healthScore: number;
  updateMetric: (id: string, value: number) => void;
  addGoal: (goal: Omit<HealthGoal, 'id'>) => void;
  updateGoal: (id: string, updates: Partial<HealthGoal>) => void;
  markAlertAsRead: (id: string) => void;
  refreshHealthScore: () => void;
}

const HealthDataContext = createContext<HealthDataContextType | undefined>(undefined);

interface HealthDataProviderProps {
  children: ReactNode;
}

// Mock data - in real app this would come from Supabase
const mockMetrics: HealthMetric[] = [
  { id: '1', name: 'Daily Steps', value: 8500, target: 10000, unit: 'steps', icon: 'Activity', category: 'fitness', lastUpdated: new Date(), trend: 'up' },
  { id: '2', name: 'Water Intake', value: 6, target: 8, unit: 'glasses', icon: 'Droplets', category: 'nutrition', lastUpdated: new Date(), trend: 'stable' },
  { id: '3', name: 'Sleep Hours', value: 7.2, target: 8, unit: 'hours', icon: 'Moon', category: 'sleep', lastUpdated: new Date(), trend: 'down' },
  { id: '4', name: 'Heart Rate', value: 72, target: 65, unit: 'bpm', icon: 'Heart', category: 'vitals', lastUpdated: new Date(), trend: 'stable' },
  { id: '5', name: 'Weight', value: 75.2, target: 73.0, unit: 'kg', icon: 'Weight', category: 'vitals', lastUpdated: new Date(), trend: 'down' },
  { id: '6', name: 'Body Temperature', value: 98.6, target: 98.6, unit: '°F', icon: 'Thermometer', category: 'vitals', lastUpdated: new Date(), trend: 'stable' },
];

const mockGoals: HealthGoal[] = [
  { id: '1', title: 'Lose 5kg', description: 'Reach target weight through healthy diet', targetValue: 70, currentValue: 75.2, unit: 'kg', deadline: new Date('2024-06-01'), category: 'weight-loss', priority: 'high', achieved: false },
  { id: '2', title: 'Walk 10k steps daily', description: 'Maintain consistent daily activity', targetValue: 10000, currentValue: 8500, unit: 'steps', deadline: new Date('2024-03-31'), category: 'fitness', priority: 'medium', achieved: false },
];

const mockAlerts: HealthAlert[] = [
  { id: '1', type: 'warning', title: 'Sleep Quality', message: 'Your sleep duration has decreased by 15% this week', timestamp: new Date(), read: false, actionRequired: true },
  { id: '2', type: 'info', title: 'Medication Reminder', message: 'Prescription renewal due in 3 days', timestamp: new Date(), read: false, actionRequired: true },
];

export const HealthDataProvider: React.FC<HealthDataProviderProps> = ({ children }) => {
  const [metrics, setMetrics] = useState<HealthMetric[]>(mockMetrics);
  const [goals, setGoals] = useState<HealthGoal[]>(mockGoals);
  const [alerts, setAlerts] = useState<HealthAlert[]>(mockAlerts);
  const [healthScore, setHealthScore] = useState(78);

  const updateMetric = (id: string, value: number) => {
    setMetrics(prev => prev.map(metric => 
      metric.id === id 
        ? { ...metric, value, lastUpdated: new Date() }
        : metric
    ));
    refreshHealthScore();
  };

  const addGoal = (goal: Omit<HealthGoal, 'id'>) => {
    const newGoal: HealthGoal = {
      ...goal,
      id: `goal_${Date.now()}`,
    };
    setGoals(prev => [...prev, newGoal]);
  };

  const updateGoal = (id: string, updates: Partial<HealthGoal>) => {
    setGoals(prev => prev.map(goal => 
      goal.id === id ? { ...goal, ...updates } : goal
    ));
  };

  const markAlertAsRead = (id: string) => {
    setAlerts(prev => prev.map(alert => 
      alert.id === id ? { ...alert, read: true } : alert
    ));
  };

  const refreshHealthScore = () => {
    // Simple health score calculation based on metrics progress
    const avgProgress = metrics.reduce((sum, metric) => {
      const progress = Math.min((metric.value / metric.target) * 100, 100);
      return sum + progress;
    }, 0) / metrics.length;
    
    setHealthScore(Math.round(avgProgress));
  };

  useEffect(() => {
    refreshHealthScore();
  }, [metrics]);

  const value: HealthDataContextType = {
    metrics,
    goals,
    alerts,
    healthScore,
    updateMetric,
    addGoal,
    updateGoal,
    markAlertAsRead,
    refreshHealthScore,
  };

  return (
    <HealthDataContext.Provider value={value}>
      {children}
    </HealthDataContext.Provider>
  );
};

export const useHealthData = (): HealthDataContextType => {
  const context = useContext(HealthDataContext);
  if (context === undefined) {
    throw new Error('useHealthData must be used within a HealthDataProvider');
  }
  return context;
};