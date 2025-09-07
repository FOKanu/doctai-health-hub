
import React from 'react';
import { WelcomeSection } from './home/WelcomeSection';
import { AchievementsSection } from './home/AchievementsSection';
import { QuickActionsSection } from './home/QuickActionsSection';
import { StatsSection } from './home/StatsSection';
import { InteractiveMetricsSection } from './home/InteractiveMetricsSection';
import { WeeklyGoalsSection } from './home/WeeklyGoalsSection';
import { HealthInsightsSection } from './home/HealthInsightsSection';
import { HealthAlertsSection } from './home/HealthAlertsSection';
import { HealthManagementSection } from './home/HealthManagementSection';
import { NotificationCenter } from './notifications/NotificationCenter';
import { useHealthData } from '@/contexts/HealthDataContext';

const HomeScreen = () => {
  const { healthScore } = useHealthData();

  return (
    <div className="space-y-6">
      <WelcomeSection healthScore={healthScore} />
      <QuickActionsSection />
      <StatsSection />
      <InteractiveMetricsSection />
      <WeeklyGoalsSection />
      <HealthInsightsSection />
      <HealthManagementSection />
      <AchievementsSection />
      <HealthAlertsSection />
      <NotificationCenter />
    </div>
  );
};

export default HomeScreen;
