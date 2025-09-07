
import React from 'react';
import { WelcomeSection } from './home/WelcomeSection';
import { AchievementsSection } from './home/AchievementsSection';
import { QuickActionsSection } from './home/QuickActionsSection';
import { StatsSection } from './home/StatsSection';
import { InteractiveMetricsSection } from './home/InteractiveMetricsSection';
import { HealthInsightsSection } from './home/HealthInsightsSection';
import { HealthAlertsSection } from './home/HealthAlertsSection';
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
      <HealthInsightsSection />
      <AchievementsSection />
      <HealthAlertsSection />
      <NotificationCenter />
    </div>
  );
};

export default HomeScreen;
