
import React from 'react';
import { WelcomeSection } from './home/WelcomeSection';
import { AchievementsSection } from './home/AchievementsSection';
import { QuickActionsSection } from './home/QuickActionsSection';
import { StatsSection } from './home/StatsSection';
import { WeeklyGoalsSection } from './home/WeeklyGoalsSection';
import { HealthAlertsSection } from './home/HealthAlertsSection';
import { HealthManagementSection } from './home/HealthManagementSection';
import { NotificationCenter } from './notifications/NotificationCenter';

const HomeScreen = () => {
  const healthScore = 78;

  return (
    <div className="space-y-6">
      <WelcomeSection healthScore={healthScore} />
      <AchievementsSection />
      <QuickActionsSection />
      <StatsSection />
      <WeeklyGoalsSection />
      <HealthAlertsSection />
      <NotificationCenter />
      <HealthManagementSection />
    </div>
  );
};

export default HomeScreen;
