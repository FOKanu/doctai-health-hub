
import React from 'react';
import { Trophy, Target, Star, Award } from 'lucide-react';

interface Achievement {
  icon: React.ComponentType<{ className?: string }>;
  title: string;
  description: string;
  unlocked: boolean;
}

export const AchievementsSection: React.FC = () => {
  const achievements: Achievement[] = [
    { icon: Trophy, title: 'Health Warrior', description: 'Completed 30 health checks', unlocked: true },
    { icon: Target, title: 'Consistency Champion', description: 'Logged health data for 7 days straight', unlocked: true },
    { icon: Star, title: 'Early Bird', description: 'Complete morning health routine', unlocked: false },
    { icon: Award, title: 'Wellness Expert', description: 'Achieve 90% health score', unlocked: false },
  ];

  return (
      <div className="bg-card rounded-lg p-6 shadow-sm border border-border">
        <h2 className="text-xl font-semibold text-foreground mb-4">Recent Achievements</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {achievements.map((achievement, index) => (
            <div key={index} className={`p-4 rounded-lg border-2 ${
              achievement.unlocked
                ? 'border-primary bg-accent'
                : 'border-border bg-muted'
            }`}>
              <div className="flex items-center space-x-3">
                <achievement.icon className={`w-8 h-8 ${
                  achievement.unlocked ? 'text-primary' : 'text-muted-foreground'
                }`} />
                <div>
                  <h3 className={`font-semibold text-sm ${
                    achievement.unlocked ? 'text-foreground' : 'text-muted-foreground'
                  }`}>
                    {achievement.title}
                  </h3>
                  <p className={`text-xs ${
                    achievement.unlocked ? 'text-muted-foreground' : 'text-muted-foreground'
                  }`}>
                    {achievement.description}
                  </p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
  );
};
