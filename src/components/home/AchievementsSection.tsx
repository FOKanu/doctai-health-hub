
import React from 'react';
import { Trophy, Target, Star, Award } from 'lucide-react';

interface Achievement {
  icon: React.ComponentType<any>;
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
    <div className="bg-white rounded-lg p-6 shadow-sm border border-gray-200">
      <h2 className="text-xl font-semibold text-gray-900 mb-4">Recent Achievements</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {achievements.map((achievement, index) => (
          <div key={index} className={`p-4 rounded-lg border-2 ${
            achievement.unlocked 
              ? 'border-yellow-300 bg-yellow-50' 
              : 'border-gray-200 bg-gray-50'
          }`}>
            <div className="flex items-center space-x-3">
              <achievement.icon className={`w-8 h-8 ${
                achievement.unlocked ? 'text-yellow-600' : 'text-gray-400'
              }`} />
              <div>
                <h3 className={`font-semibold text-sm ${
                  achievement.unlocked ? 'text-gray-900' : 'text-gray-500'
                }`}>
                  {achievement.title}
                </h3>
                <p className={`text-xs ${
                  achievement.unlocked ? 'text-gray-600' : 'text-gray-400'
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
