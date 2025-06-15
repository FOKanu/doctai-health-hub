
import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { RotateCcw, CheckCircle, AlertTriangle, TrendingUp, Calendar } from 'lucide-react';

interface RescanComplianceProps {
  dateRange: string;
}

export const RescanCompliance: React.FC<RescanComplianceProps> = ({ dateRange }) => {
  const weeklyCompliance = [
    { week: 'Week 1', completed: 6, missed: 1, rate: 85.7 },
    { week: 'Week 2', completed: 7, missed: 0, rate: 100 },
    { week: 'Week 3', completed: 5, missed: 2, rate: 71.4 },
    { week: 'Week 4', completed: 7, missed: 0, rate: 100 },
  ];

  const dailyCompliance = [
    { date: '2024-02-08', completed: 1, missed: 0 },
    { date: '2024-02-09', completed: 1, missed: 0 },
    { date: '2024-02-10', completed: 0, missed: 1 },
    { date: '2024-02-11', completed: 1, missed: 0 },
    { date: '2024-02-12', completed: 1, missed: 0 },
    { date: '2024-02-13', completed: 1, missed: 0 },
    { date: '2024-02-14', completed: 1, missed: 0 },
  ];

  const rescheduleHistory = [
    { id: 1, type: 'Skin Lesion Monitor', originalDate: '2024-02-10', newDate: '2024-02-12', reason: 'User rescheduled' },
    { id: 2, type: 'Heart Rate Check', originalDate: '2024-02-08', newDate: '2024-02-09', reason: 'Reminder dismissed' },
    { id: 3, type: 'Weight Tracking', originalDate: '2024-02-06', newDate: '2024-02-07', reason: 'Technical issue' },
  ];

  const upcomingRescans = [
    { id: 1, type: 'Full Body Skin Check', dueDate: '2024-02-16', lesionCount: 5, priority: 'high' },
    { id: 2, type: 'Mole Asymmetry Monitor', dueDate: '2024-02-18', lesionCount: 2, priority: 'medium' },
    { id: 3, type: 'Heart Rate Baseline', dueDate: '2024-02-20', lesionCount: 0, priority: 'low' },
  ];

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'high': return 'text-red-600 bg-red-50 border-red-200';
      case 'medium': return 'text-orange-600 bg-orange-50 border-orange-200';
      case 'low': return 'text-green-600 bg-green-50 border-green-200';
      default: return 'text-gray-600 bg-gray-50 border-gray-200';
    }
  };

  const currentStreak = 7;
  const averageCompliance = 89.3;

  return (
    <div className="space-y-6">
      {/* Compliance Overview */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-2">
              <CheckCircle className="w-5 h-5 text-green-600" />
              <div>
                <div className="text-2xl font-bold text-green-600">{currentStreak}</div>
                <div className="text-sm text-gray-600">Day Streak</div>
              </div>
            </div>
          </CardContent>
        </Card>
        
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-2">
              <TrendingUp className="w-5 h-5 text-blue-600" />
              <div>
                <div className="text-2xl font-bold text-blue-600">{averageCompliance}%</div>
                <div className="text-sm text-gray-600">Avg Compliance</div>
              </div>
            </div>
          </CardContent>
        </Card>
        
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-2">
              <RotateCcw className="w-5 h-5 text-purple-600" />
              <div>
                <div className="text-2xl font-bold text-purple-600">25</div>
                <div className="text-sm text-gray-600">Completed</div>
              </div>
            </div>
          </CardContent>
        </Card>
        
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-2">
              <AlertTriangle className="w-5 h-5 text-orange-600" />
              <div>
                <div className="text-2xl font-bold text-orange-600">3</div>
                <div className="text-sm text-gray-600">Missed</div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Weekly Compliance Rate */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BarChart className="w-5 h-5 text-blue-600" />
              Weekly Compliance Rate
            </CardTitle>
            <CardDescription>
              Rescan completion rate by week
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={weeklyCompliance}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="week" />
                <YAxis domain={[0, 100]} />
                <Tooltip 
                  formatter={(value, name) => [
                    name === 'rate' ? `${value}%` : value,
                    name === 'completed' ? 'Completed' : name === 'missed' ? 'Missed' : 'Compliance Rate'
                  ]}
                />
                <Bar dataKey="completed" fill="#10b981" name="completed" />
                <Bar dataKey="missed" fill="#ef4444" name="missed" />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Daily Tracking */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Calendar className="w-5 h-5 text-green-600" />
              Daily Rescan Activity
            </CardTitle>
            <CardDescription>
              Last 7 days rescan completion tracking
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={dailyCompliance}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="date" 
                  tickFormatter={(value) => new Date(value).toLocaleDateString('en-US', { weekday: 'short' })}
                />
                <YAxis />
                <Tooltip 
                  labelFormatter={(value) => new Date(value).toLocaleDateString()}
                  formatter={(value, name) => [value, name === 'completed' ? 'Completed' : 'Missed']}
                />
                <Bar dataKey="completed" fill="#10b981" name="completed" />
                <Bar dataKey="missed" fill="#ef4444" name="missed" />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      {/* Upcoming Rescans */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <RotateCcw className="w-5 h-5 text-purple-600" />
            Upcoming Rescans
          </CardTitle>
          <CardDescription>
            Scheduled health monitoring activities
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {upcomingRescans.map((rescan) => (
              <div key={rescan.id} className="flex items-center justify-between p-3 border rounded-lg">
                <div className="flex items-center gap-3">
                  <div className="p-2 bg-purple-100 rounded-full">
                    <RotateCcw className="w-4 h-4 text-purple-600" />
                  </div>
                  <div>
                    <div className="font-medium">{rescan.type}</div>
                    <div className="text-sm text-gray-600">
                      {rescan.lesionCount > 0 ? `${rescan.lesionCount} lesions to monitor` : 'General health check'}
                    </div>
                  </div>
                </div>
                <div className="text-right">
                  <div className={`inline-flex px-2 py-1 rounded-full text-xs font-medium border ${getPriorityColor(rescan.priority)}`}>
                    {rescan.priority.charAt(0).toUpperCase() + rescan.priority.slice(1)} Priority
                  </div>
                  <div className="text-sm text-gray-600 mt-1">Due: {rescan.dueDate}</div>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Reschedule History */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <AlertTriangle className="w-5 h-5 text-orange-600" />
            Recent Reschedules
          </CardTitle>
          <CardDescription>
            History of missed or rescheduled rescan reminders
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {rescheduleHistory.map((item) => (
              <div key={item.id} className="flex items-center justify-between p-3 bg-orange-50 border border-orange-200 rounded-lg">
                <div className="flex items-center gap-3">
                  <AlertTriangle className="w-4 h-4 text-orange-600" />
                  <div>
                    <div className="font-medium">{item.type}</div>
                    <div className="text-sm text-gray-600">{item.reason}</div>
                  </div>
                </div>
                <div className="text-right text-sm">
                  <div className="text-gray-600">From: {item.originalDate}</div>
                  <div className="text-gray-600">To: {item.newDate}</div>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
