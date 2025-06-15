
import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { PieChart, Pie, Cell, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { AlertTriangle, TrendingUp, Shield, Eye } from 'lucide-react';

interface RiskPredictionsProps {
  dateRange: string;
}

export const RiskPredictions: React.FC<RiskPredictionsProps> = ({ dateRange }) => {
  const riskDistribution = [
    { name: 'Low Risk', value: 65, color: '#10b981', count: 13 },
    { name: 'Medium Risk', value: 25, color: '#f59e0b', count: 5 },
    { name: 'High Risk', value: 10, color: '#ef4444', count: 2 },
  ];

  const riskProgression = [
    { date: '2024-01-01', low: 70, medium: 25, high: 5 },
    { date: '2024-01-15', low: 68, medium: 27, high: 5 },
    { date: '2024-02-01', low: 65, medium: 25, high: 10 },
    { date: '2024-02-15', low: 65, medium: 25, high: 10 },
  ];

  const alerts = [
    {
      id: 1,
      type: 'skin',
      title: 'Mole Changes Detected',
      description: 'Asymmetrical changes noted in lesion ID: MSK-2024-003',
      risk: 'high',
      date: '2024-02-14',
      action: 'Schedule dermatologist appointment',
    },
    {
      id: 2,
      type: 'cardiovascular',
      title: 'Elevated Resting Heart Rate',
      description: 'Average resting HR increased by 15% over 7 days',
      risk: 'medium',
      date: '2024-02-13',
      action: 'Monitor for 3 more days',
    },
    {
      id: 3,
      type: 'metabolic',
      title: 'Sleep Pattern Disruption',
      description: 'Sleep quality decreased by 20% this week',
      risk: 'medium',
      date: '2024-02-12',
      action: 'Review sleep hygiene',
    },
  ];

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case 'high': return 'text-red-600 bg-red-50 border-red-200';
      case 'medium': return 'text-orange-600 bg-orange-50 border-orange-200';
      case 'low': return 'text-green-600 bg-green-50 border-green-200';
      default: return 'text-gray-600 bg-gray-50 border-gray-200';
    }
  };

  const getRiskIcon = (risk: string) => {
    switch (risk) {
      case 'high': return <AlertTriangle className="w-4 h-4" />;
      case 'medium': return <TrendingUp className="w-4 h-4" />;
      case 'low': return <Shield className="w-4 h-4" />;
      default: return <Eye className="w-4 h-4" />;
    }
  };

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Risk Distribution */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <AlertTriangle className="w-5 h-5 text-orange-600" />
              Risk Category Distribution
            </CardTitle>
            <CardDescription>
              Current health risk assessment based on AI analysis
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex items-center justify-center mb-4">
              <ResponsiveContainer width="100%" height={200}>
                <PieChart>
                  <Pie
                    data={riskDistribution}
                    cx="50%"
                    cy="50%"
                    outerRadius={80}
                    paddingAngle={5}
                    dataKey="value"
                  >
                    {riskDistribution.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip formatter={(value) => `${value}%`} />
                </PieChart>
              </ResponsiveContainer>
            </div>
            
            <div className="space-y-2">
              {riskDistribution.map((item, index) => (
                <div key={index} className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <div 
                      className="w-3 h-3 rounded-full"
                      style={{ backgroundColor: item.color }}
                    />
                    <span className="text-sm font-medium">{item.name}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-sm text-gray-600">{item.count} items</span>
                    <span className="text-sm font-medium">{item.value}%</span>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Risk Progression */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <TrendingUp className="w-5 h-5 text-blue-600" />
              Risk Level Progression
            </CardTitle>
            <CardDescription>
              How your risk levels have changed over time
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={riskProgression}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="date" 
                  tickFormatter={(value) => new Date(value).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
                />
                <YAxis domain={[0, 100]} />
                <Tooltip 
                  labelFormatter={(value) => new Date(value).toLocaleDateString()}
                  formatter={(value, name) => [`${value}%`, name.charAt(0).toUpperCase() + name.slice(1) + ' Risk']}
                />
                <Line 
                  type="monotone" 
                  dataKey="low" 
                  stroke="#10b981" 
                  strokeWidth={2}
                  name="low"
                />
                <Line 
                  type="monotone" 
                  dataKey="medium" 
                  stroke="#f59e0b" 
                  strokeWidth={2}
                  name="medium"
                />
                <Line 
                  type="monotone" 
                  dataKey="high" 
                  stroke="#ef4444" 
                  strokeWidth={2}
                  name="high"
                />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      {/* AI Risk Alerts */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <AlertTriangle className="w-5 h-5 text-red-600" />
            AI Risk Alerts
          </CardTitle>
          <CardDescription>
            Recent health alerts and recommendations from AI analysis
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {alerts.map((alert) => (
              <div 
                key={alert.id} 
                className={`p-4 rounded-lg border ${getRiskColor(alert.risk)}`}
              >
                <div className="flex items-start justify-between">
                  <div className="flex items-start gap-3">
                    <div className="mt-1">
                      {getRiskIcon(alert.risk)}
                    </div>
                    <div className="flex-1">
                      <h4 className="font-semibold text-sm">{alert.title}</h4>
                      <p className="text-sm opacity-90 mt-1">{alert.description}</p>
                      <p className="text-xs mt-2 font-medium">
                        Recommended Action: {alert.action}
                      </p>
                    </div>
                  </div>
                  <div className="text-xs opacity-75">
                    {new Date(alert.date).toLocaleDateString()}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
