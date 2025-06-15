
import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { AlertTriangle } from 'lucide-react';
import { getRiskColor, getRiskIcon } from './riskUtils';

interface Alert {
  id: number;
  type: string;
  title: string;
  description: string;
  risk: string;
  date: string;
  action: string;
}

interface RiskAlertsProps {
  alerts: Alert[];
}

export const RiskAlerts: React.FC<RiskAlertsProps> = ({ alerts }) => {
  return (
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
  );
};
