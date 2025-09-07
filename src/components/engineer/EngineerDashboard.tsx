import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import {
  Server,
  Cpu,
  HardDrive,
  Network,
  AlertTriangle,
  CheckCircle,
  Clock,
  TrendingUp,
  Code,
  Bug,
  Zap,
  Monitor,
  Database,
  Shield,
  Activity,
  GitBranch,
  Cloud,
  Lock,
  Users,
  Globe
} from 'lucide-react';
import { useAuth } from '@/contexts/AuthContext';

function EngineerDashboard() {
  const { user } = useAuth();

  // System metrics
  const systemMetrics = [
    {
      title: "CPU Usage",
      value: "68%",
      change: "+5%",
      icon: Cpu,
      color: "text-blue-400",
      bgColor: "bg-blue-900/20",
      status: "normal"
    },
    {
      title: "Memory Usage",
      value: "82%",
      change: "+12%",
      icon: HardDrive,
      color: "text-yellow-400",
      bgColor: "bg-yellow-900/20",
      status: "warning"
    },
    {
      title: "Network",
      value: "45%",
      change: "-8%",
      icon: Network,
      color: "text-green-400",
      bgColor: "bg-green-900/20",
      status: "good"
    },
    {
      title: "Storage",
      value: "73%",
      change: "+3%",
      icon: Server,
      color: "text-purple-400",
      bgColor: "bg-purple-900/20",
      status: "normal"
    }
  ];

  const recentDeployments = [
    {
      id: "1",
      service: "API Gateway",
      status: "success",
      time: "2 minutes ago",
      duration: "45s",
      commit: "a1b2c3d"
    },
    {
      id: "2",
      service: "Database Migration",
      status: "success",
      time: "15 minutes ago",
      duration: "2m 30s",
      commit: "e4f5g6h"
    },
    {
      id: "3",
      service: "Frontend Build",
      status: "failed",
      time: "1 hour ago",
      duration: "1m 15s",
      commit: "i7j8k9l"
    }
  ];

  const activeAlerts = [
    {
      id: "1",
      type: "High CPU Usage",
      severity: "warning",
      service: "API Server",
      time: "5 minutes ago",
      description: "CPU usage exceeded 85% threshold"
    },
    {
      id: "2",
      type: "Database Connection",
      severity: "error",
      service: "Main DB",
      time: "12 minutes ago",
      description: "Connection pool exhausted"
    },
    {
      id: "3",
      type: "Memory Leak",
      severity: "warning",
      service: "Worker Process",
      time: "25 minutes ago",
      description: "Memory usage growing steadily"
    }
  ];

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'success':
        return 'text-green-400';
      case 'failed':
        return 'text-red-400';
      case 'warning':
        return 'text-yellow-400';
      default:
        return 'text-gray-400';
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'error':
        return 'bg-red-500';
      case 'warning':
        return 'bg-yellow-500';
      case 'info':
        return 'bg-blue-500';
      default:
        return 'bg-gray-500';
    }
  };

  return (
    <div className="space-y-6">
      {/* Welcome Section */}
      <div className="bg-gradient-to-r from-gray-800 to-gray-900 rounded-lg p-6 border border-gray-700">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-white">System Dashboard</h1>
            <p className="text-gray-300 mt-1">
              {user?.department ? `${user.department} • ` : ''}Real-time monitoring & analytics
            </p>
          </div>
          <div className="flex items-center space-x-4">
            <div className="text-right">
              <p className="text-sm text-gray-400">Uptime</p>
              <p className="font-semibold text-white">99.9%</p>
            </div>
            <div className="w-3 h-3 bg-green-400 rounded-full"></div>
          </div>
        </div>
      </div>

      {/* System Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {systemMetrics.map((metric) => (
          <Card key={metric.title} className="bg-gray-800 border-gray-700 hover:bg-gray-750 transition-colors">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium text-gray-300">
                {metric.title}
              </CardTitle>
              <div className={`p-2 rounded-lg ${metric.bgColor}`}>
                <metric.icon className={`w-4 h-4 ${metric.color}`} />
              </div>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-white">{metric.value}</div>
              <p className="text-xs text-gray-400 mt-1">
                <span className={metric.change.startsWith('+') ? 'text-green-400' : 'text-red-400'}>
                  {metric.change}
                </span> from last hour
              </p>
              <Progress value={parseInt(metric.value)} className="mt-2" />
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Recent Deployments */}
        <Card className="bg-gray-800 border-gray-700">
          <CardHeader>
            <CardTitle className="flex items-center space-x-2 text-white">
              <Zap className="w-5 h-5" />
              <span>Recent Deployments</span>
            </CardTitle>
            <CardDescription className="text-gray-400">
              Latest deployment status and performance
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {recentDeployments.map((deployment) => (
                <div key={deployment.id} className="flex items-center justify-between p-3 rounded-lg bg-gray-700/50 border border-gray-600">
                  <div className="flex items-center space-x-3">
                    <div className={`w-2 h-2 rounded-full ${deployment.status === 'success' ? 'bg-green-400' : 'bg-red-400'}`}></div>
                    <div>
                      <p className="font-medium text-white">{deployment.service}</p>
                      <p className="text-xs text-gray-400">
                        {deployment.time} • {deployment.duration}
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Badge variant="outline" className="text-xs border-gray-600 text-gray-300">
                      {deployment.commit}
                    </Badge>
                    <CheckCircle className={`w-4 h-4 ${getStatusColor(deployment.status)}`} />
                  </div>
                </div>
              ))}
            </div>
            <Button variant="outline" className="w-full mt-4 border-gray-600 text-gray-300 hover:bg-gray-700">
              View All Deployments
            </Button>
          </CardContent>
        </Card>

        {/* Active Alerts */}
        <Card className="bg-gray-800 border-gray-700">
          <CardHeader>
            <CardTitle className="flex items-center space-x-2 text-white">
              <AlertTriangle className="w-5 h-5" />
              <span>Active Alerts</span>
            </CardTitle>
            <CardDescription className="text-gray-400">
              System alerts and notifications
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {activeAlerts.map((alert) => (
                <div key={alert.id} className="p-3 rounded-lg border border-gray-600 bg-gray-700/50">
                  <div className="flex items-start justify-between mb-2">
                    <Badge variant="secondary" className="text-xs bg-gray-600 text-gray-200">
                      {alert.type}
                    </Badge>
                    <div className={`w-2 h-2 rounded-full ${getSeverityColor(alert.severity)}`}></div>
                  </div>
                  <p className="text-sm font-medium text-white mb-1">
                    {alert.service}
                  </p>
                  <p className="text-xs text-gray-400 mb-2">
                    {alert.description}
                  </p>
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-gray-500">
                      {alert.time}
                    </span>
                    <Button size="sm" variant="ghost" className="text-xs text-gray-300 hover:text-white">
                      Investigate
                    </Button>
                  </div>
                </div>
              ))}
            </div>
            <Button variant="outline" className="w-full mt-4 border-gray-600 text-gray-300 hover:bg-gray-700">
              View All Alerts
            </Button>
          </CardContent>
        </Card>
      </div>

      {/* Quick Actions */}
      <Card className="bg-gray-800 border-gray-700">
        <CardHeader>
          <CardTitle className="text-white">Quick Actions</CardTitle>
          <CardDescription className="text-gray-400">
            Common development and monitoring tasks
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <Button variant="outline" className="h-20 flex-col space-y-2 border-gray-600 text-gray-300 hover:bg-gray-700">
              <Zap className="w-6 h-6" />
              <span className="text-sm">Deploy</span>
            </Button>
            <Button variant="outline" className="h-20 flex-col space-y-2 border-gray-600 text-gray-300 hover:bg-gray-700">
              <Monitor className="w-6 h-6" />
              <span className="text-sm">Monitor</span>
            </Button>
            <Button variant="outline" className="h-20 flex-col space-y-2 border-gray-600 text-gray-300 hover:bg-gray-700">
              <Bug className="w-6 h-6" />
              <span className="text-sm">Debug</span>
            </Button>
            <Button variant="outline" className="h-20 flex-col space-y-2 border-gray-600 text-gray-300 hover:bg-gray-700">
              <Database className="w-6 h-6" />
              <span className="text-sm">Database</span>
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* System Status */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Card className="bg-gray-800 border-gray-700">
          <CardHeader>
            <CardTitle className="text-white">Git Status</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-gray-400">Current Branch</span>
                <span className="text-white font-mono">main</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-gray-400">Last Commit</span>
                <span className="text-white font-mono">a1b2c3d</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-gray-400">Status</span>
                <Badge className="bg-green-500">Clean</Badge>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-gray-800 border-gray-700">
          <CardHeader>
            <CardTitle className="text-white">Security</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-gray-400">Firewall</span>
                <CheckCircle className="w-4 h-4 text-green-400" />
              </div>
              <div className="flex items-center justify-between">
                <span className="text-gray-400">SSL Certificates</span>
                <CheckCircle className="w-4 h-4 text-green-400" />
              </div>
              <div className="flex items-center justify-between">
                <span className="text-gray-400">Vulnerabilities</span>
                <Badge className="bg-green-500">0</Badge>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-gray-800 border-gray-700">
          <CardHeader>
            <CardTitle className="text-white">Performance</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-gray-400">Response Time</span>
                <span className="text-white">45ms</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-gray-400">Throughput</span>
                <span className="text-white">1.2k req/s</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-gray-400">Error Rate</span>
                <span className="text-green-400">0.1%</span>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

export default EngineerDashboard;
