import React from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import {
  Home,
  Settings,
  Activity,
  Database,
  Shield,
  Terminal,
  GitBranch,
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
  BarChart3,
  FileCode,
  Cloud,
  Lock,
  Users,
  Globe
} from 'lucide-react';
import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarHeader,
  SidebarFooter,
} from '@/components/ui/sidebar';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import { useAuth } from '@/contexts/AuthContext';
import { RoleBasedMobileNavigation } from './RoleBasedMobileNavigation';

export function EngineerLayout({ children }: { children: React.ReactNode }) {
  const navigate = useNavigate();
  const location = useLocation();
  const { user } = useAuth();

  // Engineer-specific navigation
  const mainNavigation = [
    {
      title: "System Dashboard",
      url: "/engineer/dashboard",
      icon: Home,
      description: "Uptime, latency, error logs"
    },
    {
      title: "Development Tools",
      url: "/engineer/dev-tools",
      icon: Code,
      description: "Git, CI/CD, test suite"
    },
    {
      title: "Data Management",
      url: "/engineer/data",
      icon: Database,
      description: "Backup, analytics, data flow"
    },
    {
      title: "Security & Compliance",
      url: "/engineer/security",
      icon: Shield,
      description: "Auth logs, breach warnings"
    },
    {
      title: "Logs & Alerts",
      url: "/engineer/logs",
      icon: Terminal,
      description: "Real-time monitoring"
    }
  ];

  const systemNavigation = [
    {
      title: "Infrastructure",
      url: "/engineer/infrastructure",
      icon: Server,
      color: "text-blue-500"
    },
    {
      title: "Performance",
      url: "/engineer/performance",
      icon: Cpu,
      color: "text-green-500"
    },
    {
      title: "Storage",
      url: "/engineer/storage",
      icon: HardDrive,
      color: "text-purple-500"
    },
    {
      title: "Network",
      url: "/engineer/network",
      icon: Network,
      color: "text-orange-500"
    }
  ];

  const secondaryItems = [
    {
      title: "Settings",
      url: "/engineer/settings",
      icon: Settings
    }
  ];

  const getInitials = (name: string) => {
    return name
      .split(' ')
      .map(n => n[0])
      .join('')
      .toUpperCase()
      .slice(0, 2);
  };

  return (
    <div className="min-h-screen flex w-full bg-gray-900 text-gray-100">
      <Sidebar className="hidden md:flex border-r border-gray-700 bg-gray-800">
        <SidebarHeader className="p-6 border-b border-gray-700">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-blue-600 rounded-xl flex items-center justify-center shadow-lg">
              <Terminal className="w-6 h-6 text-white" />
            </div>
            <div>
              <h2 className="text-xl font-bold text-white">DoctAI</h2>
              <p className="text-sm text-blue-400 font-medium">Engineering Portal</p>
            </div>
          </div>

          {/* Engineer Info */}
          {user && (
            <div className="mt-4 p-3 bg-gray-700 rounded-lg">
              <div className="flex items-center space-x-3">
                <Avatar className="h-8 w-8">
                  <AvatarImage src={user.avatar} alt={user.name} />
                  <AvatarFallback className="bg-blue-500 text-white text-sm">
                    {getInitials(user.name)}
                  </AvatarFallback>
                </Avatar>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-semibold text-white truncate">
                    {user.name}
                  </p>
                  {user.department && (
                    <p className="text-xs text-blue-400 font-medium">
                      {user.department}
                    </p>
                  )}
                </div>
              </div>
            </div>
          )}
        </SidebarHeader>

        <SidebarContent className="flex-1">
          {/* Main Navigation */}
          <SidebarGroup>
            <SidebarGroupLabel className="text-xs font-semibold text-gray-400 uppercase tracking-wider px-6 py-2">
              Main Menu
            </SidebarGroupLabel>
            <SidebarGroupContent>
              <SidebarMenu>
                {mainNavigation.map((item) => (
                  <SidebarMenuItem key={item.title}>
                    <SidebarMenuButton
                      asChild
                      isActive={location.pathname === item.url}
                      className="hover:bg-gray-700 data-[active=true]:bg-blue-600 data-[active=true]:text-white"
                    >
                      <button
                        onClick={() => navigate(item.url)}
                        className="w-full text-left p-3 rounded-lg transition-all duration-200"
                      >
                        <div className="flex items-center space-x-3">
                          <item.icon className="w-5 h-5" />
                          <div className="flex-1">
                            <span className="font-medium">{item.title}</span>
                            <p className="text-xs text-gray-400 mt-1">{item.description}</p>
                          </div>
                        </div>
                      </button>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                ))}
              </SidebarMenu>
            </SidebarGroupContent>
          </SidebarGroup>

          {/* System Navigation */}
          <SidebarGroup>
            <SidebarGroupLabel className="text-xs font-semibold text-gray-400 uppercase tracking-wider px-6 py-2">
              System Tools
            </SidebarGroupLabel>
            <SidebarGroupContent>
              <SidebarMenu>
                {systemNavigation.map((item) => (
                  <SidebarMenuItem key={item.title}>
                    <SidebarMenuButton
                      asChild
                      isActive={location.pathname === item.url}
                      className="hover:bg-gray-700 data-[active=true]:bg-blue-600 data-[active=true]:text-white"
                    >
                      <button
                        onClick={() => navigate(item.url)}
                        className="w-full text-left p-3 rounded-lg transition-all duration-200"
                      >
                        <div className="flex items-center space-x-3">
                          <item.icon className={`w-5 h-5 ${item.color}`} />
                          <span className="font-medium">{item.title}</span>
                        </div>
                      </button>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                ))}
              </SidebarMenu>
            </SidebarGroupContent>
          </SidebarGroup>
        </SidebarContent>

        <SidebarFooter className="p-4 border-t border-gray-700">
          <SidebarMenu>
            {secondaryItems.map((item) => (
              <SidebarMenuItem key={item.title}>
                <SidebarMenuButton
                  asChild
                  isActive={location.pathname === item.url}
                  className="hover:bg-gray-700 data-[active=true]:bg-blue-600 data-[active=true]:text-white"
                >
                  <button
                    onClick={() => navigate(item.url)}
                    className="w-full text-left p-3 rounded-lg transition-all duration-200"
                  >
                    <div className="flex items-center space-x-3">
                      <item.icon className="w-5 h-5" />
                      <span className="font-medium">{item.title}</span>
                    </div>
                  </button>
                </SidebarMenuButton>
              </SidebarMenuItem>
            ))}
          </SidebarMenu>
        </SidebarFooter>
      </Sidebar>

      <div className="flex-1 flex flex-col">
        {/* Engineer Header */}
        <header className="bg-gray-800 border-b border-gray-700 px-6 py-4 shadow-lg">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-6">
              <div>
                <h1 className="text-2xl font-bold text-white">Engineering Dashboard</h1>
                <p className="text-sm text-blue-400">
                  {user?.department ? `${user.department} • ` : ''}System Health & Monitoring
                </p>
              </div>

              {/* Quick Action Buttons */}
              <div className="flex items-center space-x-3">
                <Button size="sm" className="bg-blue-600 hover:bg-blue-700 text-white">
                  <Zap className="w-4 h-4 mr-2" />
                  Deploy
                </Button>
                <Button size="sm" variant="outline" className="border-gray-600 text-gray-300 hover:bg-gray-700">
                  <Monitor className="w-4 h-4 mr-2" />
                  Monitor
                </Button>
                <Button size="sm" variant="outline" className="border-gray-600 text-gray-300 hover:bg-gray-700">
                  <Bug className="w-4 h-4 mr-2" />
                  Debug
                </Button>
              </div>
            </div>

            <div className="flex items-center space-x-4">
              {/* System Status Indicators */}
              <div className="flex items-center space-x-4">
                <div className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                  <span className="text-sm text-gray-300">System Online</span>
                </div>

                <div className="flex items-center space-x-2">
                  <Clock className="w-4 h-4 text-gray-400" />
                  <span className="text-sm text-gray-300">Uptime: 99.9%</span>
                </div>
              </div>

              {/* Alerts */}
              <Button variant="ghost" size="sm" className="relative text-gray-300 hover:text-white">
                <AlertTriangle className="w-5 h-5 text-orange-400" />
                <Badge className="absolute -top-1 -right-1 h-5 w-5 rounded-full bg-red-500 text-xs">
                  2
                </Badge>
              </Button>
            </div>
          </div>
        </header>

        {/* Main Content */}
        <main className="flex-1 p-6 overflow-auto bg-gray-900">
          <div className="max-w-7xl mx-auto">
            {children}
          </div>
        </main>
      </div>
      <RoleBasedMobileNavigation role="engineer" />
    </div>
  );
}
