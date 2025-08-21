import React, { useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '@/contexts/AuthContext';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { 
  Sidebar, 
  SidebarContent, 
  SidebarHeader, 
  SidebarMenu, 
  SidebarMenuButton, 
  SidebarMenuItem, 
  SidebarGroup, 
  SidebarGroupContent, 
  SidebarGroupLabel,
  SidebarFooter,
  SidebarProvider,
  SidebarTrigger
} from '@/components/ui/sidebar';
import { 
  Home, 
  Users, 
  Shield, 
  Settings, 
  Activity, 
  Database, 
  Server, 
  AlertTriangle, 
  BarChart3,
  UserCheck,
  UserX,
  LogOut,
  Menu,
  X,
  Crown
} from 'lucide-react';

export function AdminLayout({ children }: { children: React.ReactNode }) {
  const navigate = useNavigate();
  const location = useLocation();
  const { user, logout } = useAuth();
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  // Admin-specific navigation
  const mainNavigation = [
    {
      title: "Dashboard",
      url: "/admin/dashboard",
      icon: Home,
      description: "System overview"
    },
    {
      title: "User Management",
      url: "/admin/users",
      icon: Users,
      description: "Manage all users"
    },
    {
      title: "Security",
      url: "/admin/security",
      icon: Shield,
      description: "Security settings"
    },
    {
      title: "System Health",
      url: "/admin/system",
      icon: Activity,
      description: "Monitor system status"
    },
    {
      title: "Data Management",
      url: "/admin/data",
      icon: Database,
      description: "Database and storage"
    },
    {
      title: "Analytics",
      url: "/admin/analytics",
      icon: BarChart3,
      description: "System analytics"
    }
  ];

  const secondaryItems = [
    {
      title: "Settings",
      url: "/admin/settings",
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

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  return (
    <SidebarProvider>
      {/* Skip to content link for accessibility */}
      <a
        href="#main-content"
        className="sr-only focus:not-sr-only focus:absolute focus:top-4 focus:left-4 bg-blue-600 text-white px-4 py-2 rounded-lg z-50"
        aria-label="Skip to main content"
      >
        Skip to main content
      </a>
      
      <div className="min-h-screen flex w-full bg-gradient-to-br from-gray-50 to-gray-100">
        {/* Mobile Header with Hamburger */}
        <div className="md:hidden fixed top-0 left-0 right-0 z-50 bg-white/95 backdrop-blur-sm border-b border-gray-200 px-4 py-3">
          <div className="flex items-center justify-between">
            <button
              onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
              className="p-2 rounded-lg hover:bg-gray-50 transition-colors"
              aria-label="Toggle sidebar"
            >
              {sidebarCollapsed ? <Menu className="w-5 h-5" /> : <X className="w-5 h-5" />}
            </button>
            <div className="flex items-center space-x-2">
              <Crown className="w-6 h-6 text-purple-600" />
              <span className="text-lg font-bold text-gray-900">DoctAI Admin</span>
            </div>
            <div className="w-9"></div> {/* Spacer for centering */}
          </div>
        </div>

        {/* Sidebar */}
        <Sidebar className={`hidden md:flex border-r border-gray-200 bg-white/95 backdrop-blur-sm transition-all duration-300 ${
          sidebarCollapsed ? 'w-16' : 'w-80'
        }`}>
          <SidebarHeader className="p-8 border-b border-gray-200">
            <div className="flex items-center space-x-4">
              <div className="w-12 h-12 bg-purple-600 rounded-xl flex items-center justify-center shadow-lg">
                <Crown className="w-7 h-7 text-white" />
              </div>
              {!sidebarCollapsed && (
                <div>
                  <h2 className="text-2xl font-bold text-gray-900">DoctAI</h2>
                  <p className="text-sm text-purple-600 font-medium">Admin Portal</p>
                </div>
              )}
            </div>

            {/* Admin Info */}
            {user && !sidebarCollapsed && (
              <div className="mt-6 p-4 bg-purple-50 rounded-xl">
                <div className="flex items-center space-x-4">
                  <Avatar className="h-10 w-10">
                    <AvatarImage src={user.avatar} alt={user.name} />
                    <AvatarFallback className="bg-purple-500 text-white text-sm">
                      {getInitials(user.name)}
                    </AvatarFallback>
                  </Avatar>
                  <div className="flex-1 min-w-0">
                    <p className="text-base font-semibold text-gray-900 truncate">
                      {user.name}
                    </p>
                    <div className="flex items-center space-x-2 mt-1">
                      <Badge variant="secondary" className="text-xs">
                        Admin
                      </Badge>
                      <span className="text-xs text-purple-600 font-medium">
                        System Administrator
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </SidebarHeader>

          <SidebarContent className="flex-1">
            {/* Main Navigation */}
            <SidebarGroup>
              {!sidebarCollapsed && (
                <SidebarGroupLabel className="text-xs font-semibold text-gray-500 uppercase tracking-wider px-8 py-4">
                  Administration
                </SidebarGroupLabel>
              )}
              <SidebarGroupContent className="px-4">
                <SidebarMenu className="space-y-2">
                  {mainNavigation.map((item) => (
                    <SidebarMenuItem key={item.title}>
                      <SidebarMenuButton
                        asChild
                        isActive={location.pathname === item.url}
                        className="hover:bg-purple-50 data-[active=true]:bg-purple-100 data-[active=true]:text-purple-700 data-[active=true]:border-r-2 data-[active=true]:border-purple-600"
                      >
                        <button
                          onClick={() => navigate(item.url)}
                          className="w-full text-left p-4 rounded-lg transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:ring-offset-1"
                          title={sidebarCollapsed ? item.title : item.description}
                          aria-label={`Navigate to ${item.title} - ${item.description}`}
                          data-testid={`nav-${item.title.toLowerCase().replace(/\s+/g, '-')}`}
                          tabIndex={0}
                          onKeyDown={(e) => {
                            if (e.key === 'Enter' || e.key === ' ') {
                              e.preventDefault();
                              navigate(item.url);
                            }
                          }}
                        >
                          <div className="flex items-center space-x-3">
                            <item.icon className="w-5 h-5 flex-shrink-0" />
                            {!sidebarCollapsed && (
                              <div className="flex-1 min-w-0">
                                <span className="block text-sm font-medium">{item.title}</span>
                                <span className="block text-xs text-gray-500 mt-1">{item.description}</span>
                              </div>
                            )}
                          </div>
                        </button>
                      </SidebarMenuButton>
                    </SidebarMenuItem>
                  ))}
                </SidebarMenu>
              </SidebarGroupContent>
            </SidebarGroup>

            {/* Secondary Navigation */}
            <SidebarGroup>
              {!sidebarCollapsed && (
                <SidebarGroupLabel className="text-xs font-semibold text-gray-500 uppercase tracking-wider px-8 py-4">
                  System
                </SidebarGroupLabel>
              )}
              <SidebarGroupContent className="px-4">
                <SidebarMenu className="space-y-2">
                  {secondaryItems.map((item) => (
                    <SidebarMenuItem key={item.title}>
                      <SidebarMenuButton
                        asChild
                        isActive={location.pathname === item.url}
                        className="hover:bg-gray-50 data-[active=true]:bg-gray-100 data-[active=true]:text-gray-700"
                      >
                        <button
                          onClick={() => navigate(item.url)}
                          className="w-full text-left p-4 rounded-lg transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-offset-1"
                          title={sidebarCollapsed ? item.title : undefined}
                        >
                          <div className="flex items-center space-x-3">
                            <item.icon className="w-5 h-5 flex-shrink-0" />
                            {!sidebarCollapsed && (
                              <span className="text-sm font-medium">{item.title}</span>
                            )}
                          </div>
                        </button>
                      </SidebarMenuButton>
                    </SidebarMenuItem>
                  ))}
                </SidebarMenu>
              </SidebarGroupContent>
            </SidebarGroup>
          </SidebarContent>

          {/* Footer */}
          <SidebarFooter className="p-4 border-t border-gray-200">
            <Button
              onClick={handleLogout}
              variant="ghost"
              className="w-full justify-start text-red-600 hover:text-red-700 hover:bg-red-50"
            >
              <LogOut className="w-5 h-5 mr-3" />
              {!sidebarCollapsed && <span>Logout</span>}
            </Button>
          </SidebarFooter>
        </Sidebar>

        {/* Main Content */}
        <main id="main-content" className="flex-1 md:ml-0 transition-all duration-300">
          {/* Desktop Header */}
          <div className="hidden md:flex items-center justify-between p-6 bg-white/80 backdrop-blur-sm border-b border-gray-200">
            <div className="flex items-center space-x-4">
              <SidebarTrigger className="md:hidden" />
              <div>
                <h1 className="text-2xl font-bold text-gray-900">Admin Portal</h1>
                <p className="text-sm text-gray-600">System administration and management</p>
              </div>
            </div>
            
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                <span className="text-sm text-gray-600">System Online</span>
              </div>
              <Badge variant="outline" className="text-xs">
                Admin Mode
              </Badge>
            </div>
          </div>

          {/* Page Content */}
          <div className="p-6">
            {children}
          </div>
        </main>
      </div>
    </SidebarProvider>
  );
}
