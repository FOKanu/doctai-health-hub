import React, { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import {
  User,
  Calendar,
  Pill,
  FileText,
  Home,
  Activity,
  Settings,
  Mail,
  Stethoscope,
  Microscope,
  Shield,
  MessageSquare,
  AlertTriangle,
  TrendingUp,
  Users,
  Clipboard,
  Heart,
  Brain,
  Eye,
  Bone,
  Menu,
  X,
  Bot
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
  SidebarProvider,
} from '@/components/ui/sidebar';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import { useAuth } from '@/contexts/AuthContext';
import { ProviderStatusIndicator } from '@/components/provider/ProviderStatusIndicator';
import { NotificationIndicator } from '@/components/provider/NotificationIndicator';

export function ProviderLayout({ children }: { children: React.ReactNode }) {
  const navigate = useNavigate();
  const location = useLocation();
  const { user } = useAuth();
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  // Provider-specific navigation
  const mainNavigation = [
    {
      title: "Dashboard",
      url: "/provider/dashboard",
      icon: Home,
      description: "Overview & key metrics"
    },
    {
      title: "Patient Management",
      url: "/provider/patients",
      icon: Users,
      description: "Patient roster & records"
    },
    {
      title: "Schedule",
      url: "/provider/schedule",
      icon: Calendar,
      description: "Manage appointments"
    },
    {
      title: "Clinical Workflow",
      url: "/provider/clinical",
      icon: Clipboard,
      description: "Labs, prescriptions, vitals"
    },
    {
      title: "AI Diagnostic Support",
      url: "/provider/ai-support",
      icon: Bot,
      description: "Smart diagnosis & alerts"
    },
    {
      title: "Compliance Center",
      url: "/provider/compliance",
      icon: Shield,
      description: "HIPAA & audit logs"
    },
    {
      title: "Messages/Chat",
      url: "/provider/messages",
      icon: MessageSquare,
      description: "Patient & team communication"
    }
  ];

  const specialtyNavigation = [
    {
      title: "Cardiology",
      url: "/provider/specialty/cardiology",
      icon: Heart,
      color: "text-red-600"
    },
    {
      title: "Neurology",
      url: "/provider/specialty/neurology",
      icon: Brain,
      color: "text-purple-600"
    },
    {
      title: "Ophthalmology",
      url: "/provider/specialty/ophthalmology",
      icon: Eye,
      color: "text-blue-600"
    },
    {
      title: "Orthopedics",
      url: "/provider/specialty/orthopedics",
      icon: Bone,
      color: "text-orange-600"
    }
  ];

  const secondaryItems = [
    {
      title: "Settings",
      url: "/provider/settings",
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
    <SidebarProvider>
      {/* Skip to content link for accessibility */}
      <a
        href="#main-content"
        className="sr-only focus:not-sr-only focus:absolute focus:top-4 focus:left-4 bg-blue-600 text-white px-4 py-2 rounded-lg z-50"
        aria-label="Skip to main content"
      >
        Skip to main content
      </a>
      
      <div className="min-h-screen flex w-full bg-gradient-to-br from-blue-50 to-indigo-100">
        {/* Mobile Header with Hamburger */}
        <div className="md:hidden fixed top-0 left-0 right-0 z-50 bg-white/95 backdrop-blur-sm border-b border-blue-200 px-4 py-3">
          <div className="flex items-center justify-between">
            <button
              onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
              className="p-2 rounded-lg hover:bg-blue-50 transition-colors"
              aria-label="Toggle sidebar"
            >
              {sidebarCollapsed ? <Menu className="w-5 h-5" /> : <X className="w-5 h-5" />}
            </button>
            <div className="flex items-center space-x-2">
              <Stethoscope className="w-6 h-6 text-blue-600" />
              <span className="text-lg font-bold text-gray-900">DoctAI</span>
            </div>
            <div className="w-9"></div> {/* Spacer for centering */}
          </div>
        </div>

        {/* Sidebar */}
        <Sidebar className={`hidden md:flex border-r border-blue-200 bg-white/95 backdrop-blur-sm transition-all duration-300 ${
          sidebarCollapsed ? 'w-17' : 'w-80'
        }`}>
          <SidebarHeader className="p-8 border-b border-blue-200">
            <div className="flex items-center space-x-4">
              <div className="w-12 h-12 bg-blue-600 rounded-xl flex items-center justify-center shadow-lg">
                <Stethoscope className="w-7 h-7 text-white" />
              </div>
              {!sidebarCollapsed && (
                <div>
                  <h2 className="text-2xl font-bold text-gray-900">DoctAI</h2>
                  <p className="text-sm text-blue-600 font-medium">Provider Portal</p>
                </div>
              )}
            </div>

            {/* Provider Info */}
            {user && !sidebarCollapsed && (
              <div className="mt-6 p-4 bg-blue-50 rounded-xl">
                <div className="flex items-center space-x-4">
                  <Avatar className="h-10 w-10">
                    <AvatarImage src={user.avatar} alt={user.name} />
                    <AvatarFallback className="bg-blue-500 text-white text-sm">
                      {getInitials(user.name)}
                    </AvatarFallback>
                  </Avatar>
                  <div className="flex-1 min-w-0">
                    <p className="text-base font-semibold text-gray-900 truncate">
                      {user.name}
                    </p>
                    {user.specialty && (
                      <p className="text-sm text-blue-600 font-medium mt-1">
                        {user.specialty}
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
              {!sidebarCollapsed && (
                <SidebarGroupLabel className="text-xs font-semibold text-gray-500 uppercase tracking-wider px-8 py-4">
                  Main Menu
                </SidebarGroupLabel>
              )}
              <SidebarGroupContent className="px-4">
                <SidebarMenu className="space-y-2">
                  {mainNavigation.map((item) => (
                    <SidebarMenuItem key={item.title}>
                      <SidebarMenuButton
                        asChild
                        isActive={location.pathname === item.url}
                        className="hover:bg-blue-50 data-[active=true]:bg-blue-100 data-[active=true]:text-blue-700 data-[active=true]:border-r-2 data-[active=true]:border-blue-600"
                      >
                        <button
                          onClick={() => navigate(item.url)}
                          className="w-full text-left p-4 rounded-lg transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-1"
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
                          <div className={`flex items-center ${sidebarCollapsed ? 'justify-center' : 'space-x-4'}`}>
                            <item.icon className="w-5 h-5 flex-shrink-0" />
                            {!sidebarCollapsed && (
                              <div className="flex-1 min-w-0">
                                <span className="font-medium text-sm">{item.title}</span>
                                <p className="text-xs text-gray-500 mt-1 line-clamp-1">{item.description}</p>
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

            {/* Specialty Navigation */}
            {user?.specialty && (
              <SidebarGroup>
                {!sidebarCollapsed && (
                  <SidebarGroupLabel className="text-xs font-semibold text-gray-500 uppercase tracking-wider px-8 py-4">
                    Specialty Tools
                  </SidebarGroupLabel>
                )}
                <SidebarGroupContent className="px-4">
                  <SidebarMenu className="space-y-2">
                    {specialtyNavigation.map((item) => (
                      <SidebarMenuItem key={item.title}>
                        <SidebarMenuButton
                          asChild
                          isActive={location.pathname === item.url}
                          className="hover:bg-blue-50 data-[active=true]:bg-blue-100 data-[active=true]:text-blue-700"
                        >
                          <button
                            onClick={() => navigate(item.url)}
                            className="w-full text-left p-4 rounded-lg transition-all duration-200"
                            title={sidebarCollapsed ? item.title : undefined}
                          >
                            <div className={`flex items-center ${sidebarCollapsed ? 'justify-center' : 'space-x-4'}`}>
                              <item.icon className={`w-5 h-5 flex-shrink-0 ${item.color}`} />
                              {!sidebarCollapsed && (
                                <span className="font-medium text-sm">{item.title}</span>
                              )}
                            </div>
                          </button>
                        </SidebarMenuButton>
                      </SidebarMenuItem>
                    ))}
                  </SidebarMenu>
                </SidebarGroupContent>
              </SidebarGroup>
            )}
          </SidebarContent>

          <SidebarFooter className="p-4 border-t border-blue-200">
            <SidebarMenu>
              {secondaryItems.map((item) => (
                <SidebarMenuItem key={item.title}>
                  <SidebarMenuButton
                    asChild
                    isActive={location.pathname === item.url}
                    className="hover:bg-blue-50 data-[active=true]:bg-blue-100 data-[active=true]:text-blue-700"
                  >
                    <button
                      onClick={() => navigate(item.url)}
                      className="w-full text-left p-3 rounded-lg transition-all duration-200"
                      title={sidebarCollapsed ? item.title : undefined}
                    >
                      <div className={`flex items-center ${sidebarCollapsed ? 'justify-center' : 'space-x-3'}`}>
                        <item.icon className="w-5 h-5" />
                        {!sidebarCollapsed && (
                          <span className="font-medium">{item.title}</span>
                        )}
                      </div>
                    </button>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>

            {/* Sidebar Toggle Button */}
            <div className="mt-4 pt-4 border-t border-blue-200">
              <button
                onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
                className="w-full p-3 rounded-lg hover:bg-blue-50 transition-colors text-gray-600 hover:text-blue-700"
                aria-label={sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
              >
                <div className={`flex items-center ${sidebarCollapsed ? 'justify-center' : 'space-x-3'}`}>
                  {sidebarCollapsed ? <Menu className="w-5 h-5" /> : <X className="w-5 h-5" />}
                  {!sidebarCollapsed && (
                    <span className="text-sm font-medium">
                      {sidebarCollapsed ? "Expand" : "Collapse"}
                    </span>
                  )}
                </div>
              </button>
            </div>
          </SidebarFooter>
        </Sidebar>

        {/* Mobile Sidebar Overlay */}
        {!sidebarCollapsed && (
          <div className="md:hidden fixed inset-0 z-40 bg-black/50 backdrop-blur-sm">
            <div className="fixed left-0 top-0 bottom-0 w-80 bg-white shadow-2xl">
              <div className="p-6 border-b border-gray-200">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <div className="w-10 h-10 bg-blue-600 rounded-xl flex items-center justify-center">
                      <Stethoscope className="w-6 h-6 text-white" />
                    </div>
                    <div>
                      <h2 className="text-xl font-bold text-gray-900">DoctAI</h2>
                      <p className="text-sm text-blue-600">Provider Portal</p>
                    </div>
                  </div>
                  <button
                    onClick={() => setSidebarCollapsed(true)}
                    className="p-2 rounded-lg hover:bg-gray-100"
                  >
                    <X className="w-5 h-5" />
                  </button>
                </div>
              </div>

              <div className="flex-1 overflow-y-auto">
                <div className="p-4">
                  <div className="space-y-1">
                    {mainNavigation.map((item) => (
                      <button
                        key={item.title}
                        onClick={() => {
                          navigate(item.url);
                          setSidebarCollapsed(true);
                        }}
                        className={`w-full flex items-center space-x-3 p-3 rounded-lg transition-colors text-left ${
                          location.pathname === item.url
                            ? 'bg-blue-100 text-blue-700 border-r-2 border-blue-600'
                            : 'hover:bg-gray-50'
                        }`}
                      >
                        <item.icon className="w-5 h-5 flex-shrink-0" />
                        <div>
                          <span className="font-medium text-sm">{item.title}</span>
                          <p className="text-xs text-gray-500 mt-1">{item.description}</p>
                        </div>
                      </button>
                    ))}
                  </div>

                  {user?.specialty && (
                    <div className="mt-6">
                      <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider px-3 py-2">
                        Specialty Tools
                      </h3>
                      <div className="space-y-1">
                        {specialtyNavigation.map((item) => (
                          <button
                            key={item.title}
                            onClick={() => {
                              navigate(item.url);
                              setSidebarCollapsed(true);
                            }}
                            className={`w-full flex items-center space-x-3 p-3 rounded-lg transition-colors text-left ${
                              location.pathname === item.url
                                ? 'bg-blue-100 text-blue-700'
                                : 'hover:bg-gray-50'
                            }`}
                          >
                            <item.icon className={`w-5 h-5 flex-shrink-0 ${item.color}`} />
                            <span className="font-medium text-sm">{item.title}</span>
                          </button>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}

        <div className="flex-1 flex flex-col">
          {/* Provider Header */}
          <header className="bg-white/95 backdrop-blur-sm border-b border-blue-200 px-6 py-4 shadow-sm md:mt-0 mt-16">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-8">
                <div className="hidden md:block">
                  <h1 className="text-2xl font-bold text-gray-900">DoctAI</h1>
                  <p className="text-sm text-blue-600 font-medium">Provider Portal</p>
                </div>

                {/* Search Input */}
                <div className="relative">
                  <input
                    type="text"
                    placeholder="Search patients, records..."
                    className="w-80 pl-4 pr-10 py-2 bg-gray-50 border border-gray-200 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    aria-label="Search patients and records"
                  />
                  <div className="absolute inset-y-0 right-0 flex items-center pr-3 pointer-events-none">
                    <svg className="w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                    </svg>
                  </div>
                </div>
              </div>

              <div className="flex items-center space-x-4">
                {/* Quick Action Buttons - Hidden on mobile */}
                <div className="hidden lg:flex items-center space-x-3">
                  <Button 
                    size="sm" 
                    className="bg-blue-600 hover:bg-blue-700 focus:ring-2 focus:ring-blue-500 focus:ring-offset-1" 
                    aria-label="Create new patient"
                    data-testid="new-patient-button"
                    title="Add a new patient to your roster"
                    onClick={() => {
                      // This will trigger NewPatientModal in the consuming component
                      window.dispatchEvent(new CustomEvent('openNewPatientModal'));
                    }}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault();
                        window.dispatchEvent(new CustomEvent('openNewPatientModal'));
                      }
                    }}
                    tabIndex={0}
                  >
                    <User className="w-4 h-4 mr-2" />
                    New Patient
                  </Button>
                  <Button size="sm" variant="outline" className="border-blue-200 text-blue-700 hover:bg-blue-50" aria-label="Open schedule">
                    <Calendar className="w-4 h-4 mr-2" />
                    Schedule
                  </Button>
                  <Button size="sm" variant="outline" className="border-blue-200 text-blue-700 hover:bg-blue-50" aria-label="Open messages">
                    <MessageSquare className="w-4 h-4 mr-2" />
                    Messages
                  </Button>
                </div>

                {/* Status & Notification Indicators */}
                <div className="flex items-center space-x-3">
                  {/* Online/Offline Status Pill */}
                  <ProviderStatusIndicator />
                  
                  {/* Notification Indicator */}
                  <NotificationIndicator />

                  {/* Emergency Alert */}
                  <Button
                    variant="ghost"
                    size="sm"
                    className="p-2 hover:bg-orange-50 rounded-xl relative"
                    aria-label="View alerts"
                  >
                    <AlertTriangle className="w-5 h-5 text-orange-500" />
                    <Badge className="absolute -top-1 -right-1 h-4 w-4 text-xs bg-orange-500 hover:bg-orange-500 rounded-full p-0 flex items-center justify-center">
                      1
                    </Badge>
                  </Button>
                </div>

                {/* User Avatar */}
                <div className="flex items-center space-x-3 px-3 py-2 bg-blue-50 border border-blue-200 rounded-xl">
                  <Avatar className="h-8 w-8">
                    <AvatarImage src={user?.avatar} alt={user?.name} />
                    <AvatarFallback className="bg-blue-500 text-white text-sm font-medium">
                      DS
                    </AvatarFallback>
                  </Avatar>
                  <span className="text-sm font-medium text-gray-700 hidden sm:block">Dr. Sarah</span>
                </div>
              </div>
            </div>
          </header>

          {/* Main Content */}
          <main 
            id="main-content"
            className="flex-1 p-4 sm:p-6 overflow-auto bg-gradient-to-br from-blue-50/30 to-indigo-100/30"
            role="main"
            aria-label="Provider portal main content"
          >
            <div className="max-w-7xl mx-auto space-y-4 lg:space-y-6">
              {children}
            </div>
          </main>
        </div>

        {/* Mobile Utility Bar - Sticky Bottom */}
        <div className="md:hidden fixed bottom-0 left-0 right-0 bg-white/95 backdrop-blur-sm border-t border-blue-200 px-4 py-3 z-40">
          <div className="flex items-center justify-around space-x-2">
            <Button 
              size="sm" 
              className="flex-1 bg-blue-600 hover:bg-blue-700 text-white text-xs"
              onClick={() => window.dispatchEvent(new CustomEvent('openNewPatientModal'))}
              aria-label="Create new patient"
            >
              <User className="w-4 h-4 mr-1" />
              New Patient
            </Button>
            <Button 
              size="sm" 
              variant="outline" 
              className="flex-1 border-blue-200 text-blue-700 hover:bg-blue-50 text-xs"
              onClick={() => navigate('/provider/schedule')}
              aria-label="Open schedule"
            >
              <Calendar className="w-4 h-4 mr-1" />
              Schedule
            </Button>
            <Button 
              size="sm" 
              variant="outline" 
              className="flex-1 border-blue-200 text-blue-700 hover:bg-blue-50 text-xs"
              onClick={() => navigate('/provider/messages')}
              aria-label="Open messages"
            >
              <MessageSquare className="w-4 h-4 mr-1" />
              Messages
            </Button>
          </div>
        </div>
      </div>
    </SidebarProvider>
  );
}
