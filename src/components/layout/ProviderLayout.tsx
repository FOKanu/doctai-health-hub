import React from 'react';
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
  Bone
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
import { UserProfileDropdown } from '@/components/UserProfileDropdown';
import { RoleBasedMobileNavigation } from './RoleBasedMobileNavigation';

export function ProviderLayout({ children }: { children: React.ReactNode }) {
  const navigate = useNavigate();
  const location = useLocation();
  const { user } = useAuth();

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
      title: "Clinical Workflow",
      url: "/provider/clinical",
      icon: Clipboard,
      description: "Labs, prescriptions, vitals"
    },
    {
      title: "AI Diagnostic Support",
      url: "/provider/ai-support",
      icon: Brain,
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
      url: "/provider/cardiology",
      icon: Heart,
      color: "text-red-600"
    },
    {
      title: "Neurology",
      url: "/provider/neurology",
      icon: Brain,
      color: "text-purple-600"
    },
    {
      title: "Ophthalmology",
      url: "/provider/ophthalmology",
      icon: Eye,
      color: "text-blue-600"
    },
    {
      title: "Orthopedics",
      url: "/provider/orthopedics",
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
      <div className="min-h-screen flex w-full bg-gradient-to-br from-blue-50 to-indigo-100">
        <Sidebar className="hidden md:flex border-r border-blue-200 bg-white/95 backdrop-blur-sm w-80">
          <SidebarHeader className="p-8 border-b border-blue-200">
            <div className="flex items-center space-x-4">
              <div className="w-12 h-12 bg-blue-600 rounded-xl flex items-center justify-center shadow-lg">
                <Stethoscope className="w-7 h-7 text-white" />
              </div>
              <div>
                <h2 className="text-2xl font-bold text-gray-900">DoctAI</h2>
                <p className="text-sm text-blue-600 font-medium">Provider Portal</p>
              </div>
            </div>

            {/* Provider Info */}
            {user && (
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
              <SidebarGroupLabel className="text-xs font-semibold text-gray-500 uppercase tracking-wider px-8 py-4">
                Main Menu
              </SidebarGroupLabel>
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
                          className="w-full text-left p-4 rounded-lg transition-all duration-200"
                        >
                          <div className="flex items-center space-x-4">
                            <item.icon className="w-5 h-5 flex-shrink-0" />
                            <div className="flex-1 min-w-0">
                              <span className="font-medium text-sm">{item.title}</span>
                              <p className="text-xs text-gray-500 mt-1 line-clamp-1">{item.description}</p>
                            </div>
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
                <SidebarGroupLabel className="text-xs font-semibold text-gray-500 uppercase tracking-wider px-8 py-4">
                  Specialty Tools
                </SidebarGroupLabel>
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
                          >
                            <div className="flex items-center space-x-4">
                              <item.icon className={`w-5 h-5 flex-shrink-0 ${item.color}`} />
                              <span className="font-medium text-sm">{item.title}</span>
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
          {/* Provider Header */}
          <header className="bg-white/95 backdrop-blur-sm border-b border-blue-200 px-6 py-4 shadow-sm">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-8">
                <div>
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
                {/* Quick Action Buttons */}
                <Button size="sm" className="bg-blue-600 hover:bg-blue-700" aria-label="Create new patient">
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

                {/* Status Pill */}
                <div className="flex items-center space-x-2 px-3 py-1 bg-green-50 border border-green-200 rounded-full">
                  <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                  <span className="text-sm text-green-700 font-medium">Online</span>
                </div>

                {/* Notifications */}
                <Button variant="ghost" size="sm" className="relative" aria-label="Notifications">
                  <AlertTriangle className="w-5 h-5 text-orange-500" />
                  <Badge className="absolute -top-1 -right-1 h-5 w-5 rounded-full bg-red-500 text-xs">
                    3
                  </Badge>
                </Button>

                {/* User Avatar */}
                <div className="flex items-center space-x-3 px-3 py-2 bg-blue-50 border border-blue-200 rounded-xl">
                  <Avatar className="h-8 w-8">
                    <AvatarImage src={user?.avatar} alt={user?.name} />
                    <AvatarFallback className="bg-blue-500 text-white text-sm font-medium">
                      DS
                    </AvatarFallback>
                  </Avatar>
                  <span className="text-sm font-medium text-gray-700">Dr. Sarah</span>
                </div>
              </div>
            </div>
          </header>

          {/* Main Content */}
          <main className="flex-1 p-6 overflow-auto bg-gradient-to-br from-blue-50/30 to-indigo-100/30">
            <div className="max-w-7xl mx-auto space-y-6">
              {children}
            </div>
          </main>
        </div>
        <RoleBasedMobileNavigation role="provider" />
      </div>
    </SidebarProvider>
  );
}
