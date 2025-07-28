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
} from '@/components/ui/sidebar';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import { useAuth } from '@/contexts/AuthContext';
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
    <div className="min-h-screen flex w-full bg-gradient-to-br from-blue-50 to-indigo-100">
      <Sidebar className="hidden md:flex border-r border-blue-200 bg-white/95 backdrop-blur-sm">
        <SidebarHeader className="p-6 border-b border-blue-200">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-blue-600 rounded-xl flex items-center justify-center shadow-lg">
              <Stethoscope className="w-6 h-6 text-white" />
            </div>
            <div>
              <h2 className="text-xl font-bold text-gray-900">DoctAI</h2>
              <p className="text-sm text-blue-600 font-medium">Provider Portal</p>
            </div>
          </div>

          {/* Provider Info */}
          {user && (
            <div className="mt-4 p-3 bg-blue-50 rounded-lg">
              <div className="flex items-center space-x-3">
                <Avatar className="h-8 w-8">
                  <AvatarImage src={user.avatar} alt={user.name} />
                  <AvatarFallback className="bg-blue-500 text-white text-sm">
                    {getInitials(user.name)}
                  </AvatarFallback>
                </Avatar>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-semibold text-gray-900 truncate">
                    {user.name}
                  </p>
                  {user.specialty && (
                    <p className="text-xs text-blue-600 font-medium">
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
            <SidebarGroupLabel className="text-xs font-semibold text-gray-500 uppercase tracking-wider px-6 py-2">
              Main Menu
            </SidebarGroupLabel>
            <SidebarGroupContent>
              <SidebarMenu>
                {mainNavigation.map((item) => (
                  <SidebarMenuItem key={item.title}>
                    <SidebarMenuButton
                      asChild
                      isActive={location.pathname === item.url}
                      className="hover:bg-blue-50 data-[active=true]:bg-blue-100 data-[active=true]:text-blue-700 data-[active=true]:border-r-2 data-[active=true]:border-blue-600"
                    >
                      <button
                        onClick={() => navigate(item.url)}
                        className="w-full text-left p-3 rounded-lg transition-all duration-200"
                      >
                        <div className="flex items-center space-x-3">
                          <item.icon className="w-5 h-5" />
                          <div className="flex-1">
                            <span className="font-medium">{item.title}</span>
                            <p className="text-xs text-gray-500 mt-1">{item.description}</p>
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
              <SidebarGroupLabel className="text-xs font-semibold text-gray-500 uppercase tracking-wider px-6 py-2">
                Specialty Tools
              </SidebarGroupLabel>
              <SidebarGroupContent>
                <SidebarMenu>
                  {specialtyNavigation.map((item) => (
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
        <header className="bg-white border-b border-blue-200 px-6 py-4 shadow-sm">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-6">
              <div>
                <h1 className="text-2xl font-bold text-gray-900">Provider Dashboard</h1>
                <p className="text-sm text-blue-600">
                  {user?.specialty ? `${user.specialty} • ` : ''}AI-Powered Medical Care
                </p>
              </div>

              {/* Quick Action Buttons */}
              <div className="flex items-center space-x-3">
                <Button size="sm" className="bg-blue-600 hover:bg-blue-700">
                  <User className="w-4 h-4 mr-2" />
                  New Patient
                </Button>
                <Button size="sm" variant="outline" className="border-blue-200 text-blue-700 hover:bg-blue-50">
                  <Calendar className="w-4 h-4 mr-2" />
                  Schedule
                </Button>
                <Button size="sm" variant="outline" className="border-blue-200 text-blue-700 hover:bg-blue-50">
                  <MessageSquare className="w-4 h-4 mr-2" />
                  Messages
                </Button>
              </div>
            </div>

            <div className="flex items-center space-x-4">
              {/* Status Indicators */}
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                <span className="text-sm text-gray-600">Online</span>
              </div>

              {/* Notifications */}
              <Button variant="ghost" size="sm" className="relative">
                <AlertTriangle className="w-5 h-5 text-orange-500" />
                <Badge className="absolute -top-1 -right-1 h-5 w-5 rounded-full bg-red-500 text-xs">
                  3
                </Badge>
              </Button>
            </div>
          </div>
        </header>

        {/* Main Content */}
        <main className="flex-1 p-6 overflow-auto">
          <div className="max-w-7xl mx-auto">
            {children}
          </div>
        </main>
      </div>
      <RoleBasedMobileNavigation role="provider" />
    </div>
  );
}
