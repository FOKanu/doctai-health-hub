import React from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import {
  User,
  Calendar,
  Pill,
  FileText,
  Home,
  History,
  Activity,
  Settings,
  Mail,
  FolderOpen,
  BarChart3,
  UserCheck,
  Heart,
  Brain,
  Shield,
  MessageSquare
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
import { useAuth } from '@/contexts/AuthContext';
import { RoleBasedMobileNavigation } from './RoleBasedMobileNavigation';

export function ClientLayout({ children }: { children: React.ReactNode }) {
  const navigate = useNavigate();
  const location = useLocation();
  const { user } = useAuth();

  // Client-specific navigation (existing patient interface)
  const mainNavigation = [
    {
      title: "Dashboard",
      url: "/patient/",
      icon: Home
    },
    {
      title: "Analytics",
      url: "/patient/analytics",
      icon: BarChart3
    },
    {
      title: "Postbox",
      url: "/patient/postbox",
      icon: Mail
    },
    {
      title: "Messages",
      url: "/patient/messages",
      icon: MessageSquare
    },
    {
      title: "Medical Records",
      url: "/patient/medical-records",
      icon: FolderOpen
    },
    {
      title: "Find Care",
      url: "/patient/specialists",
      icon: User
    },
    {
      title: "Schedule",
      url: "/patient/appointments",
      icon: Calendar
    },
    {
      title: "Book Appointment",
      url: "/patient/book-appointment",
      icon: Calendar
    },
    {
      title: "Medications",
      url: "/patient/medications",
      icon: Pill
    },
    {
      title: "History",
      url: "/patient/history",
      icon: History
    },
    {
      title: "Treatment Plans",
      url: "/patient/treatments",
      icon: FileText
    }
  ];

  const healthNavigation = [
    {
      title: "Health Overview",
      url: "/patient/health-overview",
      icon: Heart,
      color: "text-red-500"
    },
    {
      title: "Mental Health",
      url: "/patient/mental-health",
      icon: Brain,
      color: "text-purple-500"
    },
    {
      title: "Security & Privacy",
      url: "/patient/security",
      icon: Shield,
      color: "text-green-500"
    }
  ];

  const secondaryItems = [
    {
      title: "Settings",
      url: "/patient/settings",
      icon: Settings
    }
  ];

  return (
    <SidebarProvider>
      <div className="min-h-screen flex w-full bg-white/80 backdrop-blur-sm">
        <Sidebar className="hidden md:flex">
          <SidebarHeader className="p-6">
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center">
                <Activity className="w-5 h-5 text-white" />
              </div>
              <div>
                <h2 className="text-lg font-bold text-gray-900">DoctAI</h2>
                <p className="text-sm text-gray-500">Health Assistant</p>
              </div>
            </div>
          </SidebarHeader>

          <SidebarContent>
            <SidebarGroup>
              <SidebarGroupLabel className="text-xs font-semibold text-gray-500 uppercase tracking-wider">
                Menu
              </SidebarGroupLabel>
              <SidebarGroupContent>
                <SidebarMenu>
                  {mainNavigation.map((item) => (
                    <SidebarMenuItem key={item.title}>
                      <SidebarMenuButton
                        asChild
                        isActive={location.pathname === item.url}
                        className="hover:bg-gray-100 data-[active=true]:bg-blue-100 data-[active=true]:text-blue-700"
                      >
                        <button onClick={() => navigate(item.url)} className="w-full text-left">
                          <item.icon className="w-4 h-4" />
                          <span>{item.title}</span>
                        </button>
                      </SidebarMenuButton>
                    </SidebarMenuItem>
                  ))}
                </SidebarMenu>
              </SidebarGroupContent>
            </SidebarGroup>

            {/* Health-specific navigation */}
            <SidebarGroup>
              <SidebarGroupLabel className="text-xs font-semibold text-gray-500 uppercase tracking-wider">
                Health & Wellness
              </SidebarGroupLabel>
              <SidebarGroupContent>
                <SidebarMenu>
                  {healthNavigation.map((item) => (
                    <SidebarMenuItem key={item.title}>
                      <SidebarMenuButton
                        asChild
                        isActive={location.pathname === item.url}
                        className="hover:bg-gray-100 data-[active=true]:bg-blue-100 data-[active=true]:text-blue-700"
                      >
                        <button onClick={() => navigate(item.url)} className="w-full text-left">
                          <item.icon className={`w-4 h-4 ${item.color}`} />
                          <span>{item.title}</span>
                        </button>
                      </SidebarMenuButton>
                    </SidebarMenuItem>
                  ))}
                </SidebarMenu>
              </SidebarGroupContent>
            </SidebarGroup>
          </SidebarContent>

          <SidebarFooter className="p-4">
            <SidebarMenu>
              {secondaryItems.map((item) => (
                <SidebarMenuItem key={item.title}>
                  <SidebarMenuButton
                    asChild
                    isActive={location.pathname === item.url}
                    className="hover:bg-gray-100 data-[active=true]:bg-blue-100 data-[active=true]:text-blue-700"
                  >
                    <button onClick={() => navigate(item.url)} className="w-full text-left">
                      <item.icon className="w-4 h-4" />
                      <span>{item.title}</span>
                    </button>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarFooter>
        </Sidebar>

        <div className="flex-1 flex flex-col">
          {/* Client Header */}
          <header className="bg-white border-b border-gray-200 px-4 md:px-6 py-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-4">
                <div className="hidden md:block">
                  <h1 className="text-xl font-semibold text-gray-900">DoctAI Dashboard</h1>
                  <p className="text-sm text-gray-500">Your AI-powered health companion</p>
                </div>
                <div className="md:hidden">
                  <h1 className="text-lg font-semibold text-gray-900">DoctAI</h1>
                </div>
              </div>

              <div className="flex items-center space-x-3">
                {/* User status indicator */}
                {user && (
                  <div className="flex items-center space-x-2">
                    <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                    <span className="text-sm text-gray-600">Active</span>
                  </div>
                )}
              </div>
            </div>
          </header>

          {/* Main Content */}
          <main className="flex-1 p-4 md:p-6 lg:p-8 pb-20 md:pb-6">
            <div className="max-w-7xl mx-auto">
              {children}
            </div>
          </main>
        </div>
        <RoleBasedMobileNavigation role="patient" />
      </div>
    </SidebarProvider>
  );
}
