
import React from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import {
  Camera,
  Upload,
  User,
  Calendar,
  Pill,
  FileText,
  Home,
  History,
  Activity,
  Settings
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

export function AppSidebar() {
  const navigate = useNavigate();
  const location = useLocation();

  const coreActions = [
    {
      title: "Skin Scan",
      url: "/scan",
      icon: Camera,
      description: "Camera scan for skin lesions"
    },
    {
      title: "Upload Medical Image",
      url: "/upload",
      icon: Upload,
      description: "CT, MRI, EEG analysis"
    }
  ];

  const mainNavigation = [
    {
      title: "Dashboard",
      url: "/",
      icon: Home
    },
    {
      title: "Specialists",
      url: "/specialists",
      icon: User
    },
    {
      title: "Appointments",
      url: "/appointments",
      icon: Calendar
    },
    {
      title: "Medications",
      url: "/medications",
      icon: Pill
    },
    {
      title: "History",
      url: "/history",
      icon: History
    },
    {
      title: "Treatment Plans",
      url: "/treatments",
      icon: FileText
    }
  ];

  const secondaryItems = [
    {
      title: "Profile",
      url: "/profile",
      icon: Settings
    }
  ];

  return (
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
            Core Actions
          </SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {coreActions.map((item) => (
                <SidebarMenuItem key={item.title}>
                  <SidebarMenuButton
                    asChild
                    isActive={location.pathname === item.url}
                    className="h-auto p-4 hover:bg-blue-50 data-[active=true]:bg-blue-100 data-[active=true]:text-blue-700"
                  >
                    <button onClick={() => navigate(item.url)} className="w-full text-left">
                      <div className="flex items-start space-x-3">
                        <item.icon className="w-5 h-5 mt-0.5 text-blue-600" />
                        <div>
                          <div className="font-medium text-gray-900">{item.title}</div>
                          <div className="text-xs text-gray-500 mt-1">{item.description}</div>
                        </div>
                      </div>
                    </button>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>

        <SidebarGroup>
          <SidebarGroupLabel className="text-xs font-semibold text-gray-500 uppercase tracking-wider">
            Navigation
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
  );
}
