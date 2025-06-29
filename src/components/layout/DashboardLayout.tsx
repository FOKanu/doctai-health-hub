
import React from 'react';
import { SidebarProvider } from '@/components/ui/sidebar';
import { AppSidebar } from '../AppSidebar';
import { AppHeader } from '../AppHeader';
import { MobileNavigation } from '../MobileNavigation';

interface DashboardLayoutProps {
  children: React.ReactNode;
}

export function DashboardLayout({ children }: DashboardLayoutProps) {
  return (
    <SidebarProvider>
      <div className="min-h-screen flex w-full bg-gray-50">
        <AppSidebar />
        <div className="flex-1 flex flex-col">
          <AppHeader />
          <main className="flex-1 p-4 md:p-6 lg:p-8 pb-20 md:pb-6">
            <div className="max-w-7xl mx-auto">
              {children}
            </div>
          </main>
        </div>
        <MobileNavigation />
      </div>
    </SidebarProvider>
  );
}
