
import React from 'react';
import { SidebarTrigger } from '@/components/ui/sidebar';
import { Bell, User, Menu } from 'lucide-react';
import ResponsiveSearchBar from './ResponsiveSearchBar';

export function AppHeader() {
  const handleSearchSelect = (result: any) => {
    console.log('Selected:', result);
    // Handle navigation or other actions based on the selected result
  };

  return (
    <header className="bg-white border-b border-gray-200 px-4 md:px-6 py-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <SidebarTrigger className="md:hidden">
            <Menu className="w-5 h-5" />
          </SidebarTrigger>
          <div className="hidden md:block">
            <h1 className="text-xl font-semibold text-gray-900">DoctAI Dashboard</h1>
            <p className="text-sm text-gray-500">Your AI-powered health companion</p>
          </div>
          <div className="md:hidden">
            <h1 className="text-lg font-semibold text-gray-900">DoctAI</h1>
          </div>
        </div>

        {/* Responsive Search Bar */}
        <div className="hidden md:block flex-1 max-w-lg mx-8">
          <ResponsiveSearchBar
            onSelect={handleSearchSelect}
            placeholder="Search doctors, medications, records..."
          />
        </div>

        <div className="flex items-center space-x-3">
          {/* Mobile Search - Show search icon that could trigger a modal */}
          <div className="md:hidden">
            <ResponsiveSearchBar
              onSelect={handleSearchSelect}
              placeholder="Search..."
            />
          </div>
          
          <button className="p-2 text-gray-400 hover:text-gray-600 rounded-full hover:bg-gray-100 relative">
            <Bell className="w-5 h-5" />
            <span className="absolute top-1 right-1 w-2 h-2 bg-red-500 rounded-full"></span>
          </button>
          <button className="p-2 text-gray-400 hover:text-gray-600 rounded-full hover:bg-gray-100">
            <User className="w-5 h-5" />
          </button>
        </div>
      </div>
    </header>
  );
}
