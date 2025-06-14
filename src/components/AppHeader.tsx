
import React from 'react';
import { SidebarTrigger } from '@/components/ui/sidebar';
import { Bell, Search, User } from 'lucide-react';

export function AppHeader() {
  return (
    <header className="bg-white border-b border-gray-200 px-4 md:px-6 py-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <SidebarTrigger className="md:hidden" />
          <div className="hidden md:block">
            <h1 className="text-xl font-semibold text-gray-900">DoctAI Dashboard</h1>
            <p className="text-sm text-gray-500">Your AI-powered health companion</p>
          </div>
          <div className="md:hidden">
            <h1 className="text-lg font-semibold text-gray-900">DoctAI</h1>
          </div>
        </div>

        <div className="flex items-center space-x-3">
          <button className="p-2 text-gray-400 hover:text-gray-600 rounded-full hover:bg-gray-100">
            <Search className="w-5 h-5" />
          </button>
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
