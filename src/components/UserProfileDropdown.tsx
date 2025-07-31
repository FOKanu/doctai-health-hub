
import React from 'react';
import { useNavigate } from 'react-router-dom';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import { Button } from '@/components/ui/button';
import { User, Settings, Bell, LogOut } from 'lucide-react';
import { useAuth } from '@/contexts/AuthContext';

interface UserProfileDropdownProps {
  user?: {
    name: string;
    email?: string;
    avatar?: string;
  };
  onLogout?: () => void;
}

export function UserProfileDropdown({ user, onLogout }: UserProfileDropdownProps) {
  const navigate = useNavigate();
  const { logout: authLogout, user: authUser } = useAuth();

  const handleViewProfile = () => {
    navigate('/patient/profile');
  };

  const handleAccountSettings = () => {
    navigate('/patient/settings');
  };

  const handleNotificationPreferences = () => {
    navigate('/patient/settings');
  };

  const handleLogout = () => {
    // Use the auth context logout method
    authLogout();

    // Call the provided logout handler if any
    if (onLogout) {
      onLogout();
    }

    // Navigate to login page
    navigate('/login');
  };

  const getInitials = (name: string) => {
    return name
      .split(' ')
      .map(n => n[0])
      .join('')
      .toUpperCase()
      .slice(0, 2);
  };

  // Use auth context user if available, otherwise fall back to props
  const currentUser = authUser || user || {
    name: 'User',
    email: 'user@doctai.com',
    avatar: undefined
  };

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          variant="ghost"
          className="relative h-8 w-8 rounded-full hover:bg-gray-100 focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
        >
          <Avatar className="h-8 w-8">
            <AvatarImage src={currentUser.avatar} alt={currentUser.name} />
            <AvatarFallback className="bg-blue-500 text-white text-sm">
              {getInitials(currentUser.name)}
            </AvatarFallback>
          </Avatar>
        </Button>
      </DropdownMenuTrigger>

      <DropdownMenuContent
        className="w-64 bg-white shadow-lg border border-gray-200 rounded-lg p-0"
        align="end"
        sideOffset={8}
      >
        {/* User Info Section */}
        <DropdownMenuLabel className="px-4 py-3 border-b border-gray-100">
          <div className="flex items-center space-x-3">
            <Avatar className="h-10 w-10">
              <AvatarImage src={currentUser.avatar} alt={currentUser.name} />
              <AvatarFallback className="bg-blue-500 text-white">
                {getInitials(currentUser.name)}
              </AvatarFallback>
            </Avatar>
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium text-gray-900 truncate">
                {currentUser.name}
              </p>
              {currentUser.email && (
                <p className="text-xs text-gray-500 truncate">
                  {currentUser.email}
                </p>
              )}
            </div>
          </div>
        </DropdownMenuLabel>

        {/* Menu Items */}
        <div className="py-1">
          <DropdownMenuItem
            onClick={handleViewProfile}
            className="px-4 py-2 text-sm text-gray-700 hover:bg-gray-50 cursor-pointer flex items-center space-x-2"
          >
            <User className="h-4 w-4" />
            <span>View Profile</span>
          </DropdownMenuItem>

          <DropdownMenuItem
            onClick={handleAccountSettings}
            className="px-4 py-2 text-sm text-gray-700 hover:bg-gray-50 cursor-pointer flex items-center space-x-2"
          >
            <Settings className="h-4 w-4" />
            <span>Account Settings</span>
          </DropdownMenuItem>

          <DropdownMenuItem
            onClick={handleNotificationPreferences}
            className="px-4 py-2 text-sm text-gray-700 hover:bg-gray-50 cursor-pointer flex items-center space-x-2"
          >
            <Bell className="h-4 w-4" />
            <span>Notification Preferences</span>
          </DropdownMenuItem>
        </div>

        <DropdownMenuSeparator className="bg-gray-100" />

        {/* Logout Section */}
        <div className="py-1">
          <DropdownMenuItem
            onClick={handleLogout}
            className="px-4 py-2 text-sm text-red-600 hover:bg-red-50 cursor-pointer flex items-center space-x-2"
          >
            <LogOut className="h-4 w-4" />
            <span>Log out</span>
          </DropdownMenuItem>
        </div>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
