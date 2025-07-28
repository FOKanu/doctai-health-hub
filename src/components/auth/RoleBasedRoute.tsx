import React from 'react';
import { Navigate, useLocation } from 'react-router-dom';
import { useAuth } from '@/contexts/AuthContext';
import { UserRole } from '@/contexts/AuthContext';

interface RoleBasedRouteProps {
  children: React.ReactNode;
  allowedRoles: UserRole[];
  fallbackPath?: string;
}

export function RoleBasedRoute({
  children,
  allowedRoles,
  fallbackPath = '/login'
}: RoleBasedRouteProps) {
  const { user, isAuthenticated, isLoading } = useAuth();
  const location = useLocation();

  // Show loading state while checking authentication
  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading...</p>
        </div>
      </div>
    );
  }

  // Redirect to login if not authenticated
  if (!isAuthenticated) {
    return <Navigate to={fallbackPath} state={{ from: location }} replace />;
  }

  // Check if user has required role
  if (!user || !allowedRoles.includes(user.role)) {
    // Redirect to appropriate dashboard based on user role
    const roleRedirects: Record<UserRole, string> = {
      patient: '/',
      provider: '/provider/dashboard',
      engineer: '/engineer/dashboard',
      admin: '/admin/dashboard'
    };

    const redirectPath = roleRedirects[user?.role || 'patient'];
    return <Navigate to={redirectPath} replace />;
  }

  return <>{children}</>;
}

// Specific route components for each role
export function PatientRoute({ children }: { children: React.ReactNode }) {
  return (
    <RoleBasedRoute allowedRoles={['patient']}>
      {children}
    </RoleBasedRoute>
  );
}

export function ProviderRoute({ children }: { children: React.ReactNode }) {
  return (
    <RoleBasedRoute allowedRoles={['provider']}>
      {children}
    </RoleBasedRoute>
  );
}

export function EngineerRoute({ children }: { children: React.ReactNode }) {
  return (
    <RoleBasedRoute allowedRoles={['engineer']}>
      {children}
    </RoleBasedRoute>
  );
}

export function AdminRoute({ children }: { children: React.ReactNode }) {
  return (
    <RoleBasedRoute allowedRoles={['admin']}>
      {children}
    </RoleBasedRoute>
  );
}

// Route for multiple roles
export function MultiRoleRoute({
  children,
  roles
}: {
  children: React.ReactNode;
  roles: UserRole[];
}) {
  return (
    <RoleBasedRoute allowedRoles={roles}>
      {children}
    </RoleBasedRoute>
  );
}
