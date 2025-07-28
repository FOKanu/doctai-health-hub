import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';

// User role types
export type UserRole = 'patient' | 'provider' | 'engineer' | 'admin';

// User interface
export interface User {
  id: string;
  email: string;
  name: string;
  role: UserRole;
  avatar?: string;
  specialty?: string; // For providers
  department?: string; // For engineers
  permissions: string[];
  isActive: boolean;
  lastLogin: Date;
  mfaEnabled: boolean;
}

// Authentication context interface
interface AuthContextType {
  user: User | null;
  isLoading: boolean;
  isAuthenticated: boolean;
  login: (email: string, password: string) => Promise<boolean>;
  logout: () => void;
  hasPermission: (permission: string) => boolean;
  hasRole: (role: UserRole) => boolean;
  updateUser: (updates: Partial<User>) => void;
}

// Create the context
const AuthContext = createContext<AuthContextType | undefined>(undefined);

// Mock user data for development
const mockUsers: User[] = [
  {
    id: '1',
    email: 'patient@doctai.com',
    name: 'John Doe',
    role: 'patient',
    permissions: ['read:own-record', 'write:own-appointment', 'read:own-appointment'],
    isActive: true,
    lastLogin: new Date(),
    mfaEnabled: false
  },
  {
    id: '2',
    email: 'doctor@doctai.com',
    name: 'Dr. Sarah Johnson',
    role: 'provider',
    specialty: 'Cardiology',
    permissions: [
      'read:patient',
      'write:medical-record',
      'read:appointment',
      'write:appointment',
      'read:lab-results',
      'write:prescription'
    ],
    isActive: true,
    lastLogin: new Date(),
    mfaEnabled: true
  },
  {
    id: '3',
    email: 'engineer@doctai.com',
    name: 'Alex Chen',
    role: 'engineer',
    department: 'Backend Development',
    permissions: [
      'read:system-logs',
      'write:system-config',
      'read:security-logs',
      'write:deployment',
      'read:performance-metrics'
    ],
    isActive: true,
    lastLogin: new Date(),
    mfaEnabled: true
  },
  {
    id: '4',
    email: 'admin@doctai.com',
    name: 'Admin User',
    role: 'admin',
    permissions: ['*:*'],
    isActive: true,
    lastLogin: new Date(),
    mfaEnabled: true
  }
];

// Role-based permissions mapping
const rolePermissions: Record<UserRole, string[]> = {
  patient: [
    'read:own-record',
    'write:own-appointment',
    'read:own-appointment',
    'read:own-medications',
    'read:own-treatments'
  ],
  provider: [
    'read:patient',
    'write:medical-record',
    'read:appointment',
    'write:appointment',
    'read:lab-results',
    'write:prescription',
    'read:patient-history',
    'write:treatment-plan',
    'read:compliance-reports'
  ],
  engineer: [
    'read:system-logs',
    'write:system-config',
    'read:security-logs',
    'write:deployment',
    'read:performance-metrics',
    'write:database-schema',
    'read:error-logs',
    'write:api-config'
  ],
  admin: ['*:*']
};

// Auth provider component
interface AuthProviderProps {
  children: ReactNode;
}

export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // Check for existing session on mount
  useEffect(() => {
    const checkAuth = async () => {
      try {
        const storedUser = localStorage.getItem('doctai_user');
        if (storedUser) {
          const userData = JSON.parse(storedUser);
          setUser(userData);
        }
      } catch (error) {
        console.error('Auth check failed:', error);
      } finally {
        setIsLoading(false);
      }
    };

    checkAuth();
  }, []);

  // Login function
  const login = async (email: string, password: string): Promise<boolean> => {
    setIsLoading(true);

    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1000));

      // Find user by email
      const foundUser = mockUsers.find(u => u.email === email);

      if (foundUser && password === 'password') { // Simple password check for demo
        setUser(foundUser);
        localStorage.setItem('doctai_user', JSON.stringify(foundUser));
        return true;
      } else {
        throw new Error('Invalid credentials');
      }
    } catch (error) {
      console.error('Login failed:', error);
      return false;
    } finally {
      setIsLoading(false);
    }
  };

  // Logout function
  const logout = () => {
    setUser(null);
    localStorage.removeItem('doctai_user');
  };

  // Check if user has specific permission
  const hasPermission = (permission: string): boolean => {
    if (!user) return false;

    // Admin has all permissions
    if (user.role === 'admin') return true;

    // Check user's specific permissions
    return user.permissions.includes(permission) ||
           user.permissions.includes('*:*');
  };

  // Check if user has specific role
  const hasRole = (role: UserRole): boolean => {
    return user?.role === role;
  };

  // Update user data
  const updateUser = (updates: Partial<User>) => {
    if (user) {
      const updatedUser = { ...user, ...updates };
      setUser(updatedUser);
      localStorage.setItem('doctai_user', JSON.stringify(updatedUser));
    }
  };

  const value: AuthContextType = {
    user,
    isLoading,
    isAuthenticated: !!user,
    login,
    logout,
    hasPermission,
    hasRole,
    updateUser
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};

// Custom hook to use auth context
export const useAuth = (): AuthContextType => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

// Hook to get role-specific permissions
export const useRolePermissions = () => {
  const { user } = useAuth();

  if (!user) return [];

  return rolePermissions[user.role] || [];
};

// Hook to check if user can access a specific feature
export const useFeatureAccess = (feature: string): boolean => {
  const { hasPermission } = useAuth();

  const featurePermissions: Record<string, string> = {
    'patient-management': 'read:patient',
    'medical-records': 'write:medical-record',
    'appointments': 'write:appointment',
    'prescriptions': 'write:prescription',
    'system-logs': 'read:system-logs',
    'deployment': 'write:deployment',
    'security-logs': 'read:security-logs',
    'compliance-reports': 'read:compliance-reports'
  };

  const requiredPermission = featurePermissions[feature];
  return requiredPermission ? hasPermission(requiredPermission) : false;
};
