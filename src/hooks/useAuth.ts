
import { useState, useEffect } from 'react';

interface User {
  id: string;
  email?: string;
}

export const useAuth = () => {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Mock user for development
    setUser({ id: 'mock-user-id', email: 'user@example.com' });
    setLoading(false);
  }, []);

  return {
    user,
    loading,
    signIn: async (email: string, password: string) => {
      // Mock sign in
      setUser({ id: 'mock-user-id', email });
      return { error: null };
    },
    signOut: async () => {
      setUser(null);
      return { error: null };
    }
  };
};
