
import React, { useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Eye, EyeOff, Activity, Stethoscope, Terminal, User } from 'lucide-react';
import { useAuth } from '@/contexts/AuthContext';

const LoginScreen = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const navigate = useNavigate();
  const location = useLocation();
  const { login } = useAuth();

  // Demo users for testing
  const demoUsers = [
    {
      email: 'patient@doctai.com',
      password: 'password',
      role: 'patient',
      name: 'John Doe',
      description: 'Patient Portal'
    },
    {
      email: 'doctor@doctai.com',
      password: 'password',
      role: 'provider',
      name: 'Dr. Sarah Johnson',
      description: 'Healthcare Provider'
    },
    {
      email: 'engineer@doctai.com',
      password: 'password',
      role: 'engineer',
      name: 'Alex Chen',
      description: 'System Engineer'
    },
    {
      email: 'admin@doctai.com',
      password: 'password',
      role: 'admin',
      name: 'Admin User',
      description: 'System Administrator'
    }
  ];

  const validateForm = () => {
    if (!email || !password) {
      setError('Please fill in all fields');
      return false;
    }
    if (password.length < 6) {
      setError('Password must be at least 6 characters');
      return false;
    }
    return true;
  };

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!validateForm()) {
      return;
    }

    setIsLoading(true);
    setError('');

    try {
      const success = await login(email, password);

      if (success) {
        // Redirect based on user role
        const user = demoUsers.find(u => u.email === email);
        if (user) {
          switch (user.role) {
            case 'patient':
              navigate('/');
              break;
            case 'provider':
              navigate('/provider/dashboard');
              break;
            case 'engineer':
              navigate('/engineer/dashboard');
              break;
            case 'admin':
              navigate('/admin/dashboard');
              break;
            default:
              navigate('/');
          }
        } else {
          navigate('/');
        }
      } else {
        setError('Invalid credentials. Please try again.');
      }
    } catch (error) {
      setError('Login failed. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleDemoLogin = (demoUser: typeof demoUsers[0]) => {
    setEmail(demoUser.email);
    setPassword(demoUser.password);
  };

  const getRoleIcon = (role: string) => {
    switch (role) {
      case 'patient':
        return <User className="w-4 h-4" />;
      case 'provider':
        return <Stethoscope className="w-4 h-4" />;
      case 'engineer':
        return <Terminal className="w-4 h-4" />;
      case 'admin':
        return <Activity className="w-4 h-4" />;
      default:
        return <User className="w-4 h-4" />;
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center p-4">
      <div className="w-full max-w-4xl grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Login Form */}
        <Card className="w-full max-w-md shadow-xl">
          <CardHeader className="space-y-1 text-center">
            <div className="mx-auto w-12 h-12 bg-blue-600 rounded-xl flex items-center justify-center mb-4">
              <Activity className="w-6 h-6 text-white" />
            </div>
            <CardTitle className="text-2xl font-bold text-gray-900">
              Welcome to DoctAI
            </CardTitle>
            <CardDescription className="text-gray-600">
              Sign in to access your role-specific dashboard
            </CardDescription>
          </CardHeader>

          <CardContent>
            <form onSubmit={handleLogin} className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="email">Email</Label>
                <Input
                  id="email"
                  type="email"
                  placeholder="Enter your email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  required
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="password">Password</Label>
                <div className="relative">
                  <Input
                    id="password"
                    type={showPassword ? 'text' : 'password'}
                    placeholder="Enter your password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    required
                  />
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    className="absolute right-0 top-0 h-full px-3 py-2 hover:bg-transparent"
                    onClick={() => setShowPassword(!showPassword)}
                  >
                    {showPassword ? (
                      <EyeOff className="h-4 w-4" />
                    ) : (
                      <Eye className="h-4 w-4" />
                    )}
                  </Button>
                </div>
              </div>

              {error && (
                <Alert variant="destructive">
                  <AlertDescription>{error}</AlertDescription>
                </Alert>
              )}

              <Button
                type="submit"
                className="w-full"
                disabled={isLoading}
              >
                {isLoading ? 'Signing in...' : 'Sign In'}
              </Button>
            </form>
          </CardContent>
        </Card>

        {/* Demo Users */}
        <div className="space-y-4">
          <div className="text-center lg:text-left">
            <h2 className="text-2xl font-bold text-gray-900 mb-2">Try Different Roles</h2>
            <p className="text-gray-600">Click on any demo account to test the role-based interfaces</p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {demoUsers.map((user) => (
              <Card
                key={user.email}
                className="cursor-pointer hover:shadow-lg transition-shadow border-2 hover:border-blue-300"
                onClick={() => handleDemoLogin(user)}
              >
                <CardContent className="p-4">
                  <div className="flex items-center space-x-3">
                    <div className="p-2 bg-blue-100 rounded-lg">
                      {getRoleIcon(user.role)}
                    </div>
                    <div className="flex-1">
                      <h3 className="font-semibold text-gray-900">{user.name}</h3>
                      <p className="text-sm text-gray-600">{user.description}</p>
                      <p className="text-xs text-gray-500 mt-1">{user.email}</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>

          {/* Role Descriptions */}
          <div className="space-y-3">
            <h3 className="font-semibold text-gray-900">Role Overview</h3>
            <div className="space-y-2 text-sm text-gray-600">
              <div className="flex items-center space-x-2">
                <User className="w-4 h-4 text-blue-600" />
                <span><strong>Patient:</strong> Health monitoring, appointments, medical records</span>
              </div>
              <div className="flex items-center space-x-2">
                <Stethoscope className="w-4 h-4 text-green-600" />
                <span><strong>Provider:</strong> Patient care, AI diagnostics, clinical workflow</span>
              </div>
              <div className="flex items-center space-x-2">
                <Terminal className="w-4 h-4 text-purple-600" />
                <span><strong>Engineer:</strong> System monitoring, deployment, security</span>
              </div>
              <div className="flex items-center space-x-2">
                <Activity className="w-4 h-4 text-orange-600" />
                <span><strong>Admin:</strong> System administration, user management</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LoginScreen;
