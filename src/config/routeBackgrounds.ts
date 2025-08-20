// Route-specific background configuration for DoctAI
export interface RouteBackground {
  id: string;
  name: string;
  url: string;
  overlay: string;
  darkOverlay?: string;
  description: string;
}

export const routeBackgrounds: Record<string, RouteBackground> = {
  // Login / Registration
  '/login': {
    id: 'login-ai-network',
    name: 'AI Network Login',
    url: 'https://images.unsplash.com/photo-1559757148-5c350d0d3c56?auto=format&fit=crop&w=1920&q=80',
    overlay: 'linear-gradient(135deg, rgba(59, 130, 246, 0.4), rgba(147, 197, 253, 0.5))',
    darkOverlay: 'linear-gradient(135deg, rgba(30, 58, 138, 0.6), rgba(59, 130, 246, 0.4))',
    description: 'Futuristic DNA helix with soft blue gradient overlay'
  },
  '/register': {
    id: 'register-hospital',
    name: 'Hospital Registration',
    url: 'https://images.unsplash.com/photo-1576091160399-112ba8d25d1f?auto=format&fit=crop&w=1920&q=80',
    overlay: 'linear-gradient(135deg, rgba(16, 185, 129, 0.4), rgba(167, 243, 208, 0.5))',
    darkOverlay: 'linear-gradient(135deg, rgba(6, 95, 70, 0.6), rgba(16, 185, 129, 0.4))',
    description: 'Modern digital healthcare with teal gradient'
  },

  // Patient Portal
  '/patient': {
    id: 'patient-care',
    name: 'Patient Care',
    url: 'https://images.unsplash.com/photo-1582719471384-894fbb16e074?auto=format&fit=crop&w=1920&q=80',
    overlay: 'linear-gradient(135deg, rgba(99, 102, 241, 0.3), rgba(196, 181, 253, 0.4))',
    darkOverlay: 'linear-gradient(135deg, rgba(55, 48, 163, 0.6), rgba(99, 102, 241, 0.4))',
    description: 'Human-centered medical research laboratory'
  },
  '/patient/dashboard': {
    id: 'patient-dashboard',
    name: 'Patient Dashboard',
    url: 'https://images.unsplash.com/photo-1559757175-0eb30cd8c063?auto=format&fit=crop&w=1920&q=80',
    overlay: 'linear-gradient(135deg, rgba(14, 165, 233, 0.3), rgba(186, 230, 253, 0.4))',
    darkOverlay: 'linear-gradient(135deg, rgba(3, 105, 161, 0.6), rgba(14, 165, 233, 0.4))',
    description: 'Neural pathways representing patient health data'
  },

  // Healthcare Provider
  '/provider': {
    id: 'provider-dashboard',
    name: 'Provider Dashboard',
    url: 'https://images.unsplash.com/photo-1576086213369-97a306d36557?auto=format&fit=crop&w=1920&q=80',
    overlay: 'linear-gradient(135deg, rgba(139, 92, 246, 0.3), rgba(221, 214, 254, 0.4))',
    darkOverlay: 'linear-gradient(135deg, rgba(91, 33, 182, 0.6), rgba(139, 92, 246, 0.4))',
    description: 'Molecular network for healthcare providers'
  },

  // Staff Engineer
  '/staff': {
    id: 'staff-tech',
    name: 'Staff Engineering',
    url: 'https://images.unsplash.com/photo-1628595351029-c2bf17511435?auto=format&fit=crop&w=1920&q=80',
    overlay: 'linear-gradient(135deg, rgba(6, 182, 212, 0.3), rgba(165, 243, 252, 0.4))',
    darkOverlay: 'linear-gradient(135deg, rgba(8, 145, 178, 0.6), rgba(6, 182, 212, 0.4))',
    description: 'Cellular biology tech fusion for engineering'
  },

  // Appointments
  '/appointments': {
    id: 'appointments-calendar',
    name: 'Appointments',
    url: 'linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 25%, #bae6fd 50%, #7dd3fc 75%, #38bdf8 100%)',
    overlay: 'linear-gradient(135deg, rgba(255, 255, 255, 0.4), rgba(59, 130, 246, 0.3))',
    darkOverlay: 'linear-gradient(135deg, rgba(30, 41, 59, 0.6), rgba(59, 130, 246, 0.4))',
    description: 'Calendar-inspired gradient for scheduling'
  },

  // Diagnostics / AI
  '/diagnostics': {
    id: 'diagnostics-ai',
    name: 'AI Diagnostics',
    url: 'https://images.unsplash.com/photo-1559757146-8c3d6e7a3e2b?auto=format&fit=crop&w=1920&q=80',
    overlay: 'linear-gradient(135deg, rgba(168, 85, 247, 0.3), rgba(233, 213, 255, 0.4))',
    darkOverlay: 'linear-gradient(135deg, rgba(109, 40, 217, 0.6), rgba(168, 85, 247, 0.4))',
    description: 'Heartbeat wave patterns for AI analysis'
  },
  '/ai-assistant': {
    id: 'ai-assistant',
    name: 'AI Assistant',
    url: 'https://images.unsplash.com/photo-1559757148-5c350d0d3c56?auto=format&fit=crop&w=1920&q=80',
    overlay: 'linear-gradient(135deg, rgba(16, 185, 129, 0.3), rgba(167, 243, 208, 0.4))',
    darkOverlay: 'linear-gradient(135deg, rgba(6, 95, 70, 0.6), rgba(16, 185, 129, 0.4))',
    description: 'DNA helix for AI health assistant'
  },

  // Medications/Prescriptions
  '/medications': {
    id: 'medications',
    name: 'Medications',
    url: 'linear-gradient(135deg, #fef3c7 0%, #fcd34d 25%, #f59e0b 50%, #d97706 75%, #92400e 100%)',
    overlay: 'linear-gradient(135deg, rgba(255, 255, 255, 0.5), rgba(245, 158, 11, 0.3))',
    darkOverlay: 'linear-gradient(135deg, rgba(30, 41, 59, 0.6), rgba(245, 158, 11, 0.4))',
    description: 'Pharmacy-inspired warm gradient'
  },
  '/prescriptions': {
    id: 'prescriptions',
    name: 'Prescriptions',
    url: 'https://images.unsplash.com/photo-1576086213369-97a306d36557?auto=format&fit=crop&w=1920&q=80',
    overlay: 'linear-gradient(135deg, rgba(34, 197, 94, 0.3), rgba(187, 247, 208, 0.4))',
    darkOverlay: 'linear-gradient(135deg, rgba(21, 128, 61, 0.6), rgba(34, 197, 94, 0.4))',
    description: 'Molecular patterns for prescription management'
  },

  // Telemedicine
  '/telemedicine': {
    id: 'telemedicine',
    name: 'Telemedicine',
    url: 'linear-gradient(135deg, #ede9fe 0%, #c4b5fd 25%, #a78bfa 50%, #8b5cf6 75%, #7c3aed 100%)',
    overlay: 'linear-gradient(135deg, rgba(255, 255, 255, 0.4), rgba(139, 92, 246, 0.3))',
    darkOverlay: 'linear-gradient(135deg, rgba(30, 41, 59, 0.6), rgba(139, 92, 246, 0.4))',
    description: 'Communication wave patterns for teleconsultation'
  },
  '/video-call': {
    id: 'video-call',
    name: 'Video Call',
    url: 'https://images.unsplash.com/photo-1576091160399-112ba8d25d1f?auto=format&fit=crop&w=1920&q=80',
    overlay: 'linear-gradient(135deg, rgba(59, 130, 246, 0.3), rgba(191, 219, 254, 0.4))',
    darkOverlay: 'linear-gradient(135deg, rgba(30, 58, 138, 0.6), rgba(59, 130, 246, 0.4))',
    description: 'Digital healthcare technology for video calls'
  },

  // Settings/Profile
  '/settings': {
    id: 'settings',
    name: 'Settings',
    url: 'linear-gradient(135deg, #f8fafc 0%, #e2e8f0 25%, #cbd5e1 50%, #94a3b8 75%, #64748b 100%)',
    overlay: 'linear-gradient(135deg, rgba(255, 255, 255, 0.6), rgba(148, 163, 184, 0.3))',
    darkOverlay: 'linear-gradient(135deg, rgba(15, 23, 42, 0.6), rgba(100, 116, 139, 0.4))',
    description: 'Neutral geometric gradient for settings'
  },
  '/profile': {
    id: 'profile',
    name: 'Profile',
    url: 'linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 25%, #cbd5e1 50%, #94a3b8 75%, #64748b 100%)',
    overlay: 'linear-gradient(135deg, rgba(255, 255, 255, 0.5), rgba(100, 116, 139, 0.3))',
    darkOverlay: 'linear-gradient(135deg, rgba(15, 23, 42, 0.6), rgba(100, 116, 139, 0.4))',
    description: 'Clean profile gradient with subtle medical watermark'
  },

  // Fallback/Default
  default: {
    id: 'default-medical',
    name: 'Default Medical',
    url: 'linear-gradient(135deg, #00d4aa 0%, #00a3cc 25%, #0066ff 50%, #6366f1 75%, #8b5cf6 100%)',
    overlay: 'linear-gradient(135deg, rgba(255, 255, 255, 0.4), rgba(99, 102, 241, 0.3))',
    darkOverlay: 'linear-gradient(135deg, rgba(30, 41, 59, 0.6), rgba(99, 102, 241, 0.4))',
    description: 'Dynamic health-themed gradient'
  }
};

// Helper function to get background for current route
export const getRouteBackground = (pathname: string, isDarkMode = false): RouteBackground => {
  // Find exact match first
  if (routeBackgrounds[pathname]) {
    return routeBackgrounds[pathname];
  }

  // Find partial match for nested routes
  const matchingRoute = Object.keys(routeBackgrounds).find(route => 
    pathname.startsWith(route) && route !== 'default'
  );

  if (matchingRoute) {
    return routeBackgrounds[matchingRoute];
  }

  // Return default
  return routeBackgrounds.default;
};

// Helper to get all available backgrounds for settings
export const getAllRouteBackgrounds = (): RouteBackground[] => {
  return Object.values(routeBackgrounds);
};