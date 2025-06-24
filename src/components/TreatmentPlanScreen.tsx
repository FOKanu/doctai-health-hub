
import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from '@/components/ui/accordion';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  Search, 
  MapPin, 
  Calendar, 
  Pill, 
  Hospital, 
  Pharmacy, 
  Dumbbell, 
  Utensils,
  Phone,
  Clock,
  Star,
  Filter,
  Plus,
  FileText,
  User
} from 'lucide-react';
import TreatmentMap from './treatment/TreatmentMap';
import TreatmentOverview from './treatment/TreatmentOverview';
import QuickActions from './treatment/QuickActions';

// Mock treatment data
const mockTreatmentData = {
  userName: "Anna Schmitt",
  diagnosis: "Eczema",
  prescriptions: [
    { 
      medication: "Hydrocortisone Cream", 
      dosage: "Apply twice daily", 
      duration: "2 weeks",
      instructions: "Apply to affected areas after cleansing"
    },
    { 
      medication: "Cetirizine", 
      dosage: "10mg once daily", 
      duration: "1 month",
      instructions: "Take with food, preferably in the evening"
    }
  ],
  specialistRecommendations: ["Dermatologist", "Allergy Testing"],
  nextCheckup: "2025-07-05",
  assignedClinic: "Charité – Universitätsmedizin Berlin",
  notes: "Avoid fragranced products and heat exposure. Monitor skin condition daily.",
  progress: {
    improvementScore: 75,
    lastUpdated: "2025-06-20"
  }
};

const TreatmentPlanScreen = () => {
  const [searchQuery, setSearchQuery] = useState('');
  const [activeFilters, setActiveFilters] = useState<string[]>(['all']);
  const [userLocation, setUserLocation] = useState<{lat: number, lng: number} | null>(null);
  const [isMapExpanded, setIsMapExpanded] = useState(false);

  // Get user location
  useEffect(() => {
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          setUserLocation({
            lat: position.coords.latitude,
            lng: position.coords.longitude
          });
        },
        (error) => {
          console.log('Location access denied, using default location');
          // Default to Berlin coordinates
          setUserLocation({ lat: 52.5200, lng: 13.4050 });
        }
      );
    } else {
      setUserLocation({ lat: 52.5200, lng: 13.4050 });
    }
  }, []);

  const filterOptions = [
    { id: 'all', label: 'All', icon: MapPin },
    { id: 'hospital', label: 'Hospitals', icon: Hospital },
    { id: 'pharmacy', label: 'Pharmacies', icon: Pharmacy },
    { id: 'fitness', label: 'Fitness', icon: Dumbbell },
    { id: 'dental', label: 'Dental', icon: Utensils }
  ];

  const toggleFilter = (filterId: string) => {
    if (filterId === 'all') {
      setActiveFilters(['all']);
    } else {
      const newFilters = activeFilters.includes(filterId)
        ? activeFilters.filter(f => f !== filterId)
        : [...activeFilters.filter(f => f !== 'all'), filterId];
      
      setActiveFilters(newFilters.length === 0 ? ['all'] : newFilters);
    }
  };

  const searchSuggestions = [
    "Open pharmacy near me",
    "Emergency hospital",
    "Dermatologist clinic",
    "24/7 pharmacy",
    "Skin specialist",
    "Allergy testing center"
  ];

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white border-b border-gray-200 px-4 py-6">
        <div className="max-w-7xl mx-auto">
          <h1 className="text-2xl font-bold text-gray-900 mb-2">Treatment Plan</h1>
          <p className="text-gray-600">Manage your health journey and find nearby care</p>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 py-6 space-y-6">
        {/* Search Bar */}
        <Card>
          <CardContent className="p-4">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
              <Input
                placeholder="Search for healthcare facilities, specialists, or services..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-10 pr-4"
              />
            </div>
            
            {/* Quick Suggestions */}
            {searchQuery === '' && (
              <div className="mt-3 flex flex-wrap gap-2">
                {searchSuggestions.slice(0, 3).map((suggestion, index) => (
                  <Button
                    key={index}
                    variant="outline"
                    size="sm"
                    onClick={() => setSearchQuery(suggestion)}
                    className="text-xs"
                  >
                    {suggestion}
                  </Button>
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Filter Toggles */}
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-2 mb-3">
              <Filter className="w-4 h-4 text-gray-600" />
              <span className="text-sm font-medium text-gray-700">Filter by type:</span>
            </div>
            <div className="flex flex-wrap gap-2">
              {filterOptions.map((filter) => {
                const Icon = filter.icon;
                const isActive = activeFilters.includes(filter.id);
                return (
                  <Button
                    key={filter.id}
                    variant={isActive ? "default" : "outline"}
                    size="sm"
                    onClick={() => toggleFilter(filter.id)}
                    className="flex items-center gap-2"
                  >
                    <Icon className="w-3 h-3" />
                    {filter.label}
                  </Button>
                );
              })}
            </div>
          </CardContent>
        </Card>

        {/* Main Content Tabs */}
        <Tabs defaultValue="overview" className="space-y-6">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="overview">Treatment Overview</TabsTrigger>
            <TabsTrigger value="map">Find Care Nearby</TabsTrigger>
          </TabsList>

          <TabsContent value="overview" className="space-y-6">
            <TreatmentOverview treatmentData={mockTreatmentData} />
          </TabsContent>

          <TabsContent value="map" className="space-y-6">
            <TreatmentMap 
              userLocation={userLocation}
              activeFilters={activeFilters}
              searchQuery={searchQuery}
              isExpanded={isMapExpanded}
              onToggleExpanded={() => setIsMapExpanded(!isMapExpanded)}
            />
          </TabsContent>
        </Tabs>
      </div>

      {/* Floating Quick Actions */}
      <QuickActions onPharmacySearch={() => setSearchQuery('pharmacy near me')} />
    </div>
  );
};

export default TreatmentPlanScreen;
