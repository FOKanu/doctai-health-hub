import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { 
  MapPin, 
  Hospital, 
  Pills, 
  Dumbbell, 
  Utensils,
  Clock,
  Star,
  Phone,
  Navigation,
  Maximize2,
  Minimize2
} from 'lucide-react';

interface Location {
  id: string;
  name: string;
  type: 'hospital' | 'pharmacy' | 'fitness' | 'dental' | 'clinic';
  address: string;
  distance: string;
  rating: number;
  isOpen: boolean;
  openUntil?: string;
  phone?: string;
  coordinates: { lat: number; lng: number };
}

interface TreatmentMapProps {
  userLocation: { lat: number; lng: number } | null;
  activeFilters: string[];
  searchQuery: string;
  isExpanded: boolean;
  onToggleExpanded: () => void;
}

// Mock location data
const mockLocations: Location[] = [
  {
    id: '1',
    name: 'Medico Pharmacy',
    type: 'pharmacy',
    address: 'Unter den Linden 12, Berlin',
    distance: '350m',
    rating: 4.5,
    isOpen: true,
    openUntil: '20:00',
    phone: '+49 30 12345678',
    coordinates: { lat: 52.5170, lng: 13.4000 }
  },
  {
    id: '2',
    name: 'Charité Emergency',
    type: 'hospital',
    address: 'Charitéplatz 1, Berlin',
    distance: '800m',
    rating: 4.8,
    isOpen: true,
    openUntil: '24/7',
    phone: '+49 30 450 50',
    coordinates: { lat: 52.5250, lng: 13.3800 }
  },
  {
    id: '3',
    name: 'TheraFit Physiotherapy',
    type: 'fitness',
    address: 'Friedrichstr. 45, Berlin',
    distance: '500m',
    rating: 4.6,
    isOpen: true,
    openUntil: '19:00',
    phone: '+49 30 98765432',
    coordinates: { lat: 52.5180, lng: 13.4100 }
  },
  {
    id: '4',
    name: 'Dermacenter Berlin',
    type: 'clinic',
    address: 'Potsdamer Platz 8, Berlin',
    distance: '1.2km',
    rating: 4.7,
    isOpen: false,
    openUntil: 'Opens 08:00',
    phone: '+49 30 11223344',
    coordinates: { lat: 52.5100, lng: 13.3900 }
  },
  {
    id: '5',
    name: 'Smile Dental Clinic',
    type: 'dental',
    address: 'Alexanderplatz 15, Berlin',
    distance: '900m',
    rating: 4.3,
    isOpen: true,
    openUntil: '18:00',
    phone: '+49 30 55667788',
    coordinates: { lat: 52.5220, lng: 13.4130 }
  }
];

const TreatmentMap: React.FC<TreatmentMapProps> = ({ 
  userLocation, 
  activeFilters, 
  searchQuery,
  isExpanded,
  onToggleExpanded 
}) => {
  const [selectedLocation, setSelectedLocation] = useState<Location | null>(null);
  const [filteredLocations, setFilteredLocations] = useState<Location[]>(mockLocations);

  // Filter locations based on active filters and search query
  useEffect(() => {
    let filtered = mockLocations;

    // Apply type filters
    if (!activeFilters.includes('all')) {
      filtered = filtered.filter(location => activeFilters.includes(location.type));
    }

    // Apply search filter
    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase();
      filtered = filtered.filter(location => 
        location.name.toLowerCase().includes(query) ||
        location.address.toLowerCase().includes(query) ||
        location.type.toLowerCase().includes(query)
      );
    }

    setFilteredLocations(filtered);
  }, [activeFilters, searchQuery]);

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'hospital':
        return Hospital;
      case 'pharmacy':
        return Pharmacy;
      case 'fitness':
        return Dumbbell;
      case 'dental':
        return Utensils;
      default:
        return MapPin;
    }
  };

  const getTypeColor = (type: string) => {
    switch (type) {
      case 'hospital':
        return 'text-red-600 bg-red-100 border-red-200';
      case 'pharmacy':
        return 'text-green-600 bg-green-100 border-green-200';
      case 'fitness':
        return 'text-blue-600 bg-blue-100 border-blue-200';
      case 'dental':
        return 'text-purple-600 bg-purple-100 border-purple-200';
      default:
        return 'text-gray-600 bg-gray-100 border-gray-200';
    }
  };

  return (
    <div className="space-y-4">
      {/* Map Container */}
      <Card className={`transition-all duration-300 ${isExpanded ? 'fixed inset-4 z-50' : ''}`}>
        <CardHeader className="flex flex-row items-center justify-between pb-3">
          <div>
            <CardTitle className="text-lg">Healthcare Facilities Near You</CardTitle>
            <p className="text-sm text-gray-600">{filteredLocations.length} locations found</p>
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={onToggleExpanded}
            className="flex items-center gap-2"
          >
            {isExpanded ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
            {isExpanded ? 'Minimize' : 'Expand'}
          </Button>
        </CardHeader>
        <CardContent className="p-0">
          {/* Map Placeholder - In a real app, this would be Google Maps */}
          <div className={`bg-gray-100 border-2 border-dashed border-gray-300 flex items-center justify-center relative ${
            isExpanded ? 'h-96' : 'h-64'
          }`}>
            <div className="text-center">
              <MapPin className="w-12 h-12 text-gray-400 mx-auto mb-2" />
              <p className="text-gray-500 mb-2">Interactive Google Maps would load here</p>
              <p className="text-sm text-gray-400">
                {userLocation ? `User location: ${userLocation.lat.toFixed(4)}, ${userLocation.lng.toFixed(4)}` : 'Getting location...'}
              </p>
            </div>
            
            {/* Mock map pins */}
            {filteredLocations.slice(0, 5).map((location, index) => {
              const Icon = getTypeIcon(location.type);
              return (
                <div
                  key={location.id}
                  className={`absolute cursor-pointer transform -translate-x-1/2 -translate-y-1/2 ${
                    index === 0 ? 'top-1/3 left-1/4' :
                    index === 1 ? 'top-1/2 right-1/4' :
                    index === 2 ? 'bottom-1/3 left-1/3' :
                    index === 3 ? 'top-1/4 right-1/3' :
                    'bottom-1/4 left-1/2'
                  }`}
                  onClick={() => setSelectedLocation(location)}
                >
                  <div className={`p-2 rounded-full shadow-lg border-2 ${getTypeColor(location.type)}`}>
                    <Icon className="w-4 h-4" />
                  </div>
                </div>
              );
            })}
          </div>
        </CardContent>
      </Card>

      {/* Location List */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {filteredLocations.map((location) => {
          const Icon = getTypeIcon(location.type);
          return (
            <Card 
              key={location.id} 
              className={`cursor-pointer transition-all hover:shadow-md ${
                selectedLocation?.id === location.id ? 'ring-2 ring-blue-500' : ''
              }`}
              onClick={() => setSelectedLocation(location)}
            >
              <CardContent className="p-4">
                <div className="flex items-start justify-between mb-3">
                  <div className="flex items-center gap-3">
                    <div className={`p-2 rounded-lg ${getTypeColor(location.type)}`}>
                      <Icon className="w-4 h-4" />
                    </div>
                    <div>
                      <h3 className="font-medium text-gray-900">{location.name}</h3>
                      <p className="text-sm text-gray-600">{location.address}</p>
                    </div>
                  </div>
                  <Badge variant={location.isOpen ? "default" : "secondary"} className="text-xs">
                    {location.isOpen ? "Open" : "Closed"}
                  </Badge>
                </div>

                <div className="flex items-center justify-between text-sm">
                  <div className="flex items-center gap-4">
                    <div className="flex items-center gap-1">
                      <Navigation className="w-3 h-3 text-gray-400" />
                      <span className="text-gray-600">{location.distance}</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <Star className="w-3 h-3 text-yellow-500 fill-current" />
                      <span className="text-gray-600">{location.rating}</span>
                    </div>
                  </div>
                  <div className="flex items-center gap-1 text-gray-500">
                    <Clock className="w-3 h-3" />
                    <span className="text-xs">{location.openUntil}</span>
                  </div>
                </div>

                {location.phone && (
                  <div className="mt-3 pt-3 border-t border-gray-100">
                    <Button variant="outline" size="sm" className="w-full text-xs">
                      <Phone className="w-3 h-3 mr-2" />
                      Call {location.phone}
                    </Button>
                  </div>
                )}
              </CardContent>
            </Card>
          );
        })}
      </div>

      {filteredLocations.length === 0 && (
        <Card>
          <CardContent className="p-8 text-center">
            <MapPin className="w-12 h-12 text-gray-400 mx-auto mb-4" />
            <h3 className="font-medium text-gray-900 mb-2">No locations found</h3>
            <p className="text-gray-600">Try adjusting your filters or search query</p>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default TreatmentMap;
