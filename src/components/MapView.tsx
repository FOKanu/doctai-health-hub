import React, { useState } from 'react';
import { Map, MapPin, Navigation, Layers } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent } from '@/components/ui/card';

interface MapViewProps {
  providers?: any[];
}

export function MapView({ providers = [] }: MapViewProps) {
  const [mapboxToken, setMapboxToken] = useState('');
  const [showTokenInput, setShowTokenInput] = useState(true);

  // Mock provider locations for demonstration
  const mockLocations = [
    { id: 1, name: 'Dr. Sarah Weber', lat: 52.5200, lng: 13.4050, type: 'Dermatology' },
    { id: 2, name: 'Prof. Dr. Michael Braun', lat: 48.1351, lng: 11.5820, type: 'Oncology' },
    { id: 3, name: 'Hamburg Central Pharmacy', lat: 53.5511, lng: 9.9937, type: 'Pharmacy' },
    { id: 4, name: 'Dr. Anna Müller', lat: 52.5025, lng: 13.3356, type: 'Dentist' }
  ];

  const handleMapboxSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (mapboxToken.trim()) {
      setShowTokenInput(false);
      // Here you would initialize the actual Mapbox map
    }
  };

  if (showTokenInput) {
    return (
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 h-96 flex items-center justify-center p-8">
        <div className="text-center max-w-md">
          <Map className="w-16 h-16 text-blue-600 mx-auto mb-4" />
          <h3 className="text-lg font-semibold mb-2">Interactive Map</h3>
          <p className="text-gray-600 mb-4 text-sm">
            To view provider locations on an interactive map, please enter your Mapbox public token.
            You can get one for free at{' '}
            <a href="https://mapbox.com" target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline">
              mapbox.com
            </a>
          </p>
          
          <form onSubmit={handleMapboxSubmit} className="space-y-3">
            <Input
              type="password"
              placeholder="Enter Mapbox public token"
              value={mapboxToken}
              onChange={(e) => setMapboxToken(e.target.value)}
              className="w-full"
            />
            <Button type="submit" className="w-full" disabled={!mapboxToken.trim()}>
              Load Map
            </Button>
          </form>
          
          <Button 
            variant="outline" 
            className="w-full mt-2" 
            onClick={() => setShowTokenInput(false)}
          >
            View Static Locations
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 h-96 overflow-hidden">
      {/* Map Header */}
      <div className="p-4 border-b border-gray-200 bg-gray-50">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <MapPin className="w-5 h-5 text-blue-600" />
            <h3 className="font-medium">Provider Locations</h3>
          </div>
          <div className="flex items-center gap-2">
            <Button size="sm" variant="outline" className="text-xs">
              <Navigation className="w-3 h-3 mr-1" />
              My Location
            </Button>
            <Button size="sm" variant="outline" className="text-xs">
              <Layers className="w-3 h-3 mr-1" />
              Layers
            </Button>
          </div>
        </div>
      </div>

      {/* Map Content */}
      <div className="h-full bg-gradient-to-br from-blue-50 to-green-50 relative overflow-auto">
        {/* Simulated Map Background */}
        <div className="absolute inset-0 bg-grid-pattern opacity-10"></div>
        
        {/* Provider Locations */}
        <div className="relative h-full p-4 space-y-3">
          {mockLocations.map((location, index) => (
            <Card key={location.id} className="max-w-xs hover:shadow-md transition-shadow cursor-pointer">
              <CardContent className="p-3">
                <div className="flex items-start gap-3">
                  <div className="w-3 h-3 bg-blue-600 rounded-full mt-1 flex-shrink-0"></div>
                  <div className="min-w-0">
                    <h4 className="font-medium text-sm text-gray-900 truncate">
                      {location.name}
                    </h4>
                    <p className="text-xs text-gray-600">{location.type}</p>
                    <p className="text-xs text-blue-600 mt-1">
                      {location.lat.toFixed(4)}, {location.lng.toFixed(4)}
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>

        {/* Map Instructions */}
        <div className="absolute bottom-4 left-4 right-4">
          <div className="bg-white/90 backdrop-blur-sm rounded-lg p-3 text-xs text-gray-600">
            <p className="font-medium mb-1">Map Demo Mode</p>
            <p>This is a demonstration of the map layout. With a Mapbox token, this would show an interactive map with provider locations, directions, and real-time navigation.</p>
          </div>
        </div>
      </div>
    </div>
  );
}