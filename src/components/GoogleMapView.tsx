import React, { useState, useCallback, useRef, useEffect } from 'react';
import { GoogleMap, LoadScript, Marker, InfoWindow } from '@react-google-maps/api';
import { MapPin, Navigation, Layers, Search, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { HealthcareProvider } from '../types/healthcare';

interface GoogleMapViewProps {
  providers?: HealthcareProvider[];
  onProviderSelect?: (provider: HealthcareProvider) => void;
}

interface PlaceResult {
  place_id: string;
  name: string;
  vicinity: string;
  geometry: {
    location: {
      lat: () => number;
      lng: () => number;
    };
  };
  types: string[];
  rating?: number;
  business_status?: string;
  photos?: google.maps.places.PlacePhoto[];
}

const libraries: ("places")[] = ["places"];

const mapContainerStyle = {
  width: '100%',
  height: '800px'
};

const defaultCenter = {
  lat: 52.5200,
  lng: 13.4050
};

export function GoogleMapView({ providers = [], onProviderSelect }: GoogleMapViewProps) {
  const [userLocation, setUserLocation] = useState<google.maps.LatLngLiteral | null>(null);
  const [map, setMap] = useState<google.maps.Map | null>(null);
  const [searchInput, setSearchInput] = useState('');
  const [nearbyPlaces, setNearbyPlaces] = useState<PlaceResult[]>([]);
  const [selectedPlace, setSelectedPlace] = useState<PlaceResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [locationLoading, setLocationLoading] = useState(true);
  const [apiKeyMissing, setApiKeyMissing] = useState(false);

  const autocompleteRef = useRef<google.maps.places.Autocomplete | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Get user's current location
  useEffect(() => {
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          const location = {
            lat: position.coords.latitude,
            lng: position.coords.longitude
          };
          setUserLocation(location);
          setLocationLoading(false);
        },
        (error) => {
          console.log('Location access denied:', error);
          setUserLocation(defaultCenter);
          setLocationLoading(false);
        }
      );
    } else {
      setUserLocation(defaultCenter);
      setLocationLoading(false);
    }
  }, []);

  const onLoad = useCallback((map: google.maps.Map) => {
    setMap(map);

    // Initialize autocomplete
    if (inputRef.current && window.google) {
      const autocomplete = new window.google.maps.places.Autocomplete(inputRef.current, {
        types: ['establishment'],
        fields: ['place_id', 'name', 'geometry', 'types', 'vicinity', 'rating']
      });

      autocomplete.addListener('place_changed', () => {
        const place = autocomplete.getPlace();
        if (place.geometry?.location) {
          const location = {
            lat: place.geometry.location.lat(),
            lng: place.geometry.location.lng()
          };
          map.panTo(location);
          map.setZoom(15);
          searchNearbyPlaces(location);
        }
      });

      autocompleteRef.current = autocomplete;
    }
  }, []);



  const handleCurrentLocation = useCallback(() => {
    if (userLocation && map) {
      map.panTo(userLocation);
      map.setZoom(15);
      // Call searchNearbyPlaces directly to avoid circular dependency
      if (!map || !window.google) return;

      setLoading(true);
      const service = new window.google.maps.places.PlacesService(map);

      const request = {
        location: new window.google.maps.LatLng(userLocation.lat, userLocation.lng),
        radius: 5000, // 5km radius
        type: 'health',
        keyword: 'doctor hospital clinic pharmacy dentist medical'
      };

      service.nearbySearch(request, (results, status) => {
        if (status === window.google.maps.places.PlacesServiceStatus.OK && results) {
          const healthcareResults = results.filter(place =>
            place.types?.some(type =>
              ['hospital', 'doctor', 'dentist', 'pharmacy', 'physiotherapist', 'health'].includes(type)
            )
          ) as PlaceResult[];

          setNearbyPlaces(healthcareResults.slice(0, 20)); // Limit to 20 results
        }
        setLoading(false);
      });
    }
  }, [userLocation, map, setLoading, setNearbyPlaces]);

  const handleMarkerClick = (place: PlaceResult) => {
    setSelectedPlace(place);
  };

  const getPlaceTypeIcon = (types: string[]) => {
    if (types.includes('hospital')) return '🏥';
    if (types.includes('pharmacy')) return '💊';
    if (types.includes('dentist')) return '🦷';
    if (types.includes('doctor')) return '👨‍⚕️';
    return '🏥';
  };

  // Check if Google Maps API key is available
  const googleMapsApiKey = import.meta.env.VITE_GOOGLE_MAPS_API_KEY;

  if (!googleMapsApiKey) {
    return (
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 h-96 flex items-center justify-center p-8">
        <div className="text-center max-w-md">
          <MapPin className="w-16 h-16 text-blue-600 mx-auto mb-4" />
          <h3 className="text-lg font-semibold mb-2">Google Maps Integration</h3>
          <p className="text-gray-600 mb-4 text-sm">
            To view healthcare providers on the map, please add your Google Maps API key.
            You can get one from the{' '}
            <a href="https://console.cloud.google.com/apis/credentials" target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline">
              Google Cloud Console
            </a>
          </p>
          <p className="text-xs text-gray-500">
            Make sure to enable Maps JavaScript API and Places API
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden">
      {/* Map Header */}
      <div className="p-4 border-b border-gray-200 bg-gray-50">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <MapPin className="w-5 h-5 text-blue-600" />
            <h3 className="font-medium">Healthcare Providers Near You</h3>
          </div>
          <div className="flex items-center gap-2">
            <Button
              size="sm"
              variant="outline"
              className="text-xs"
              onClick={handleCurrentLocation}
              disabled={locationLoading}
            >
              {locationLoading ? (
                <Loader2 className="w-3 h-3 mr-1 animate-spin" />
              ) : (
                <Navigation className="w-3 h-3 mr-1" />
              )}
              My Location
            </Button>
          </div>
        </div>

        {/* Search Input */}
        <div className="relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
          <Input
            ref={inputRef}
            placeholder="Search for healthcare services..."
            className="pl-10"
            value={searchInput}
            onChange={(e) => setSearchInput(e.target.value)}
          />
        </div>
      </div>

      {/* Map and Results */}
      <div className="relative">
        <LoadScript
          googleMapsApiKey={googleMapsApiKey}
          libraries={libraries}
          loadingElement={
            <div className="h-96 flex items-center justify-center">
              <Loader2 className="w-8 h-8 animate-spin text-blue-600" />
            </div>
          }
        >
          <GoogleMap
            mapContainerStyle={mapContainerStyle}
            center={userLocation || defaultCenter}
            zoom={userLocation ? 13 : 10}
            onLoad={onLoad}
            options={{
              disableDefaultUI: false,
              zoomControl: true,
              streetViewControl: false,
              mapTypeControl: false,
              fullscreenControl: false,
            }}
          >
            {/* User Location Marker */}
            {userLocation && (
              <Marker
                position={userLocation}
                icon={{
                  url: 'data:image/svg+xml;charset=UTF-8,' + encodeURIComponent(`
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                      <circle cx="12" cy="12" r="8" fill="#3B82F6" stroke="white" stroke-width="3"/>
                      <circle cx="12" cy="12" r="3" fill="white"/>
                    </svg>
                  `),
                  scaledSize: new window.google.maps.Size(24, 24),
                }}
                title="Your Location"
              />
            )}

            {/* Healthcare Provider Markers */}
            {nearbyPlaces.map((place) => (
              <Marker
                key={place.place_id}
                position={{
                  lat: place.geometry.location.lat(),
                  lng: place.geometry.location.lng()
                }}
                onClick={() => handleMarkerClick(place)}
                icon={{
                  url: 'data:image/svg+xml;charset=UTF-8,' + encodeURIComponent(`
                    <svg width="32" height="32" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                      <path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z" fill="#DC2626" stroke="white" stroke-width="2"/>
                      <circle cx="12" cy="10" r="3" fill="white"/>
                    </svg>
                  `),
                  scaledSize: new window.google.maps.Size(32, 32),
                }}
              />
            ))}

            {/* Info Window */}
            {selectedPlace && (
              <InfoWindow
                position={{
                  lat: selectedPlace.geometry.location.lat(),
                  lng: selectedPlace.geometry.location.lng()
                }}
                onCloseClick={() => setSelectedPlace(null)}
              >
                <div className="p-2 max-w-xs">
                  <div className="flex items-center gap-2 mb-2">
                    <span className="text-lg">{getPlaceTypeIcon(selectedPlace.types)}</span>
                    <h4 className="font-medium text-sm">{selectedPlace.name}</h4>
                  </div>
                  <p className="text-xs text-gray-600 mb-2">{selectedPlace.vicinity}</p>
                  {selectedPlace.rating && (
                    <div className="flex items-center gap-1">
                      <span className="text-yellow-500">⭐</span>
                      <span className="text-xs">{selectedPlace.rating}</span>
                    </div>
                  )}
                  <div className="mt-2 flex flex-wrap gap-1">
                    {selectedPlace.types.slice(0, 3).map((type) => (
                      <Badge key={type} variant="secondary" className="text-xs capitalize">
                        {type.replace(/_/g, ' ')}
                      </Badge>
                    ))}
                  </div>
                </div>
              </InfoWindow>
            )}
          </GoogleMap>
        </LoadScript>

        {/* Loading Overlay */}
        {loading && (
          <div className="absolute inset-0 bg-white/80 flex items-center justify-center">
            <div className="flex items-center gap-2 text-blue-600">
              <Loader2 className="w-5 h-5 animate-spin" />
              <span className="text-sm font-medium">Searching nearby providers...</span>
            </div>
          </div>
        )}
      </div>

      {/* Results Summary */}
      {nearbyPlaces.length > 0 && (
        <div className="p-4 border-t border-gray-200 bg-gray-50">
          <p className="text-sm text-gray-600">
            Found {nearbyPlaces.length} healthcare providers within 5km
          </p>
        </div>
      )}
    </div>
  );
}
