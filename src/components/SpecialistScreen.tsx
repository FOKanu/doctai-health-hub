
import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowLeft, List, Map, Filter, X } from 'lucide-react';
import SearchBar from './findcare/SearchBar';
import FilterSidebar from './findcare/FilterSidebar';
import ProviderCard from './findcare/ProviderCard';
import { GoogleMapView } from './GoogleMapView';
import { HealthcareProvider, FilterOptions, BookingRequest } from '../types/healthcare';

const SpecialistScreen = () => {
  const navigate = useNavigate();
  const [viewMode, setViewMode] = useState<'list' | 'map'>('list');
  const [showFilters, setShowFilters] = useState(false);
  const [providers, setProviders] = useState<HealthcareProvider[]>([]);
  const [filteredProviders, setFilteredProviders] = useState<HealthcareProvider[]>([]);
  const [loading, setLoading] = useState(false);
  const [userLocation, setUserLocation] = useState<{ lat: number; lng: number } | null>(null);

  const [filters, setFilters] = useState<FilterOptions>({
    providerType: [],
    specialties: [],
    maxDistance: 25,
    minRating: 0,
    languages: [],
    availability: 'any',
    insuranceAccepted: [],
    priceRange: []
  });

  // Mock data - replace with actual API calls
  const mockProviders: HealthcareProvider[] = [
    {
      id: '1',
      name: 'Dr. Sarah Weber',
      type: 'doctor',
      specialty: 'Dermatology',
      clinicName: 'Berlin Medical Center',
      yearsExperience: 15,
      languages: ['German', 'English'],
      rating: 4.9,
      reviewCount: 127,
      priceRange: 'medium',
      insuranceAccepted: ['Public Health Insurance', 'Private Insurance'],
      location: {
        address: 'Alexanderplatz 1, Berlin',
        lat: 52.5200,
        lng: 13.4050,
        distance: 2.3
      },
      availability: {
        nextAvailable: 'Tomorrow, 14:30',
        isOpenNow: true,
        hours: '8:00 AM - 6:00 PM'
      },
      contactInfo: {
        phone: '+49 30 12345678',
        email: 'dr.weber@berlinmedical.de'
      },
      services: ['Skin Cancer Screening', 'Acne Treatment', 'Cosmetic Dermatology'],
      image: 'https://images.unsplash.com/photo-1559839734-2b71ea197ec2?w=150&h=150&fit=crop&crop=face'
    },
    {
      id: '2',
      name: 'Prof. Dr. Michael Braun',
      type: 'doctor',
      specialty: 'Oncology',
      hospitalName: 'Munich University Hospital',
      yearsExperience: 22,
      languages: ['German', 'English', 'French'],
      rating: 4.8,
      reviewCount: 203,
      priceRange: 'high',
      insuranceAccepted: ['Public Health Insurance', 'Private Insurance'],
      location: {
        address: 'Marchioninistraße 15, Munich',
        lat: 48.1351,
        lng: 11.5820,
        distance: 5.1
      },
      availability: {
        nextAvailable: 'Friday, 10:00',
        isOpenNow: false,
        hours: '9:00 AM - 5:00 PM'
      },
      contactInfo: {
        phone: '+49 89 98765432',
        email: 'prof.braun@uniklinik-muenchen.de'
      },
      services: ['Cancer Treatment', 'Chemotherapy', 'Radiation Therapy'],
      image: 'https://images.unsplash.com/photo-1612349317150-e413f6a5b16d?w=150&h=150&fit=crop&crop=face'
    },
    {
      id: '3',
      name: 'Hamburg Central Pharmacy',
      type: 'pharmacy',
      yearsExperience: 8,
      languages: ['German', 'English'],
      rating: 4.5,
      reviewCount: 89,
      location: {
        address: 'Mönckebergstraße 7, Hamburg',
        lat: 53.5511,
        lng: 9.9937,
        distance: 1.2
      },
      availability: {
        nextAvailable: 'Open now',
        isOpenNow: true,
        hours: '24/7'
      },
      contactInfo: {
        phone: '+49 40 11223344'
      },
      services: ['Prescription Filling', 'Health Consultation', 'COVID-19 Testing']
    },
    {
      id: '4',
      name: 'Dr. Anna Müller',
      type: 'dentist',
      clinicName: 'Smile Dental Clinic',
      yearsExperience: 12,
      languages: ['German', 'English'],
      rating: 4.7,
      reviewCount: 156,
      priceRange: 'medium',
      insuranceAccepted: ['Public Health Insurance'],
      location: {
        address: 'Kurfürstendamm 101, Berlin',
        lat: 52.5025,
        lng: 13.3356,
        distance: 3.7
      },
      availability: {
        nextAvailable: 'Monday, 16:45',
        isOpenNow: true,
        hours: '8:00 AM - 8:00 PM'
      },
      contactInfo: {
        phone: '+49 30 55667788',
        email: 'info@smile-dental.de'
      },
      services: ['General Dentistry', 'Teeth Whitening', 'Orthodontics']
    }
  ];

  useEffect(() => {
    // Get user's location
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          setUserLocation({
            lat: position.coords.latitude,
            lng: position.coords.longitude
          });
        },
        (error) => {
          console.log('Location access denied:', error);
        }
      );
    }

    // Load initial data
    setLoading(true);
    setTimeout(() => {
      setProviders(mockProviders);
      setFilteredProviders(mockProviders);
      setLoading(false);
    }, 1000);
  }, []);

  useEffect(() => {
    // Apply filters
    let filtered = providers;

    if (filters.providerType.length > 0) {
      filtered = filtered.filter(p => filters.providerType.includes(p.type));
    }

    if (filters.specialties.length > 0 && filters.specialties.some(s => s !== '')) {
      filtered = filtered.filter(p => 
        p.specialty && filters.specialties.includes(p.specialty)
      );
    }

    if (filters.maxDistance > 0) {
      filtered = filtered.filter(p => 
        !p.location.distance || p.location.distance <= filters.maxDistance
      );
    }

    if (filters.minRating > 0) {
      filtered = filtered.filter(p => p.rating >= filters.minRating);
    }

    if (filters.languages.length > 0) {
      filtered = filtered.filter(p => 
        filters.languages.some(lang => p.languages.includes(lang))
      );
    }

    if (filters.availability === 'today') {
      filtered = filtered.filter(p => p.availability.isOpenNow);
    }

    if (filters.priceRange.length > 0) {
      filtered = filtered.filter(p => 
        p.priceRange && filters.priceRange.includes(p.priceRange)
      );
    }

    setFilteredProviders(filtered);
  }, [filters, providers]);

  const handleSearch = (query: string) => {
    setLoading(true);
    // Mock search - replace with actual API call
    setTimeout(() => {
      const searchResults = mockProviders.filter(provider =>
        provider.name.toLowerCase().includes(query.toLowerCase()) ||
        provider.specialty?.toLowerCase().includes(query.toLowerCase()) ||
        provider.services.some(service => 
          service.toLowerCase().includes(query.toLowerCase())
        )
      );
      setFilteredProviders(searchResults);
      setLoading(false);
    }, 500);
  };

  const handleLocationSearch = (location: string) => {
    console.log('Searching location:', location);
    // Implement location-based search
  };

  const handleShowOnMap = (provider: HealthcareProvider) => {
    setViewMode('map');
    // Focus map on provider location
    console.log('Show on map:', provider);
  };

  const handleBookAppointment = (provider: HealthcareProvider, type: 'in_person' | 'video') => {
    // Navigate to booking page or open booking modal
    console.log('Book appointment:', provider, type);
    // For now, just show an alert
    alert(`Booking ${type} appointment with ${provider.name}`);
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow-sm sticky top-0 z-30">
        <div className="flex items-center justify-between p-4">
          <div className="flex items-center">
            <button
              onClick={() => navigate('/')}
              className="p-2 -ml-2 rounded-full hover:bg-gray-100 md:hidden"
            >
              <ArrowLeft className="w-6 h-6" />
            </button>
            <h1 className="text-xl font-semibold ml-2">Find Care</h1>
          </div>

          {/* View Toggle */}
          <div className="flex items-center space-x-2">
            <div className="hidden md:flex bg-gray-100 rounded-lg p-1">
              <button
                onClick={() => setViewMode('list')}
                className={`p-2 rounded-md transition-colors ${
                  viewMode === 'list' 
                    ? 'bg-white text-blue-600 shadow-sm' 
                    : 'text-gray-600 hover:text-gray-900'
                }`}
              >
                <List className="w-4 h-4" />
              </button>
              <button
                onClick={() => setViewMode('map')}
                className={`p-2 rounded-md transition-colors ${
                  viewMode === 'map' 
                    ? 'bg-white text-blue-600 shadow-sm' 
                    : 'text-gray-600 hover:text-gray-900'
                }`}
              >
                <Map className="w-4 h-4" />
              </button>
            </div>

            {/* Filter Toggle */}
            <button
              onClick={() => setShowFilters(!showFilters)}
              className="p-2 rounded-full hover:bg-gray-100 md:hidden"
            >
              {showFilters ? <X className="w-5 h-5" /> : <Filter className="w-5 h-5" />}
            </button>
          </div>
        </div>

        {/* Search Bar */}
        <div className="p-4 pt-0">
          <SearchBar onSearch={handleSearch} onLocationSearch={handleLocationSearch} />
        </div>
      </div>

      <div className="flex">
        {/* Filter Sidebar */}
        <FilterSidebar
          filters={filters}
          onFiltersChange={setFilters}
          isOpen={showFilters}
          onClose={() => setShowFilters(false)}
        />

        {/* Main Content */}
        <div className="flex-1 p-4 md:p-6">
          {/* Results Header */}
          <div className="flex items-center justify-between mb-4">
            <div>
              <p className="text-sm text-gray-600">
                {loading ? 'Searching...' : `${filteredProviders.length} providers found`}
              </p>
            </div>
            
            {/* Mobile View Toggle */}
            <div className="md:hidden flex bg-gray-100 rounded-lg p-1">
              <button
                onClick={() => setViewMode('list')}
                className={`p-2 rounded-md transition-colors ${
                  viewMode === 'list' 
                    ? 'bg-white text-blue-600 shadow-sm' 
                    : 'text-gray-600'
                }`}
              >
                <List className="w-4 h-4" />
              </button>
              <button
                onClick={() => setViewMode('map')}
                className={`p-2 rounded-md transition-colors ${
                  viewMode === 'map' 
                    ? 'bg-white text-blue-600 shadow-sm' 
                    : 'text-gray-600'
                }`}
              >
                <Map className="w-4 h-4" />
              </button>
            </div>
          </div>

          {/* Content */}
          {viewMode === 'list' ? (
            <div className="space-y-4">
              {loading ? (
                <div className="flex items-center justify-center py-12">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
                </div>
              ) : filteredProviders.length > 0 ? (
                filteredProviders.map(provider => (
                  <ProviderCard
                    key={provider.id}
                    provider={provider}
                    onShowOnMap={handleShowOnMap}
                    onBookAppointment={handleBookAppointment}
                  />
                ))
              ) : (
                <div className="text-center py-12">
                  <p className="text-gray-500">No providers found matching your criteria.</p>
                  <button
                    onClick={() => setFilters({
                      providerType: [],
                      specialties: [],
                      maxDistance: 25,
                      minRating: 0,
                      languages: [],
                      availability: 'any',
                      insuranceAccepted: [],
                      priceRange: []
                    })}
                    className="mt-2 text-blue-600 hover:text-blue-700"
                  >
                    Clear filters and try again
                  </button>
                </div>
              )}
            </div>
          ) : (
            <GoogleMapView providers={filteredProviders} />
          )}
        </div>
      </div>

      {/* Mobile Bottom Navigation Spacing */}
      <div className="h-20 md:hidden"></div>
    </div>
  );
};

export default SpecialistScreen;
