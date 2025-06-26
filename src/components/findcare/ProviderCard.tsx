
import React from 'react';
import { Star, MapPin, Calendar, Globe, Phone, Video, Clock } from 'lucide-react';
import { HealthcareProvider } from '../../types/healthcare';

interface ProviderCardProps {
  provider: HealthcareProvider;
  onShowOnMap: (provider: HealthcareProvider) => void;
  onBookAppointment: (provider: HealthcareProvider, type: 'in_person' | 'video') => void;
}

const ProviderCard: React.FC<ProviderCardProps> = ({
  provider,
  onShowOnMap,
  onBookAppointment
}) => {
  const getPriceRangeDisplay = (range?: string) => {
    switch (range) {
      case 'low': return '€€';
      case 'medium': return '€€€';
      case 'high': return '€€€€';
      default: return 'Contact for pricing';
    }
  };

  const getTypeDisplay = (type: string) => {
    switch (type) {
      case 'doctor': return 'Doctor';
      case 'dentist': return 'Dentist';
      case 'physiotherapist': return 'Physiotherapist';
      case 'psychotherapist': return 'Psychotherapist';
      case 'hospital': return 'Hospital';
      case 'clinic': return 'Clinic';
      case 'pharmacy': return 'Pharmacy';
      default: return type;
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 hover:shadow-md transition-shadow">
      <div className="flex items-start space-x-4">
        {/* Provider Image */}
        <div className="flex-shrink-0">
          <img
            src={provider.image || `https://images.unsplash.com/photo-1559839734-2b71ea197ec2?w=80&h=80&fit=crop&crop=face`}
            alt={provider.name}
            className="w-16 h-16 rounded-full object-cover"
          />
        </div>

        {/* Provider Info */}
        <div className="flex-1 min-w-0">
          <div className="flex items-start justify-between">
            <div>
              <h3 className="text-lg font-semibold text-gray-900 truncate">
                {provider.name}
              </h3>
              <p className="text-sm text-blue-600 font-medium">
                {provider.specialty ? `${provider.specialty} • ` : ''}{getTypeDisplay(provider.type)}
              </p>
              {(provider.clinicName || provider.hospitalName) && (
                <p className="text-sm text-gray-600 mt-1">
                  {provider.clinicName || provider.hospitalName}
                </p>
              )}
            </div>

            {/* Status Badge */}
            <div className="flex-shrink-0">
              {provider.availability.isOpenNow ? (
                <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800">
                  <div className="w-2 h-2 bg-green-400 rounded-full mr-1"></div>
                  Open Now
                </span>
              ) : (
                <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-gray-100 text-gray-800">
                  Closed
                </span>
              )}
            </div>
          </div>

          {/* Rating and Experience */}
          <div className="flex items-center space-x-4 mt-2">
            <div className="flex items-center">
              <Star className="w-4 h-4 text-yellow-400 fill-current" />
              <span className="text-sm font-medium ml-1">{provider.rating}</span>
              <span className="text-xs text-gray-500 ml-1">({provider.reviewCount})</span>
            </div>
            <span className="text-sm text-gray-600">
              {provider.yearsExperience} years experience
            </span>
          </div>

          {/* Location and Distance */}
          <div className="flex items-center mt-2 text-sm text-gray-600">
            <MapPin className="w-4 h-4 mr-1" />
            <span className="truncate">{provider.location.address}</span>
            {provider.location.distance && (
              <span className="ml-2 text-blue-600 font-medium">
                {provider.location.distance.toFixed(1)} km
              </span>
            )}
          </div>

          {/* Next Available */}
          <div className="flex items-center mt-2 text-sm text-gray-600">
            <Calendar className="w-4 h-4 mr-1" />
            <span>Next: {provider.availability.nextAvailable}</span>
          </div>

          {/* Languages */}
          <div className="flex items-center mt-2 text-sm text-gray-600">
            <Globe className="w-4 h-4 mr-1" />
            <span>Languages: {provider.languages.join(', ')}</span>
          </div>

          {/* Price Range */}
          {provider.priceRange && (
            <div className="mt-2 text-sm text-gray-600">
              Price: {getPriceRangeDisplay(provider.priceRange)}
            </div>
          )}

          {/* Services */}
          {provider.services.length > 0 && (
            <div className="mt-3">
              <div className="flex flex-wrap gap-1">
                {provider.services.slice(0, 3).map((service, index) => (
                  <span
                    key={index}
                    className="inline-flex items-center px-2 py-1 rounded-md text-xs font-medium bg-blue-100 text-blue-800"
                  >
                    {service}
                  </span>
                ))}
                {provider.services.length > 3 && (
                  <span className="text-xs text-gray-500">+{provider.services.length - 3} more</span>
                )}
              </div>
            </div>
          )}

          {/* Action Buttons */}
          <div className="flex items-center justify-between mt-4 pt-4 border-t border-gray-100">
            <button
              onClick={() => onShowOnMap(provider)}
              className="text-blue-600 hover:text-blue-700 font-medium text-sm"
            >
              Show on Map
            </button>

            <div className="flex space-x-2">
              {provider.type !== 'pharmacy' && provider.type !== 'hospital' && (
                <>
                  <button
                    onClick={() => onBookAppointment(provider, 'video')}
                    className="inline-flex items-center px-3 py-1 border border-gray-300 rounded-md text-sm font-medium text-gray-700 bg-white hover:bg-gray-50"
                  >
                    <Video className="w-4 h-4 mr-1" />
                    Video
                  </button>
                  <button
                    onClick={() => onBookAppointment(provider, 'in_person')}
                    className="inline-flex items-center px-3 py-1 border border-transparent rounded-md text-sm font-medium text-white bg-blue-600 hover:bg-blue-700"
                  >
                    <Clock className="w-4 h-4 mr-1" />
                    Book
                  </button>
                </>
              )}
              {(provider.type === 'pharmacy' || provider.type === 'hospital') && (
                <div className="flex items-center text-sm text-gray-600">
                  <Phone className="w-4 h-4 mr-1" />
                  <span>{provider.contactInfo.phone}</span>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ProviderCard;
