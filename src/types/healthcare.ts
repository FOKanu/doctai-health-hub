
export interface HealthcareProvider {
  id: string;
  name: string;
  type: 'doctor' | 'dentist' | 'physiotherapist' | 'psychotherapist' | 'hospital' | 'clinic' | 'pharmacy';
  specialty?: string;
  clinicName?: string;
  hospitalName?: string;
  yearsExperience: number;
  languages: string[];
  rating: number;
  reviewCount: number;
  priceRange?: 'low' | 'medium' | 'high';
  insuranceAccepted?: string[];
  location: {
    address: string;
    lat: number;
    lng: number;
    distance?: number;
  };
  availability: {
    nextAvailable: string;
    isOpenNow: boolean;
    hours: string;
  };
  contactInfo: {
    phone: string;
    email?: string;
    website?: string;
  };
  services: string[];
  image?: string;
}

export interface FilterOptions {
  providerType: string[];
  specialties: string[];
  maxDistance: number;
  minRating: number;
  languages: string[];
  availability: 'any' | 'today' | 'this_week';
  insuranceAccepted: string[];
  priceRange: string[];
}

export interface BookingRequest {
  providerId: string;
  appointmentType: 'in_person' | 'video';
  preferredDate: string;
  preferredTime: string;
  reason: string;
  notes?: string;
}
