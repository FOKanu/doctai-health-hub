
import React from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowLeft, Star, MapPin, Calendar, Share } from 'lucide-react';

const SpecialistScreen = () => {
  const navigate = useNavigate();

  const specialists = [
    {
      id: 1,
      name: 'Dr. Sarah Weber',
      specialty: 'Dermatology',
      rating: 4.9,
      reviews: 127,
      location: 'Berlin, Germany',
      distance: '2.3 km',
      nextAvailable: 'Tomorrow, 14:30',
      experience: '15 years',
      languages: ['German', 'English'],
      image: 'https://images.unsplash.com/photo-1559839734-2b71ea197ec2?w=150&h=150&fit=crop&crop=face'
    },
    {
      id: 2,
      name: 'Prof. Dr. Michael Braun',
      specialty: 'Oncology',
      rating: 4.8,
      reviews: 203,
      location: 'Munich, Germany',
      distance: '5.1 km',
      nextAvailable: 'Friday, 10:00',
      experience: '22 years',
      languages: ['German', 'English', 'French'],
      image: 'https://images.unsplash.com/photo-1612349317150-e413f6a5b16d?w=150&h=150&fit=crop&crop=face'
    },
    {
      id: 3,
      name: 'Dr. Anna Müller',
      specialty: 'Dermatology',
      rating: 4.7,
      reviews: 89,
      location: 'Hamburg, Germany',
      distance: '3.7 km',
      nextAvailable: 'Monday, 16:45',
      experience: '12 years',
      languages: ['German', 'English'],
      image: 'https://images.unsplash.com/photo-1594824388317-fda66db2122d?w=150&h=150&fit=crop&crop=face'
    }
  ];

  const riskSummary = {
    level: 'Moderate',
    scanType: 'Skin Lesion',
    date: 'Today',
    recommendation: 'Dermatologist consultation recommended within 2 weeks'
  };

  return (
    <div className="min-h-screen bg-gray-50 pb-20">
      {/* Header */}
      <div className="bg-white shadow-sm">
        <div className="flex items-center justify-between p-4">
          <div className="flex items-center">
            <button
              onClick={() => navigate('/')}
              className="p-2 -ml-2 rounded-full hover:bg-gray-100"
            >
              <ArrowLeft className="w-6 h-6" />
            </button>
            <h1 className="text-xl font-semibold ml-2">Specialist Recommendations</h1>
          </div>
          <button className="p-2 rounded-full hover:bg-gray-100">
            <Share className="w-5 h-5 text-gray-600" />
          </button>
        </div>
      </div>

      <div className="p-4 space-y-4">
        {/* Risk Summary */}
        <div className="bg-orange-50 border-l-4 border-orange-400 p-4 rounded-r-lg">
          <h2 className="font-semibold text-orange-800 mb-2">Risk Summary</h2>
          <div className="text-sm text-orange-700 space-y-1">
            <p><span className="font-medium">Risk Level:</span> {riskSummary.level}</p>
            <p><span className="font-medium">Scan Type:</span> {riskSummary.scanType}</p>
            <p><span className="font-medium">Date:</span> {riskSummary.date}</p>
            <p className="mt-2 italic">{riskSummary.recommendation}</p>
          </div>
        </div>

        {/* Specialists List */}
        <div className="space-y-4">
          <h2 className="text-lg font-semibold text-gray-800">Recommended Specialists</h2>
          
          {specialists.map((specialist) => (
            <div key={specialist.id} className="bg-white rounded-lg shadow-sm p-4">
              <div className="flex items-start space-x-4">
                <img
                  src={specialist.image}
                  alt={specialist.name}
                  className="w-16 h-16 rounded-full object-cover"
                />
                
                <div className="flex-1">
                  <div className="flex justify-between items-start mb-2">
                    <div>
                      <h3 className="font-semibold text-gray-800">{specialist.name}</h3>
                      <p className="text-sm text-blue-600">{specialist.specialty}</p>
                    </div>
                    <div className="flex items-center">
                      <Star className="w-4 h-4 text-yellow-400 fill-current" />
                      <span className="text-sm font-medium ml-1">{specialist.rating}</span>
                      <span className="text-xs text-gray-500 ml-1">({specialist.reviews})</span>
                    </div>
                  </div>

                  <div className="text-sm text-gray-600 space-y-1 mb-3">
                    <div className="flex items-center">
                      <MapPin className="w-3 h-3 mr-1" />
                      <span>{specialist.location} • {specialist.distance}</span>
                    </div>
                    <div className="flex items-center">
                      <Calendar className="w-3 h-3 mr-1" />
                      <span>Next: {specialist.nextAvailable}</span>
                    </div>
                    <div>
                      <span className="font-medium">{specialist.experience} experience</span>
                    </div>
                    <div>
                      <span>Languages: {specialist.languages.join(', ')}</span>
                    </div>
                  </div>

                  <button className="w-full bg-blue-600 text-white py-2 rounded-lg font-medium hover:bg-blue-700 transition-colors">
                    Book Now
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Alternative Options */}
        <div className="bg-blue-50 rounded-lg p-4">
          <h3 className="font-semibold text-blue-800 mb-2">Alternative Options</h3>
          <div className="space-y-2 text-sm text-blue-700">
            <button className="block w-full text-left hover:underline">
              • Search more specialists in your area
            </button>
            <button className="block w-full text-left hover:underline">
              • Schedule a telemedicine consultation
            </button>
            <button className="block w-full text-left hover:underline">
              • Get a second opinion
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SpecialistScreen;
