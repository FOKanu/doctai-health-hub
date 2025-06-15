
import React, { useState, useEffect, useRef } from 'react';
import { Search, X, Clock, TrendingUp, User, Calendar, Pill, FileText } from 'lucide-react';
import { Input } from '@/components/ui/input';

interface SearchResult {
  id: string;
  title: string;
  type: 'doctor' | 'appointment' | 'medication' | 'record' | 'page';
  subtitle?: string;
  icon: React.ElementType;
  url?: string;
}

interface ResponsiveSearchBarProps {
  onSelect?: (result: SearchResult) => void;
  placeholder?: string;
  showRecentSearches?: boolean;
}

const ResponsiveSearchBar: React.FC<ResponsiveSearchBarProps> = ({
  onSelect,
  placeholder = "Search doctors, medications, records...",
  showRecentSearches = true
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [recentSearches, setRecentSearches] = useState<string[]>([
    'Dr. Sarah Wilson',
    'Blood test results',
    'Lisinopril prescription'
  ]);
  
  const searchRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Mock search results
  const searchResults: SearchResult[] = [
    {
      id: '1',
      title: 'Dr. Sarah Wilson',
      type: 'doctor',
      subtitle: 'Primary Care Physician',
      icon: User,
      url: '/specialists'
    },
    {
      id: '2',
      title: 'Blood Test Results',
      type: 'record',
      subtitle: 'January 15, 2024',
      icon: FileText,
      url: '/medical-records'
    },
    {
      id: '3',
      title: 'Lisinopril 10mg',
      type: 'medication',
      subtitle: 'Blood pressure medication',
      icon: Pill,
      url: '/medications'
    },
    {
      id: '4',
      title: 'Cardiology Appointment',
      type: 'appointment',
      subtitle: 'January 20, 2024 at 2:00 PM',
      icon: Calendar,
      url: '/history'
    },
    {
      id: '5',
      title: 'Medical Records',
      type: 'page',
      subtitle: 'View all your medical records',
      icon: FileText,
      url: '/medical-records'
    }
  ];

  const filteredResults = searchTerm.length > 0 
    ? searchResults.filter(result =>
        result.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
        result.subtitle?.toLowerCase().includes(searchTerm.toLowerCase())
      )
    : [];

  const getTypeColor = (type: string) => {
    switch (type) {
      case 'doctor': return 'text-blue-600 bg-blue-50';
      case 'appointment': return 'text-green-600 bg-green-50';
      case 'medication': return 'text-purple-600 bg-purple-50';
      case 'record': return 'text-orange-600 bg-orange-50';
      case 'page': return 'text-gray-600 bg-gray-50';
      default: return 'text-gray-600 bg-gray-50';
    }
  };

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (searchRef.current && !searchRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleSelect = (result: SearchResult) => {
    setSearchTerm(result.title);
    setIsOpen(false);
    
    // Add to recent searches
    setRecentSearches(prev => {
      const filtered = prev.filter(search => search !== result.title);
      return [result.title, ...filtered].slice(0, 5);
    });

    onSelect?.(result);
  };

  const handleRecentSearch = (search: string) => {
    setSearchTerm(search);
    setIsOpen(false);
  };

  const clearSearch = () => {
    setSearchTerm('');
    inputRef.current?.focus();
  };

  return (
    <div ref={searchRef} className="relative w-full max-w-2xl mx-auto">
      <div className="relative">
        <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
        <Input
          ref={inputRef}
          type="text"
          placeholder={placeholder}
          value={searchTerm}
          onChange={(e) => {
            setSearchTerm(e.target.value);
            setIsOpen(true);
          }}
          onFocus={() => setIsOpen(true)}
          className="pl-10 pr-10 py-3 text-base rounded-xl border-2 border-gray-200 focus:border-blue-500"
        />
        {searchTerm && (
          <button
            onClick={clearSearch}
            className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-gray-600"
          >
            <X className="w-5 h-5" />
          </button>
        )}
      </div>

      {isOpen && (
        <div className="absolute top-full left-0 right-0 mt-2 bg-white rounded-xl shadow-lg border border-gray-200 z-50 max-h-96 overflow-y-auto">
          {/* Search Results */}
          {searchTerm.length > 0 && (
            <div className="p-2">
              {filteredResults.length > 0 ? (
                <>
                  <div className="px-3 py-2 text-xs font-semibold text-gray-500 uppercase tracking-wider">
                    Search Results
                  </div>
                  {filteredResults.map((result) => {
                    const Icon = result.icon;
                    return (
                      <button
                        key={result.id}
                        onClick={() => handleSelect(result)}
                        className="w-full flex items-center space-x-3 p-3 hover:bg-gray-50 rounded-lg text-left"
                      >
                        <div className={`p-2 rounded-lg ${getTypeColor(result.type)}`}>
                          <Icon className="w-4 h-4" />
                        </div>
                        <div className="flex-1 min-w-0">
                          <p className="font-medium text-gray-900 truncate">{result.title}</p>
                          {result.subtitle && (
                            <p className="text-sm text-gray-500 truncate">{result.subtitle}</p>
                          )}
                        </div>
                        <span className="text-xs text-gray-400 capitalize">{result.type}</span>
                      </button>
                    );
                  })}
                </>
              ) : (
                <div className="p-4 text-center text-gray-500">
                  <Search className="w-8 h-8 mx-auto mb-2 text-gray-300" />
                  <p>No results found for "{searchTerm}"</p>
                </div>
              )}
            </div>
          )}

          {/* Recent Searches */}
          {searchTerm.length === 0 && showRecentSearches && recentSearches.length > 0 && (
            <div className="p-2">
              <div className="px-3 py-2 text-xs font-semibold text-gray-500 uppercase tracking-wider">
                Recent Searches
              </div>
              {recentSearches.map((search, index) => (
                <button
                  key={index}
                  onClick={() => handleRecentSearch(search)}
                  className="w-full flex items-center space-x-3 p-3 hover:bg-gray-50 rounded-lg text-left"
                >
                  <div className="p-2 rounded-lg bg-gray-50 text-gray-600">
                    <Clock className="w-4 h-4" />
                  </div>
                  <span className="flex-1 text-gray-700">{search}</span>
                </button>
              ))}
            </div>
          )}

          {/* Quick Actions */}
          {searchTerm.length === 0 && (
            <div className="p-2 border-t border-gray-100">
              <div className="px-3 py-2 text-xs font-semibold text-gray-500 uppercase tracking-wider">
                Quick Actions
              </div>
              <div className="grid grid-cols-2 gap-2">
                <button className="flex items-center space-x-2 p-3 hover:bg-gray-50 rounded-lg text-left">
                  <TrendingUp className="w-4 h-4 text-blue-600" />
                  <span className="text-sm text-gray-700">View Analytics</span>
                </button>
                <button className="flex items-center space-x-2 p-3 hover:bg-gray-50 rounded-lg text-left">
                  <Calendar className="w-4 h-4 text-green-600" />
                  <span className="text-sm text-gray-700">Book Appointment</span>
                </button>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default ResponsiveSearchBar;
