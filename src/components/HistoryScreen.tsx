
import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowLeft, Filter, Search, Download, Calendar, Camera, Upload, FileText } from 'lucide-react';

const HistoryScreen = () => {
  const navigate = useNavigate();
  const [filter, setFilter] = useState('all');
  const [searchTerm, setSearchTerm] = useState('');

  const historyItems = [
    {
      id: 1,
      type: 'scan',
      title: 'Skin Lesion Scan',
      date: '2024-06-13',
      time: '14:30',
      result: 'Low Risk',
      resultColor: 'text-green-600',
      icon: Camera,
      notes: 'Benign-appearing mole on left arm'
    },
    {
      id: 2,
      type: 'upload',
      title: 'MRI Brain Scan',
      date: '2024-06-10',
      time: '09:15',
      result: 'Normal',
      resultColor: 'text-green-600',
      icon: Upload,
      notes: 'No abnormalities detected'
    },
    {
      id: 3,
      type: 'diagnosis',
      title: 'Dermatologist Consultation',
      date: '2024-06-08',
      time: '11:00',
      result: 'Follow-up Required',
      resultColor: 'text-orange-600',
      icon: FileText,
      notes: 'Schedule rescan in 3 months'
    },
    {
      id: 4,
      type: 'scan',
      title: 'Skin Lesion Scan',
      date: '2024-06-05',
      time: '16:45',
      result: 'Moderate Risk',
      resultColor: 'text-orange-600',
      icon: Camera,
      notes: 'Irregular borders detected, specialist recommended'
    },
    {
      id: 5,
      type: 'upload',
      title: 'Blood Test Results',
      date: '2024-06-01',
      time: '08:30',
      result: 'Normal',
      resultColor: 'text-green-600',
      icon: Upload,
      notes: 'All markers within normal range'
    }
  ];

  const filterTypes = [
    { key: 'all', label: 'All' },
    { key: 'scan', label: 'Scans' },
    { key: 'upload', label: 'Uploads' },
    { key: 'diagnosis', label: 'Diagnoses' }
  ];

  const filteredItems = historyItems.filter(item => {
    const matchesFilter = filter === 'all' || item.type === filter;
    const matchesSearch = searchTerm === '' || 
      item.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
      item.notes.toLowerCase().includes(searchTerm.toLowerCase());
    return matchesFilter && matchesSearch;
  });

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
            <h1 className="text-xl font-semibold ml-2">Medical History</h1>
          </div>
          <button className="p-2 rounded-full hover:bg-gray-100">
            <Download className="w-5 h-5 text-gray-600" />
          </button>
        </div>

        {/* Search and Filter */}
        <div className="px-4 pb-4 space-y-3">
          <div className="relative">
            <Search className="w-5 h-5 absolute left-3 top-3 text-gray-400" />
            <input
              type="text"
              placeholder="Search history..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
          </div>

          <div className="flex space-x-2">
            {filterTypes.map((type) => (
              <button
                key={type.key}
                onClick={() => setFilter(type.key)}
                className={`px-4 py-2 rounded-full text-sm font-medium transition-colors ${
                  filter === type.key
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                }`}
              >
                {type.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* History List */}
      <div className="p-4">
        {filteredItems.length === 0 ? (
          <div className="text-center py-12">
            <FileText className="w-12 h-12 text-gray-300 mx-auto mb-4" />
            <p className="text-gray-500">No history items found</p>
          </div>
        ) : (
          <div className="space-y-3">
            {filteredItems.map((item) => (
              <div key={item.id} className="bg-white rounded-lg shadow-sm p-4">
                <div className="flex items-start space-x-3">
                  <div className="p-2 bg-blue-100 rounded-lg">
                    <item.icon className="w-5 h-5 text-blue-600" />
                  </div>
                  
                  <div className="flex-1">
                    <div className="flex justify-between items-start mb-2">
                      <h3 className="font-semibold text-gray-800">{item.title}</h3>
                      <span className={`font-medium ${item.resultColor}`}>
                        {item.result}
                      </span>
                    </div>
                    
                    <div className="flex items-center text-sm text-gray-500 mb-2">
                      <Calendar className="w-4 h-4 mr-1" />
                      <span>{item.date} at {item.time}</span>
                    </div>
                    
                    <p className="text-sm text-gray-600">{item.notes}</p>
                    
                    <div className="flex space-x-2 mt-3">
                      <button className="text-sm text-blue-600 hover:underline">
                        View Details
                      </button>
                      <span className="text-gray-300">•</span>
                      <button className="text-sm text-blue-600 hover:underline">
                        Share
                      </button>
                      <span className="text-gray-300">•</span>
                      <button className="text-sm text-blue-600 hover:underline">
                        Download
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Weekly Rescan Log */}
        <div className="mt-6 bg-white rounded-lg shadow-sm p-4">
          <h3 className="font-semibold text-gray-800 mb-3">Weekly Rescan Progress</h3>
          <div className="space-y-2">
            <div className="flex justify-between items-center p-2 bg-green-50 rounded">
              <span className="text-sm">Lesion #3 - Left Arm</span>
              <span className="text-xs text-green-600 font-medium">Completed</span>
            </div>
            <div className="flex justify-between items-center p-2 bg-yellow-50 rounded">
              <span className="text-sm">Lesion #1 - Back</span>
              <span className="text-xs text-yellow-600 font-medium">Due Tomorrow</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default HistoryScreen;
