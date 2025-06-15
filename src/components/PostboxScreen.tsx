
import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowLeft, Mail, Search, Filter, Archive, Star, Trash2, Download, Eye, Calendar } from 'lucide-react';
import { Input } from '@/components/ui/input';

const PostboxScreen = () => {
  const navigate = useNavigate();
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedFilter, setSelectedFilter] = useState('all');

  const messages = [
    {
      id: 1,
      sender: 'Dr. Sarah Wilson',
      subject: 'Lab Results - Blood Work Complete',
      preview: 'Your recent blood test results are now available. All values appear normal with...',
      date: '2024-01-15',
      time: '14:30',
      isRead: false,
      isStarred: true,
      type: 'lab-results',
      attachments: 2
    },
    {
      id: 2,
      sender: 'City General Hospital',
      subject: 'Appointment Confirmation - Cardiology',
      preview: 'Your appointment with Dr. Martinez has been confirmed for January 20th at...',
      date: '2024-01-14',
      time: '09:15',
      isRead: true,
      isStarred: false,
      type: 'appointment',
      attachments: 1
    },
    {
      id: 3,
      sender: 'Pharmacy Plus',
      subject: 'Prescription Ready for Pickup',
      preview: 'Your prescription for Lisinopril is ready for pickup at our main location...',
      date: '2024-01-13',
      time: '16:45',
      isRead: true,
      isStarred: false,
      type: 'prescription',
      attachments: 0
    },
    {
      id: 4,
      sender: 'Insurance Provider',
      subject: 'Claim Status Update',
      preview: 'Your recent claim #CLM-2024-001 has been processed and approved...',
      date: '2024-01-12',
      time: '11:20',
      isRead: false,
      isStarred: false,
      type: 'insurance',
      attachments: 3
    },
    {
      id: 5,
      sender: 'Dr. Michael Chen',
      subject: 'Follow-up Instructions',
      preview: 'Please review the attached follow-up care instructions and schedule...',
      date: '2024-01-11',
      time: '13:00',
      isRead: true,
      isStarred: true,
      type: 'follow-up',
      attachments: 1
    }
  ];

  const getTypeColor = (type: string) => {
    switch (type) {
      case 'lab-results': return 'bg-blue-100 text-blue-800';
      case 'appointment': return 'bg-green-100 text-green-800';
      case 'prescription': return 'bg-purple-100 text-purple-800';
      case 'insurance': return 'bg-orange-100 text-orange-800';
      case 'follow-up': return 'bg-gray-100 text-gray-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const filters = [
    { key: 'all', label: 'All Messages', count: messages.length },
    { key: 'unread', label: 'Unread', count: messages.filter(m => !m.isRead).length },
    { key: 'starred', label: 'Starred', count: messages.filter(m => m.isStarred).length },
    { key: 'lab-results', label: 'Lab Results', count: messages.filter(m => m.type === 'lab-results').length }
  ];

  const filteredMessages = messages.filter(message => {
    const matchesSearch = message.sender.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         message.subject.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesFilter = selectedFilter === 'all' ||
                         (selectedFilter === 'unread' && !message.isRead) ||
                         (selectedFilter === 'starred' && message.isStarred) ||
                         message.type === selectedFilter;
    return matchesSearch && matchesFilter;
  });

  return (
    <div className="min-h-screen bg-gray-50 pb-20">
      {/* Header */}
      <div className="bg-white shadow-sm">
        <div className="flex items-center p-4">
          <button
            onClick={() => navigate('/')}
            className="p-2 -ml-2 rounded-full hover:bg-gray-100"
          >
            <ArrowLeft className="w-6 h-6" />
          </button>
          <h1 className="text-xl font-semibold ml-2">Medical Postbox</h1>
        </div>
      </div>

      <div className="p-4">
        {/* Search Bar */}
        <div className="relative mb-4">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
          <Input
            type="text"
            placeholder="Search messages..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="pl-10"
          />
        </div>

        {/* Filter Tabs */}
        <div className="flex space-x-2 mb-6 overflow-x-auto">
          {filters.map((filter) => (
            <button
              key={filter.key}
              onClick={() => setSelectedFilter(filter.key)}
              className={`px-4 py-2 rounded-full text-sm font-medium whitespace-nowrap ${
                selectedFilter === filter.key
                  ? 'bg-blue-600 text-white'
                  : 'bg-white text-gray-600 hover:bg-gray-50'
              }`}
            >
              {filter.label} ({filter.count})
            </button>
          ))}
        </div>

        {/* Messages List */}
        <div className="space-y-3">
          {filteredMessages.map((message) => (
            <div
              key={message.id}
              className={`bg-white rounded-lg p-4 shadow-sm border ${
                !message.isRead ? 'border-l-4 border-l-blue-600' : 'border-gray-200'
              }`}
            >
              <div className="flex items-start justify-between mb-2">
                <div className="flex-1">
                  <div className="flex items-center space-x-2 mb-1">
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${getTypeColor(message.type)}`}>
                      {message.type.replace('-', ' ')}
                    </span>
                    {message.isStarred && <Star className="w-4 h-4 text-yellow-500 fill-current" />}
                    {!message.isRead && <div className="w-2 h-2 bg-blue-600 rounded-full" />}
                  </div>
                  <h3 className={`text-lg font-semibold ${!message.isRead ? 'text-gray-900' : 'text-gray-700'}`}>
                    {message.sender}
                  </h3>
                  <p className={`font-medium ${!message.isRead ? 'text-gray-900' : 'text-gray-600'}`}>
                    {message.subject}
                  </p>
                  <p className="text-gray-500 text-sm mt-1">{message.preview}</p>
                </div>
                
                <div className="flex flex-col items-end space-y-2 ml-4">
                  <div className="flex items-center space-x-1 text-xs text-gray-500">
                    <Calendar className="w-3 h-3" />
                    <span>{message.date}</span>
                  </div>
                  <span className="text-xs text-gray-500">{message.time}</span>
                  {message.attachments > 0 && (
                    <span className="text-xs text-blue-600 bg-blue-50 px-2 py-1 rounded">
                      {message.attachments} attachment{message.attachments > 1 ? 's' : ''}
                    </span>
                  )}
                </div>
              </div>

              <div className="flex items-center justify-between pt-3 border-t border-gray-100">
                <div className="flex space-x-3">
                  <button className="p-2 text-gray-400 hover:text-blue-600 hover:bg-blue-50 rounded-full">
                    <Eye className="w-4 h-4" />
                  </button>
                  <button className="p-2 text-gray-400 hover:text-yellow-600 hover:bg-yellow-50 rounded-full">
                    <Star className="w-4 h-4" />
                  </button>
                  <button className="p-2 text-gray-400 hover:text-green-600 hover:bg-green-50 rounded-full">
                    <Download className="w-4 h-4" />
                  </button>
                </div>
                
                <div className="flex space-x-2">
                  <button className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-full">
                    <Archive className="w-4 h-4" />
                  </button>
                  <button className="p-2 text-gray-400 hover:text-red-600 hover:bg-red-50 rounded-full">
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>

        {filteredMessages.length === 0 && (
          <div className="text-center py-12">
            <Mail className="w-16 h-16 text-gray-300 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">No messages found</h3>
            <p className="text-gray-500">Try adjusting your search or filter criteria.</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default PostboxScreen;
