
import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowLeft, Search, Filter, Download, Eye, Calendar, FileText, Image, Activity, Pill, Users, Shield, Plus } from 'lucide-react';
import { Input } from '@/components/ui/input';

const MedicalRecordsScreen = () => {
  const navigate = useNavigate();
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('all');

  const records = [
    {
      id: 1,
      title: 'Complete Blood Count (CBC)',
      category: 'lab-results',
      date: '2024-01-15',
      doctor: 'Dr. Sarah Wilson',
      facility: 'City Medical Center',
      description: 'Routine blood work showing all values within normal range',
      fileType: 'PDF',
      fileSize: '2.3 MB',
      status: 'completed',
      tags: ['routine', 'blood-work', 'normal']
    },
    {
      id: 2,
      title: 'Chest X-Ray',
      category: 'imaging',
      date: '2024-01-10',
      doctor: 'Dr. Michael Chen',
      facility: 'Radiology Associates',
      description: 'Chest X-ray examination - clear lung fields',
      fileType: 'DICOM',
      fileSize: '15.7 MB',
      status: 'completed',
      tags: ['chest', 'x-ray', 'clear']
    },
    {
      id: 3,
      title: 'Cardiology Consultation',
      category: 'consultations',
      date: '2024-01-08',
      doctor: 'Dr. Elena Martinez',
      facility: 'Heart Specialists Clinic',
      description: 'Routine cardiac evaluation and ECG interpretation',
      fileType: 'PDF',
      fileSize: '1.8 MB',
      status: 'completed',
      tags: ['cardiology', 'ecg', 'routine']
    },
    {
      id: 4,
      title: 'Prescription Record - Lisinopril',
      category: 'prescriptions',
      date: '2024-01-05',
      doctor: 'Dr. Sarah Wilson',
      facility: 'City Medical Center',
      description: 'Blood pressure medication - 10mg daily',
      fileType: 'PDF',
      fileSize: '0.5 MB',
      status: 'active',
      tags: ['blood-pressure', 'daily', 'active']
    },
    {
      id: 5,
      title: 'Annual Physical Examination',
      category: 'examinations',
      date: '2024-01-03',
      doctor: 'Dr. Sarah Wilson',
      facility: 'City Medical Center',
      description: 'Comprehensive annual health checkup and assessment',
      fileType: 'PDF',
      fileSize: '4.2 MB',
      status: 'completed',
      tags: ['annual', 'physical', 'comprehensive']
    }
  ];

  const categories = [
    { key: 'all', label: 'All Records', icon: FileText, count: records.length },
    { key: 'lab-results', label: 'Lab Results', icon: Activity, count: records.filter(r => r.category === 'lab-results').length },
    { key: 'imaging', label: 'Imaging', icon: Image, count: records.filter(r => r.category === 'imaging').length },
    { key: 'prescriptions', label: 'Prescriptions', icon: Pill, count: records.filter(r => r.category === 'prescriptions').length },
    { key: 'consultations', label: 'Consultations', icon: Users, count: records.filter(r => r.category === 'consultations').length },
    { key: 'examinations', label: 'Examinations', icon: Shield, count: records.filter(r => r.category === 'examinations').length }
  ];

  const getCategoryColor = (category: string) => {
    switch (category) {
      case 'lab-results': return 'bg-blue-100 text-blue-800';
      case 'imaging': return 'bg-purple-100 text-purple-800';
      case 'prescriptions': return 'bg-green-100 text-green-800';
      case 'consultations': return 'bg-orange-100 text-orange-800';
      case 'examinations': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'bg-green-100 text-green-800';
      case 'active': return 'bg-blue-100 text-blue-800';
      case 'pending': return 'bg-yellow-100 text-yellow-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const filteredRecords = records.filter(record => {
    const matchesSearch = record.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         record.doctor.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         record.tags.some(tag => tag.toLowerCase().includes(searchTerm.toLowerCase()));
    const matchesCategory = selectedCategory === 'all' || record.category === selectedCategory;
    return matchesSearch && matchesCategory;
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
            <h1 className="text-xl font-semibold ml-2">Medical Records</h1>
          </div>
          <button className="p-2 text-blue-600 hover:bg-blue-50 rounded-full">
            <Plus className="w-6 h-6" />
          </button>
        </div>
      </div>

      <div className="p-4">
        {/* Search Bar */}
        <div className="relative mb-4">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
          <Input
            type="text"
            placeholder="Search records, doctors, or tags..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="pl-10"
          />
        </div>

        {/* Category Filter */}
        <div className="grid grid-cols-2 md:grid-cols-3 gap-2 mb-6">
          {categories.map((category) => {
            const Icon = category.icon;
            return (
              <button
                key={category.key}
                onClick={() => setSelectedCategory(category.key)}
                className={`p-3 rounded-lg text-left transition-colors ${
                  selectedCategory === category.key
                    ? 'bg-blue-600 text-white'
                    : 'bg-white text-gray-600 hover:bg-gray-50'
                }`}
              >
                <div className="flex items-center space-x-2 mb-1">
                  <Icon className="w-4 h-4" />
                  <span className="font-medium text-sm">{category.label}</span>
                </div>
                <span className="text-xs opacity-75">{category.count} records</span>
              </button>
            );
          })}
        </div>

        {/* Records List */}
        <div className="space-y-4">
          {filteredRecords.map((record) => (
            <div key={record.id} className="bg-white rounded-lg p-4 shadow-sm border border-gray-200">
              <div className="flex items-start justify-between mb-3">
                <div className="flex-1">
                  <div className="flex items-center space-x-2 mb-2">
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${getCategoryColor(record.category)}`}>
                      {record.category.replace('-', ' ')}
                    </span>
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(record.status)}`}>
                      {record.status}
                    </span>
                  </div>
                  <h3 className="text-lg font-semibold text-gray-900 mb-1">{record.title}</h3>
                  <p className="text-gray-600 text-sm mb-2">{record.description}</p>
                  
                  <div className="flex items-center space-x-4 text-xs text-gray-500">
                    <div className="flex items-center space-x-1">
                      <Calendar className="w-3 h-3" />
                      <span>{record.date}</span>
                    </div>
                    <span>•</span>
                    <span>{record.doctor}</span>
                    <span>•</span>
                    <span>{record.facility}</span>
                  </div>
                </div>
                
                <div className="text-right ml-4">
                  <p className="text-xs text-gray-500 mb-1">{record.fileType}</p>
                  <p className="text-xs text-gray-500">{record.fileSize}</p>
                </div>
              </div>

              {/* Tags */}
              <div className="flex flex-wrap gap-1 mb-3">
                {record.tags.map((tag, index) => (
                  <span key={index} className="px-2 py-1 bg-gray-100 text-gray-600 text-xs rounded">
                    {tag}
                  </span>
                ))}
              </div>

              {/* Actions */}
              <div className="flex items-center justify-between pt-3 border-t border-gray-100">
                <div className="flex space-x-3">
                  <button className="flex items-center space-x-1 px-3 py-1 text-blue-600 bg-blue-50 rounded-full text-xs hover:bg-blue-100">
                    <Eye className="w-3 h-3" />
                    <span>View</span>
                  </button>
                  <button className="flex items-center space-x-1 px-3 py-1 text-green-600 bg-green-50 rounded-full text-xs hover:bg-green-100">
                    <Download className="w-3 h-3" />
                    <span>Download</span>
                  </button>
                </div>
                
                <button className="p-1 text-gray-400 hover:text-gray-600 rounded">
                  <Filter className="w-4 h-4" />
                </button>
              </div>
            </div>
          ))}
        </div>

        {filteredRecords.length === 0 && (
          <div className="text-center py-12">
            <FileText className="w-16 h-16 text-gray-300 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">No records found</h3>
            <p className="text-gray-500">Try adjusting your search or filter criteria.</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default MedicalRecordsScreen;
