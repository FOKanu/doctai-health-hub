
import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Calendar, Filter, Search, ArrowUpDown, Eye } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Badge } from '@/components/ui/badge';
import { getScans, ScanRecord } from '@/services/scanService';

const TotalScansScreen = () => {
  const navigate = useNavigate();
  const [scans, setScans] = useState<ScanRecord[]>([]);
  const [filteredScans, setFilteredScans] = useState<ScanRecord[]>([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [sortBy, setSortBy] = useState<'date' | 'confidence' | 'riskLevel'>('date');
  const [filterBy, setFilterBy] = useState<'all' | 'low' | 'medium' | 'high'>('all');
  const [bodyPartFilter, setBodyPartFilter] = useState<string>('all');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const loadScans = async () => {
      try {
        const scansData = await getScans();
        setScans(scansData);
        setFilteredScans(scansData);
      } catch (error) {
        console.error('Error loading scans:', error);
      } finally {
        setLoading(false);
      }
    };
    loadScans();
  }, []);

  useEffect(() => {
    const filtered = scans.filter(scan => {
      const matchesSearch = scan.bodyPart.toLowerCase().includes(searchTerm.toLowerCase()) ||
                           scan.prediction.toLowerCase().includes(searchTerm.toLowerCase());
      const matchesRisk = filterBy === 'all' || scan.riskLevel === filterBy;
      const matchesBodyPart = bodyPartFilter === 'all' || scan.bodyPart === bodyPartFilter;

      return matchesSearch && matchesRisk && matchesBodyPart;
    });

    // Sort the filtered results
    filtered.sort((a, b) => {
      switch (sortBy) {
        case 'date':
          return new Date(b.date).getTime() - new Date(a.date).getTime();
        case 'confidence':
          return b.confidence - a.confidence;
        case 'riskLevel': {
          const riskOrder = { high: 3, medium: 2, low: 1 };
          return riskOrder[b.riskLevel] - riskOrder[a.riskLevel];
        }
        default:
          return 0;
      }
    });

    setFilteredScans(filtered);
  }, [scans, searchTerm, sortBy, filterBy, bodyPartFilter]);

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case 'high': return 'bg-red-100 text-red-800 border-red-200';
      case 'medium': return 'bg-yellow-100 text-yellow-800 border-yellow-200';
      case 'low': return 'bg-green-100 text-green-800 border-green-200';
      default: return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };

  const uniqueBodyParts = [...new Set(scans.map(scan => scan.bodyPart))];

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-96">
        <div className="text-gray-500">Loading scans...</div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Total Scans</h1>
          <p className="text-gray-600">Comprehensive overview of all your medical scans</p>
        </div>
        <Button onClick={() => navigate('/')} variant="outline">
          Back to Dashboard
        </Button>
      </div>

      {/* Stats Overview */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="text-2xl font-bold text-blue-600">{scans.length}</div>
            <div className="text-sm text-gray-600">Total Scans</div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4">
            <div className="text-2xl font-bold text-red-600">
              {scans.filter(s => s.riskLevel === 'high').length}
            </div>
            <div className="text-sm text-gray-600">High Risk</div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4">
            <div className="text-2xl font-bold text-yellow-600">
              {scans.filter(s => s.riskLevel === 'medium').length}
            </div>
            <div className="text-sm text-gray-600">Medium Risk</div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4">
            <div className="text-2xl font-bold text-green-600">
              {scans.filter(s => s.riskLevel === 'low').length}
            </div>
            <div className="text-sm text-gray-600">Low Risk</div>
          </CardContent>
        </Card>
      </div>

      {/* Filters and Search */}
      <Card>
        <CardContent className="p-4">
          <div className="flex flex-col lg:flex-row gap-4">
            <div className="flex-1">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
                <Input
                  placeholder="Search by body part or prediction..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="pl-10"
                />
              </div>
            </div>
            <div className="flex gap-2">
              <Select value={sortBy} onValueChange={(value: React.SyntheticEvent) => setSortBy(value)}>
                <SelectTrigger className="w-40">
                  <ArrowUpDown className="w-4 h-4 mr-2" />
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="date">Sort by Date</SelectItem>
                  <SelectItem value="confidence">Sort by Confidence</SelectItem>
                  <SelectItem value="riskLevel">Sort by Risk</SelectItem>
                </SelectContent>
              </Select>
              <Select value={filterBy} onValueChange={(value: React.SyntheticEvent) => setFilterBy(value)}>
                <SelectTrigger className="w-32">
                  <Filter className="w-4 h-4 mr-2" />
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Risk</SelectItem>
                  <SelectItem value="high">High Risk</SelectItem>
                  <SelectItem value="medium">Medium Risk</SelectItem>
                  <SelectItem value="low">Low Risk</SelectItem>
                </SelectContent>
              </Select>
              <Select value={bodyPartFilter} onValueChange={setBodyPartFilter}>
                <SelectTrigger className="w-40">
                  <SelectValue placeholder="Body Part" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Body Parts</SelectItem>
                  {uniqueBodyParts.map(part => (
                    <SelectItem key={part} value={part}>{part}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Scans List */}
      <div className="space-y-4">
        {filteredScans.length === 0 ? (
          <Card>
            <CardContent className="p-8 text-center">
              <div className="text-gray-500">No scans found matching your criteria</div>
            </CardContent>
          </Card>
        ) : (
          filteredScans.map((scan) => (
            <Card key={scan.id} className="hover:shadow-md transition-shadow">
              <CardContent className="p-6">
                <div className="flex items-center gap-4">
                  {/* Image Thumbnail */}
                  <div className="w-16 h-16 bg-gray-100 rounded-lg flex items-center justify-center overflow-hidden">
                    {scan.imageUrl ? (
                      <img
                        src={scan.imageUrl}
                        alt="Scan"
                        className="w-full h-full object-cover"
                      />
                    ) : (
                      <div className="text-gray-400 text-xs">No Image</div>
                    )}
                  </div>

                  {/* Scan Details */}
                  <div className="flex-1">
                    <div className="flex items-start justify-between">
                      <div>
                        <h3 className="font-semibold text-gray-900">{scan.bodyPart}</h3>
                        <p className="text-sm text-gray-600">{scan.prediction}</p>
                        <div className="flex items-center gap-2 mt-1">
                          <Calendar className="w-3 h-3 text-gray-400" />
                          <span className="text-xs text-gray-500">
                            {new Date(scan.date).toLocaleDateString()} at {scan.timestamp}
                          </span>
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        <Badge className={getRiskColor(scan.riskLevel)}>
                          {scan.riskLevel.toUpperCase()} RISK
                        </Badge>
                        <div className="text-right">
                          <div className="text-sm font-semibold text-gray-900">
                            {scan.confidence}%
                          </div>
                          <div className="text-xs text-gray-500">Confidence</div>
                        </div>
                      </div>
                    </div>
                    {scan.notes && (
                      <p className="text-sm text-gray-600 mt-2">{scan.notes}</p>
                    )}
                    {scan.followUpRequired && (
                      <div className="mt-2">
                        <Badge variant="outline" className="text-orange-600 border-orange-200">
                          Follow-up Required: {scan.followUpDate}
                        </Badge>
                      </div>
                    )}
                  </div>
                </div>
              </CardContent>
            </Card>
          ))
        )}
      </div>
    </div>
  );
};

export default TotalScansScreen;
