
import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { AlertTriangle, Calendar, Filter, TrendingUp } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { getScansByRiskLevel, ScanRecord } from '@/services/scanService';
import { getRiskIndicators, RiskIndicator } from '@/services/riskService';

const RiskAssessmentsScreen = () => {
  const navigate = useNavigate();
  const [riskScans, setRiskScans] = useState<ScanRecord[]>([]);
  const [riskIndicators, setRiskIndicators] = useState<RiskIndicator[]>([]);
  const [loading, setLoading] = useState(true);
  const [sortBy, setSortBy] = useState<'date' | 'risk' | 'confidence'>('risk');
  const [riskFilter, setRiskFilter] = useState<'all' | 'high' | 'medium'>('all');

  useEffect(() => {
    const loadRiskData = async () => {
      try {
        // Load high and medium risk scans
        const [highRiskScans, mediumRiskScans, indicators] = await Promise.all([
          getScansByRiskLevel('high'),
          getScansByRiskLevel('medium'),
          getRiskIndicators()
        ]);
        
        const combinedScans = [...highRiskScans, ...mediumRiskScans];
        setRiskScans(combinedScans);
        setRiskIndicators(indicators.filter(i => i.severity === 'high' || i.severity === 'medium'));
      } catch (error) {
        console.error('Error loading risk data:', error);
      } finally {
        setLoading(false);
      }
    };
    loadRiskData();
  }, []);

  const filteredScans = riskScans
    .filter(scan => riskFilter === 'all' || scan.riskLevel === riskFilter)
    .sort((a, b) => {
      switch (sortBy) {
        case 'date':
          return new Date(b.date).getTime() - new Date(a.date).getTime();
        case 'confidence':
          return b.confidence - a.confidence;
        case 'risk':
          const riskOrder = { high: 3, medium: 2, low: 1 };
          return riskOrder[b.riskLevel] - riskOrder[a.riskLevel];
        default:
          return 0;
      }
    });

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case 'high': return 'bg-red-100 text-red-800 border-red-200';
      case 'medium': return 'bg-yellow-100 text-yellow-800 border-yellow-200';
      default: return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };

  const getRiskIcon = (risk: string) => {
    return risk === 'high' ? 
      <AlertTriangle className="w-4 h-4 text-red-600" /> : 
      <TrendingUp className="w-4 h-4 text-yellow-600" />;
  };

  const getRecommendation = (scan: ScanRecord) => {
    if (scan.riskLevel === 'high') {
      return 'Immediate dermatologist consultation required';
    } else if (scan.riskLevel === 'medium') {
      return 'Consider specialist consultation within 2-4 weeks';
    }
    return 'Continue monitoring';
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-96">
        <div className="text-gray-500">Loading risk assessments...</div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Risk Assessments</h1>
          <p className="text-gray-600">Monitor and manage your health risk indicators</p>
        </div>
        <Button onClick={() => navigate('/')} variant="outline">
          Back to Dashboard
        </Button>
      </div>

      {/* Risk Overview */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card className="border-red-200 bg-red-50">
          <CardContent className="p-4">
            <div className="flex items-center gap-2">
              <AlertTriangle className="w-5 h-5 text-red-600" />
              <div>
                <div className="text-2xl font-bold text-red-600">
                  {riskScans.filter(s => s.riskLevel === 'high').length}
                </div>
                <div className="text-sm text-red-700">High Risk Items</div>
              </div>
            </div>
          </CardContent>
        </Card>
        <Card className="border-yellow-200 bg-yellow-50">
          <CardContent className="p-4">
            <div className="flex items-center gap-2">
              <TrendingUp className="w-5 h-5 text-yellow-600" />
              <div>
                <div className="text-2xl font-bold text-yellow-600">
                  {riskScans.filter(s => s.riskLevel === 'medium').length}
                </div>
                <div className="text-sm text-yellow-700">Medium Risk Items</div>
              </div>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4">
            <div className="text-2xl font-bold text-gray-900">
              {riskScans.filter(s => s.followUpRequired).length}
            </div>
            <div className="text-sm text-gray-600">Require Follow-up</div>
          </CardContent>
        </Card>
      </div>

      <Tabs defaultValue="scans" className="space-y-4">
        <TabsList>
          <TabsTrigger value="scans">Risk Scans</TabsTrigger>
          <TabsTrigger value="indicators">Risk Indicators</TabsTrigger>
        </TabsList>

        <TabsContent value="scans" className="space-y-4">
          {/* Filters */}
          <Card>
            <CardContent className="p-4">
              <div className="flex gap-4">
                <Select value={sortBy} onValueChange={(value: any) => setSortBy(value)}>
                  <SelectTrigger className="w-48">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="risk">Sort by Risk Level</SelectItem>
                    <SelectItem value="date">Sort by Date</SelectItem>
                    <SelectItem value="confidence">Sort by Confidence</SelectItem>
                  </SelectContent>
                </Select>
                <Select value={riskFilter} onValueChange={(value: any) => setRiskFilter(value)}>
                  <SelectTrigger className="w-40">
                    <Filter className="w-4 h-4 mr-2" />
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Risk Levels</SelectItem>
                    <SelectItem value="high">High Risk Only</SelectItem>
                    <SelectItem value="medium">Medium Risk Only</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </CardContent>
          </Card>

          {/* Risk Scans List */}
          <div className="space-y-4">
            {filteredScans.length === 0 ? (
              <Card>
                <CardContent className="p-8 text-center">
                  <div className="text-gray-500">No risk assessments found</div>
                </CardContent>
              </Card>
            ) : (
              filteredScans.map((scan) => (
                <Card key={scan.id} className="hover:shadow-md transition-shadow">
                  <CardContent className="p-6">
                    <div className="flex items-center gap-4">
                      {/* Risk Icon */}
                      <div className="flex-shrink-0">
                        {getRiskIcon(scan.riskLevel)}
                      </div>

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
                                {new Date(scan.date).toLocaleDateString()}
                              </span>
                            </div>
                            <p className="text-sm text-gray-700 mt-2 font-medium">
                              {getRecommendation(scan)}
                            </p>
                          </div>
                          <div className="flex flex-col items-end gap-2">
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
                        {scan.followUpRequired && (
                          <div className="mt-3">
                            <Badge variant="outline" className="text-orange-600 border-orange-200">
                              Follow-up by: {scan.followUpDate}
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
        </TabsContent>

        <TabsContent value="indicators" className="space-y-4">
          {/* Risk Indicators List */}
          <div className="space-y-4">
            {riskIndicators.map((indicator) => (
              <Card key={indicator.id} className="hover:shadow-md transition-shadow">
                <CardContent className="p-6">
                  <div className="flex items-start justify-between">
                    <div className="flex items-start gap-3">
                      {getRiskIcon(indicator.severity)}
                      <div>
                        <h3 className="font-semibold text-gray-900">{indicator.description}</h3>
                        <p className="text-sm text-gray-600 mt-1">{indicator.recommendation}</p>
                        <div className="flex items-center gap-4 mt-2 text-xs text-gray-500">
                          <span>Type: {indicator.type.replace('_', ' ')}</span>
                          <span>Date: {new Date(indicator.date).toLocaleDateString()}</span>
                          <span>Confidence: {indicator.confidence}%</span>
                        </div>
                        {indicator.location && indicator.location !== 'N/A' && (
                          <p className="text-xs text-gray-500 mt-1">Location: {indicator.location}</p>
                        )}
                      </div>
                    </div>
                    <div className="flex flex-col items-end gap-2">
                      <Badge className={getRiskColor(indicator.severity)}>
                        {indicator.severity.toUpperCase()}
                      </Badge>
                      <Badge variant="outline" className="text-xs">
                        {indicator.status}
                      </Badge>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default RiskAssessmentsScreen;
