import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { 
  Brain, 
  AlertTriangle, 
  TrendingUp, 
  FileText,
  Search,
  Star,
  Clock,
  CheckCircle,
  XCircle,
  Info,
  Lightbulb,
  Activity,
  Pill,
  Heart,
  Zap
} from 'lucide-react';
import { useToast } from '@/hooks/use-toast';

interface AIInsight {
  id: string;
  type: 'diagnostic' | 'risk' | 'medication' | 'recommendation';
  title: string;
  description: string;
  confidence: number;
  priority: 'low' | 'medium' | 'high' | 'critical';
  patientId?: string;
  patientName?: string;
  evidence: string[];
  suggestions: string[];
  createdAt: Date;
}

interface DrugInteraction {
  id: string;
  drug1: string;
  drug2: string;
  severity: 'minor' | 'moderate' | 'major';
  description: string;
  recommendation: string;
}

interface ClinicalAlert {
  id: string;
  type: 'lab' | 'vital' | 'medication' | 'appointment';
  message: string;
  severity: 'info' | 'warning' | 'critical';
  patientName: string;
  timestamp: Date;
  acknowledged: boolean;
}

// Mock AI insights data
const mockInsights: AIInsight[] = [
  {
    id: 'AI001',
    type: 'diagnostic',
    title: 'Potential Diabetes Diagnosis',
    description: 'Based on recent lab results and patient symptoms, there is a 85% likelihood of Type 2 Diabetes.',
    confidence: 85,
    priority: 'high',
    patientId: 'P001',
    patientName: 'Sarah Johnson',
    evidence: [
      'HbA1c: 7.2% (elevated)',
      'Fasting glucose: 142 mg/dL (elevated)',
      'Patient reports increased thirst and urination',
      'BMI: 32.1 (obese category)'
    ],
    suggestions: [
      'Order glucose tolerance test',
      'Start metformin 500mg BID',
      'Refer to diabetes educator',
      'Schedule follow-up in 4 weeks'
    ],
    createdAt: new Date(2024, 1, 20, 10, 30)
  },
  {
    id: 'AI002',
    type: 'risk',
    title: 'High Cardiovascular Risk',
    description: 'Patient shows multiple risk factors for cardiovascular disease requiring immediate attention.',
    confidence: 92,
    priority: 'critical',
    patientId: 'P002',
    patientName: 'Michael Chen',
    evidence: [
      'Blood pressure: 165/95 (Stage 2 hypertension)',
      'Total cholesterol: 285 mg/dL',
      'Family history of MI',
      'Current smoker (1 pack/day)'
    ],
    suggestions: [
      'Start ACE inhibitor',
      'Prescribe high-intensity statin',
      'Smoking cessation counseling',
      'Cardiology consultation'
    ],
    createdAt: new Date(2024, 1, 20, 9, 15)
  },
  {
    id: 'AI003',
    type: 'medication',
    title: 'Drug Interaction Alert',
    description: 'Potential interaction between prescribed warfarin and newly added antibiotic.',
    confidence: 95,
    priority: 'high',
    patientId: 'P003',
    patientName: 'Emma Davis',
    evidence: [
      'Warfarin 5mg daily (anticoagulant)',
      'Ciprofloxacin 500mg BID (new prescription)',
      'Known interaction increases bleeding risk'
    ],
    suggestions: [
      'Consider alternative antibiotic',
      'If ciprofloxacin necessary, reduce warfarin dose',
      'Increase INR monitoring frequency',
      'Patient education on bleeding signs'
    ],
    createdAt: new Date(2024, 1, 20, 8, 45)
  }
];

const mockDrugInteractions: DrugInteraction[] = [
  {
    id: 'DI001',
    drug1: 'Warfarin',
    drug2: 'Aspirin',
    severity: 'major',
    description: 'Increased risk of bleeding when used together',
    recommendation: 'Monitor INR closely and watch for signs of bleeding'
  },
  {
    id: 'DI002',
    drug1: 'Lisinopril',
    drug2: 'Ibuprofen',
    severity: 'moderate',
    description: 'NSAIDs may reduce antihypertensive effect',
    recommendation: 'Consider alternative pain management or monitor BP closely'
  }
];

const mockClinicalAlerts: ClinicalAlert[] = [
  {
    id: 'CA001',
    type: 'lab',
    message: 'Critical potassium level (2.1 mEq/L) for Sarah Johnson',
    severity: 'critical',
    patientName: 'Sarah Johnson',
    timestamp: new Date(2024, 1, 20, 14, 30),
    acknowledged: false
  },
  {
    id: 'CA002',
    type: 'vital',
    message: 'Blood pressure 190/110 recorded for Michael Chen',
    severity: 'warning',
    patientName: 'Michael Chen',
    timestamp: new Date(2024, 1, 20, 13, 15),
    acknowledged: false
  }
];

export function AIDiagnosticSupport() {
  const { toast } = useToast();
  const [activeTab, setActiveTab] = useState('insights');
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedInsight, setSelectedInsight] = useState<AIInsight | null>(null);
  const [queryInput, setQueryInput] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  // Filter insights based on search and priority
  const filteredInsights = mockInsights.filter(insight =>
    insight.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
    insight.patientName?.toLowerCase().includes(searchTerm.toLowerCase()) ||
    insight.description.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const unacknowledgedAlerts = mockClinicalAlerts.filter(alert => !alert.acknowledged);

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'critical': return 'destructive';
      case 'high': return 'default';
      case 'medium': return 'secondary';
      case 'low': return 'outline';
      default: return 'secondary';
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'text-red-600 bg-red-50';
      case 'warning': return 'text-orange-600 bg-orange-50';
      case 'info': return 'text-blue-600 bg-blue-50';
      default: return 'text-gray-600 bg-gray-50';
    }
  };

  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case 'major': return <XCircle className="w-4 h-4 text-red-500" />;
      case 'moderate': return <AlertTriangle className="w-4 h-4 text-orange-500" />;
      case 'minor': return <Info className="w-4 h-4 text-blue-500" />;
      default: return <Info className="w-4 h-4 text-gray-500" />;
    }
  };

  const handleAIQuery = async () => {
    if (!queryInput.trim()) return;
    
    setIsAnalyzing(true);
    // Simulate AI processing
    setTimeout(() => {
      setIsAnalyzing(false);
      toast({
        title: "AI Analysis Complete",
        description: "Based on your query, I've generated new insights and recommendations.",
      });
      setQueryInput('');
    }, 3000);
  };

  const acknowledgeAlert = (alertId: string) => {
    toast({
      title: "Alert Acknowledged",
      description: "The clinical alert has been marked as reviewed.",
    });
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-foreground">AI Clinical Support</h1>
          <p className="text-muted-foreground mt-1">Intelligent insights and clinical decision support</p>
        </div>
        
        <div className="flex items-center space-x-3">
          <Badge variant="destructive" className="flex items-center space-x-1">
            <AlertTriangle className="w-3 h-3" />
            <span>{unacknowledgedAlerts.length} Alerts</span>
          </Badge>
          <Badge variant="secondary" className="flex items-center space-x-1">
            <Brain className="w-3 h-3" />
            <span>{filteredInsights.length} Insights</span>
          </Badge>
        </div>
      </div>

      {/* AI Query Interface */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Brain className="w-5 h-5" />
            <span>Ask AI Assistant</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex space-x-3">
            <Input
              value={queryInput}
              onChange={(e) => setQueryInput(e.target.value)}
              placeholder="Ask about patient symptoms, drug interactions, or clinical guidelines..."
              className="flex-1"
              disabled={isAnalyzing}
            />
            <Button onClick={handleAIQuery} disabled={!queryInput.trim() || isAnalyzing}>
              {isAnalyzing ? (
                <>
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                  Analyzing...
                </>
              ) : (
                <>
                  <Search className="w-4 h-4 mr-2" />
                  Analyze
                </>
              )}
            </Button>
          </div>
          
          {isAnalyzing && (
            <div className="mt-4">
              <div className="flex items-center space-x-2 mb-2">
                <Brain className="w-4 h-4 text-blue-500" />
                <span className="text-sm font-medium">AI Processing...</span>
              </div>
              <Progress value={60} className="h-2" />
              <p className="text-xs text-muted-foreground mt-1">
                Analyzing clinical data and evidence-based guidelines
              </p>
            </div>
          )}
        </CardContent>
      </Card>

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="insights">AI Insights</TabsTrigger>
          <TabsTrigger value="alerts">Clinical Alerts</TabsTrigger>
          <TabsTrigger value="interactions">Drug Interactions</TabsTrigger>
          <TabsTrigger value="analytics">Analytics</TabsTrigger>
        </TabsList>

        {/* AI Insights Tab */}
        <TabsContent value="insights" className="space-y-4">
          <div className="flex items-center space-x-3">
            <div className="relative flex-1">
              <Search className="w-4 h-4 absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground" />
              <Input 
                placeholder="Search insights..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="pl-9"
              />
            </div>
            <Select defaultValue="all">
              <SelectTrigger className="w-40">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Priorities</SelectItem>
                <SelectItem value="critical">Critical</SelectItem>
                <SelectItem value="high">High</SelectItem>
                <SelectItem value="medium">Medium</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="grid gap-4">
            {filteredInsights.map((insight) => (
              <Card key={insight.id} className="cursor-pointer hover:shadow-md transition-shadow"
                    onClick={() => setSelectedInsight(insight)}>
                <CardContent className="p-6">
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center space-x-3 mb-2">
                        <div className="p-2 rounded-full bg-blue-100">
                          {insight.type === 'diagnostic' && <FileText className="w-4 h-4 text-blue-600" />}
                          {insight.type === 'risk' && <AlertTriangle className="w-4 h-4 text-orange-600" />}
                          {insight.type === 'medication' && <Pill className="w-4 h-4 text-green-600" />}
                          {insight.type === 'recommendation' && <Lightbulb className="w-4 h-4 text-yellow-600" />}
                        </div>
                        <div>
                          <h3 className="font-semibold">{insight.title}</h3>
                          {insight.patientName && (
                            <p className="text-sm text-muted-foreground">Patient: {insight.patientName}</p>
                          )}
                        </div>
                      </div>
                      
                      <p className="text-sm text-muted-foreground mb-3">{insight.description}</p>
                      
                      <div className="flex items-center space-x-4">
                        <div className="flex items-center space-x-2">
                          <span className="text-xs font-medium">Confidence:</span>
                          <Progress value={insight.confidence} className="w-20 h-2" />
                          <span className="text-xs text-muted-foreground">{insight.confidence}%</span>
                        </div>
                        <Badge variant={getPriorityColor(insight.priority)} className="capitalize">
                          {insight.priority}
                        </Badge>
                        <span className="text-xs text-muted-foreground">
                          {insight.createdAt.toLocaleDateString()}
                        </span>
                      </div>
                    </div>
                    
                    <div className="flex flex-col items-end space-y-2">
                      <Button variant="outline" size="sm">
                        <Star className="w-3 h-3 mr-1" />
                        Save
                      </Button>
                      <Badge variant="outline" className="text-xs capitalize">
                        {insight.type}
                      </Badge>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        {/* Clinical Alerts Tab */}
        <TabsContent value="alerts" className="space-y-4">
          <div className="grid gap-4">
            {mockClinicalAlerts.map((alert) => (
              <Card key={alert.id} className={`border-l-4 ${
                alert.severity === 'critical' ? 'border-l-red-500' :
                alert.severity === 'warning' ? 'border-l-orange-500' : 'border-l-blue-500'
              }`}>
                <CardContent className="p-6">
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center space-x-3 mb-2">
                        <div className={`p-2 rounded-full ${getSeverityColor(alert.severity)}`}>
                          {alert.type === 'lab' && <Activity className="w-4 h-4" />}
                          {alert.type === 'vital' && <Heart className="w-4 h-4" />}
                          {alert.type === 'medication' && <Pill className="w-4 h-4" />}
                          {alert.type === 'appointment' && <Clock className="w-4 h-4" />}
                        </div>
                        <div>
                          <h3 className="font-semibold capitalize">{alert.type} Alert</h3>
                          <p className="text-sm text-muted-foreground">
                            {alert.timestamp.toLocaleString()}
                          </p>
                        </div>
                      </div>
                      
                      <p className="text-sm mb-3">{alert.message}</p>
                      
                      <Badge variant={
                        alert.severity === 'critical' ? 'destructive' :
                        alert.severity === 'warning' ? 'default' : 'secondary'
                      } className="capitalize">
                        {alert.severity}
                      </Badge>
                    </div>
                    
                    <div className="flex items-center space-x-2">
                      {!alert.acknowledged && (
                        <Button 
                          variant="outline" 
                          size="sm"
                          onClick={() => acknowledgeAlert(alert.id)}
                        >
                          <CheckCircle className="w-3 h-3 mr-1" />
                          Acknowledge
                        </Button>
                      )}
                      <Button variant="outline" size="sm">
                        <FileText className="w-3 h-3 mr-1" />
                        View Patient
                      </Button>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        {/* Drug Interactions Tab */}
        <TabsContent value="interactions" className="space-y-4">
          <div className="grid gap-4">
            {mockDrugInteractions.map((interaction) => (
              <Card key={interaction.id}>
                <CardContent className="p-6">
                  <div className="flex items-start space-x-4">
                    <div className="p-2 rounded-full bg-red-100">
                      {getSeverityIcon(interaction.severity)}
                    </div>
                    
                    <div className="flex-1">
                      <div className="flex items-center space-x-2 mb-2">
                        <h3 className="font-semibold">
                          {interaction.drug1} + {interaction.drug2}
                        </h3>
                        <Badge variant={
                          interaction.severity === 'major' ? 'destructive' :
                          interaction.severity === 'moderate' ? 'default' : 'secondary'
                        } className="capitalize">
                          {interaction.severity}
                        </Badge>
                      </div>
                      
                      <p className="text-sm text-muted-foreground mb-3">
                        {interaction.description}
                      </p>
                      
                      <div className="bg-blue-50 p-3 rounded-lg">
                        <h4 className="font-medium text-sm mb-1">Recommendation:</h4>
                        <p className="text-sm text-muted-foreground">
                          {interaction.recommendation}
                        </p>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        {/* Analytics Tab */}
        <TabsContent value="analytics" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <TrendingUp className="w-5 h-5" />
                  <span>AI Accuracy</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold mb-2">94.2%</div>
                <Progress value={94} className="mb-2" />
                <p className="text-sm text-muted-foreground">
                  Diagnostic prediction accuracy this month
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Zap className="w-5 h-5" />
                  <span>Alerts Generated</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold mb-2">127</div>
                <p className="text-sm text-muted-foreground mb-2">
                  Clinical alerts this week
                </p>
                <div className="text-xs text-green-600">
                  ↑ 15% from last week
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <CheckCircle className="w-5 h-5" />
                  <span>Recommendations</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold mb-2">89%</div>
                <p className="text-sm text-muted-foreground mb-2">
                  Accepted by providers
                </p>
                <div className="text-xs text-green-600">
                  ↑ 3% from last month
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>

      {/* Insight Detail Modal */}
      {selectedInsight && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center p-4 z-50">
          <Card className="w-full max-w-3xl max-h-[80vh] overflow-y-auto">
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="flex items-center space-x-2">
                  <Brain className="w-5 h-5" />
                  <span>{selectedInsight.title}</span>
                </CardTitle>
                <Button variant="ghost" onClick={() => setSelectedInsight(null)}>
                  ×
                </Button>
              </div>
            </CardHeader>
            <CardContent className="space-y-6">
              <div>
                <h4 className="font-medium mb-2">Analysis</h4>
                <p className="text-sm text-muted-foreground">{selectedInsight.description}</p>
              </div>

              <div>
                <h4 className="font-medium mb-2">Evidence</h4>
                <ul className="space-y-1">
                  {selectedInsight.evidence.map((item, index) => (
                    <li key={index} className="text-sm text-muted-foreground flex items-start space-x-2">
                      <CheckCircle className="w-3 h-3 mt-0.5 text-green-500 flex-shrink-0" />
                      <span>{item}</span>
                    </li>
                  ))}
                </ul>
              </div>

              <div>
                <h4 className="font-medium mb-2">Recommended Actions</h4>
                <ul className="space-y-1">
                  {selectedInsight.suggestions.map((item, index) => (
                    <li key={index} className="text-sm text-muted-foreground flex items-start space-x-2">
                      <Lightbulb className="w-3 h-3 mt-0.5 text-yellow-500 flex-shrink-0" />
                      <span>{item}</span>
                    </li>
                  ))}
                </ul>
              </div>

              <div className="flex items-center justify-between pt-4 border-t">
                <div className="flex items-center space-x-4">
                  <Badge variant={getPriorityColor(selectedInsight.priority)} className="capitalize">
                    {selectedInsight.priority} Priority
                  </Badge>
                  <span className="text-sm text-muted-foreground">
                    Confidence: {selectedInsight.confidence}%
                  </span>
                </div>
                <div className="space-x-2">
                  <Button variant="outline">Save to Notes</Button>
                  <Button>Accept Recommendations</Button>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}