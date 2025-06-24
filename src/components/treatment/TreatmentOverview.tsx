
import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from '@/components/ui/accordion';
import { 
  User, 
  Pill, 
  Calendar, 
  Hospital, 
  FileText, 
  TrendingUp,
  Clock,
  AlertCircle
} from 'lucide-react';

interface TreatmentData {
  userName: string;
  diagnosis: string;
  prescriptions: Array<{
    medication: string;
    dosage: string;
    duration: string;
    instructions: string;
  }>;
  specialistRecommendations: string[];
  nextCheckup: string;
  assignedClinic: string;
  notes: string;
  progress: {
    improvementScore: number;
    lastUpdated: string;
  };
}

interface TreatmentOverviewProps {
  treatmentData: TreatmentData;
}

const TreatmentOverview: React.FC<TreatmentOverviewProps> = ({ treatmentData }) => {
  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric'
    });
  };

  const getDaysUntilCheckup = (dateString: string) => {
    const checkupDate = new Date(dateString);
    const today = new Date();
    const diffTime = checkupDate.getTime() - today.getTime();
    const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
    return diffDays;
  };

  const daysUntilCheckup = getDaysUntilCheckup(treatmentData.nextCheckup);

  return (
    <div className="space-y-6">
      {/* Patient Summary Card */}
      <Card className="border-l-4 border-l-blue-500">
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-blue-100 rounded-lg">
                <User className="w-5 h-5 text-blue-600" />
              </div>
              <div>
                <CardTitle className="text-lg">{treatmentData.userName}</CardTitle>
                <p className="text-sm text-gray-600">Primary Condition: {treatmentData.diagnosis}</p>
              </div>
            </div>
            <Badge variant="outline" className="bg-blue-50 text-blue-700 border-blue-200">
              Active Treatment
            </Badge>
          </div>
        </CardHeader>
        <CardContent className="pt-0">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="text-center p-3 bg-gray-50 rounded-lg">
              <div className="text-2xl font-bold text-green-600">{treatmentData.progress.improvementScore}%</div>
              <div className="text-sm text-gray-600">Improvement</div>
            </div>
            <div className="text-center p-3 bg-gray-50 rounded-lg">
              <div className="text-2xl font-bold text-orange-600">{daysUntilCheckup}</div>
              <div className="text-sm text-gray-600">Days to Checkup</div>
            </div>
            <div className="text-center p-3 bg-gray-50 rounded-lg">
              <div className="text-2xl font-bold text-blue-600">{treatmentData.prescriptions.length}</div>
              <div className="text-sm text-gray-600">Active Medications</div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Treatment Details Accordion */}
      <Accordion type="multiple" defaultValue={["medications", "checkup"]} className="space-y-2">
        
        {/* Prescribed Medications */}
        <AccordionItem value="medications">
          <Card>
            <AccordionTrigger className="px-6 py-4 hover:no-underline">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-green-100 rounded-lg">
                  <Pill className="w-5 h-5 text-green-600" />
                </div>
                <div className="text-left">
                  <h3 className="font-semibold">Prescribed Medications</h3>
                  <p className="text-sm text-gray-600">{treatmentData.prescriptions.length} active prescriptions</p>
                </div>
              </div>
            </AccordionTrigger>
            <AccordionContent className="px-6 pb-4">
              <div className="space-y-4">
                {treatmentData.prescriptions.map((prescription, index) => (
                  <div key={index} className="border border-gray-200 rounded-lg p-4 bg-gray-50">
                    <div className="flex justify-between items-start mb-2">
                      <h4 className="font-medium text-gray-900">{prescription.medication}</h4>
                      <Badge variant="outline" className="bg-green-50 text-green-700 border-green-200">
                        {prescription.duration}
                      </Badge>
                    </div>
                    <p className="text-sm text-gray-700 mb-1"><strong>Dosage:</strong> {prescription.dosage}</p>
                    <p className="text-sm text-gray-600">{prescription.instructions}</p>
                  </div>
                ))}
              </div>
            </AccordionContent>
          </Card>
        </AccordionItem>

        {/* Next Checkup */}
        <AccordionItem value="checkup">
          <Card>
            <AccordionTrigger className="px-6 py-4 hover:no-underline">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-orange-100 rounded-lg">
                  <Calendar className="w-5 h-5 text-orange-600" />
                </div>
                <div className="text-left">
                  <h3 className="font-semibold">Next Checkup</h3>
                  <p className="text-sm text-gray-600">{formatDate(treatmentData.nextCheckup)}</p>
                </div>
                {daysUntilCheckup <= 7 && (
                  <AlertCircle className="w-5 h-5 text-orange-500 ml-auto" />
                )}
              </div>
            </AccordionTrigger>
            <AccordionContent className="px-6 pb-4">
              <div className="bg-orange-50 border border-orange-200 rounded-lg p-4">
                <div className="flex items-center justify-between mb-3">
                  <div>
                    <h4 className="font-medium text-gray-900">Upcoming Appointment</h4>
                    <p className="text-sm text-gray-600">{formatDate(treatmentData.nextCheckup)}</p>
                  </div>
                  <Badge variant={daysUntilCheckup <= 7 ? "destructive" : "outline"}>
                    {daysUntilCheckup} days
                  </Badge>
                </div>
                <div className="flex items-center gap-2 text-sm text-gray-700">
                  <Hospital className="w-4 h-4" />
                  <span>{treatmentData.assignedClinic}</span>
                </div>
              </div>
            </AccordionContent>
          </Card>
        </AccordionItem>

        {/* Specialist Recommendations */}
        <AccordionItem value="specialists">
          <Card>
            <AccordionTrigger className="px-6 py-4 hover:no-underline">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-purple-100 rounded-lg">
                  <User className="w-5 h-5 text-purple-600" />
                </div>
                <div className="text-left">
                  <h3 className="font-semibold">Specialist Recommendations</h3>
                  <p className="text-sm text-gray-600">{treatmentData.specialistRecommendations.length} recommendations</p>
                </div>
              </div>
            </AccordionTrigger>
            <AccordionContent className="px-6 pb-4">
              <div className="space-y-2">
                {treatmentData.specialistRecommendations.map((specialist, index) => (
                  <div key={index} className="flex items-center justify-between p-3 bg-purple-50 border border-purple-200 rounded-lg">
                    <span className="font-medium text-gray-900">{specialist}</span>
                    <Badge variant="outline" className="bg-purple-100 text-purple-700 border-purple-300">
                      Recommended
                    </Badge>
                  </div>
                ))}
              </div>
            </AccordionContent>
          </Card>
        </AccordionItem>

        {/* Treatment Notes */}
        <AccordionItem value="notes">
          <Card>
            <AccordionTrigger className="px-6 py-4 hover:no-underline">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-gray-100 rounded-lg">
                  <FileText className="w-5 h-5 text-gray-600" />
                </div>
                <div className="text-left">
                  <h3 className="font-semibold">Treatment Notes & Guidelines</h3>
                  <p className="text-sm text-gray-600">Important care instructions</p>
                </div>
              </div>
            </AccordionTrigger>
            <AccordionContent className="px-6 pb-4">
              <div className="bg-gray-50 border border-gray-200 rounded-lg p-4">
                <p className="text-gray-700 leading-relaxed">{treatmentData.notes}</p>
              </div>
            </AccordionContent>
          </Card>
        </AccordionItem>

      </Accordion>

      {/* Progress Tracking */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-3">
            <div className="p-2 bg-green-100 rounded-lg">
              <TrendingUp className="w-5 h-5 text-green-600" />
            </div>
            <div>
              <CardTitle className="text-lg">Treatment Progress</CardTitle>
              <p className="text-sm text-gray-600">Last updated: {formatDate(treatmentData.progress.lastUpdated)}</p>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-sm font-medium text-gray-700">Overall Improvement</span>
              <span className="text-sm font-semibold text-green-600">{treatmentData.progress.improvementScore}%</span>
            </div>
            <Progress value={treatmentData.progress.improvementScore} className="h-2" />
            <p className="text-xs text-gray-500">
              Based on symptom tracking and clinical assessments
            </p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default TreatmentOverview;
