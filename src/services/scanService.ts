export interface ScanRecord {
  id: string;
  date: string;
  timestamp: string;
  bodyPart: string;
  imageUrl: string;
  prediction: string;
  confidence: number;
  riskLevel: 'low' | 'medium' | 'high';
  notes?: string;
  type: 'skin_lesion' | 'xray' | 'mri' | 'ct_scan' | 'eeg';
  location?: string;
  followUpRequired?: boolean;
  followUpDate?: string;
}

export const getScans = async (): Promise<ScanRecord[]> => {
  // In a real app, this would fetch from your database
  // For now, return mock data
  return [
    {
      id: '1',
      date: '2024-06-15',
      timestamp: '14:30',
      bodyPart: 'Left Arm',
      imageUrl: '/scans/scan1.jpg',
      prediction: 'Benign Mole',
      confidence: 92,
      riskLevel: 'low',
      notes: 'Regular monitoring recommended',
      type: 'skin_lesion'
    },
    {
      id: '2',
      date: '2024-06-13',
      timestamp: '16:45',
      bodyPart: 'Back',
      imageUrl: '/scans/scan2.jpg',
      prediction: 'Atypical Nevus',
      confidence: 78,
      riskLevel: 'medium',
      notes: 'Irregular borders detected, specialist recommended',
      type: 'skin_lesion',
      followUpRequired: true,
      followUpDate: '2024-09-13'
    },
    {
      id: '3',
      date: '2024-06-10',
      timestamp: '09:15',
      bodyPart: 'Right Shoulder',
      imageUrl: '/scans/scan3.jpg',
      prediction: 'Melanoma',
      confidence: 95,
      riskLevel: 'high',
      notes: 'Immediate dermatologist consultation required',
      type: 'skin_lesion',
      followUpRequired: true,
      followUpDate: '2024-06-17'
    }
  ];
};

export const getScansByType = async (type: ScanRecord['type']): Promise<ScanRecord[]> => {
  const scans = await getScans();
  return scans.filter(scan => scan.type === type);
};

export const getScansByRiskLevel = async (riskLevel: ScanRecord['riskLevel']): Promise<ScanRecord[]> => {
  const scans = await getScans();
  return scans.filter(scan => scan.riskLevel === riskLevel);
};

export const saveScan = async (scan: Omit<ScanRecord, 'id'>): Promise<ScanRecord> => {
  // In a real app, this would save to your database
  const newScan: ScanRecord = {
    ...scan,
    id: Date.now().toString()
  };
  return newScan;
};
