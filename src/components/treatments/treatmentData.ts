
export const treatmentData = {
  overview: {
    patientName: "John Doe",
    diagnosis: "Hypertension Management",
    startDate: "2024-01-15",
    endDate: "2024-07-15",
    assignedDoctor: "Dr. Sarah Johnson",
    clinic: "Heart Care Clinic"
  },
  medications: [
    {
      id: "1",
      name: "Lisinopril",
      dosage: "10mg",
      frequency: "Once daily",
      duration: "6 months",
      color: "blue",
      takenToday: true,
      adherenceRate: 95
    },
    {
      id: "2",
      name: "Metformin",
      dosage: "500mg",
      frequency: "Twice daily",
      duration: "Ongoing",
      color: "green",
      takenToday: false,
      adherenceRate: 88
    },
    {
      id: "3",
      name: "Aspirin",
      dosage: "81mg",
      frequency: "Once daily",
      duration: "As needed",
      color: "red",
      takenToday: true,
      adherenceRate: 92
    }
  ],
  procedures: [
    {
      id: "1",
      name: "Cardiac Stress Test",
      date: "2024-07-05",
      time: "10:00 AM",
      specialist: "Dr. Michael Chen",
      hospital: "City Medical Center",
      status: "confirmed"
    },
    {
      id: "2",
      name: "Blood Work Follow-up",
      date: "2024-07-20",
      time: "9:30 AM",
      specialist: "Dr. Sarah Johnson",
      hospital: "Heart Care Clinic",
      status: "pending"
    }
  ],
  notes: [
    "Patient showing good response to current medication regimen",
    "Blood pressure readings have stabilized within normal range",
    "Continue monitoring sodium intake and regular exercise"
  ],
  instructions: [
    "Take medications at the same time each day",
    "Monitor blood pressure daily and log readings",
    "Avoid excessive sodium intake (less than 2300mg daily)",
    "Engage in moderate exercise 30 minutes daily",
    "Report any chest pain or shortness of breath immediately"
  ],
  appointments: [
    {
      id: "1",
      date: "2024-07-10",
      time: "2:00 PM",
      purpose: "Blood Pressure Check",
      location: "Heart Care Clinic",
      specialist: "Dr. Sarah Johnson",
      status: "confirmed"
    },
    {
      id: "2",
      date: "2024-07-25",
      time: "11:00 AM",
      purpose: "Medication Review",
      location: "Heart Care Clinic",
      specialist: "Dr. Sarah Johnson",
      status: "scheduled"
    }
  ],
  documents: [
    {
      id: "1",
      name: "Blood Test Results - June 2024",
      type: "PDF",
      uploadDate: "2024-06-28",
      size: "245 KB"
    },
    {
      id: "2",
      name: "EKG Report",
      type: "PDF",
      uploadDate: "2024-06-15",
      size: "180 KB"
    },
    {
      id: "3",
      name: "Prescription History",
      type: "PDF",
      uploadDate: "2024-06-01",
      size: "120 KB"
    }
  ]
};
