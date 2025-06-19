export interface Appointment {
  id: string;
  type: 'dermatologist' | 'oncologist' | 'general_practitioner' | 'specialist';
  doctor: string;
  date: string;
  time: string;
  location: string;
  address: string;
  notes?: string;
  outcome?: string;
  status: 'confirmed' | 'pending' | 'completed' | 'cancelled';
  duration: string;
  insurance: string;
  followUpRequired?: boolean;
  followUpDate?: string;
}

export const getAppointments = async (type: 'upcoming' | 'past' | 'all' = 'all'): Promise<Appointment[]> => {
  // In a real app, this would fetch from your database
  const appointments: Appointment[] = [
    {
      id: '1',
      type: 'dermatologist',
      doctor: 'Dr. Sarah Weber',
      date: '2024-06-20',
      time: '14:30',
      location: 'Berlin Medical Center',
      address: 'Kurfürstendamm 123, 10719 Berlin',
      notes: 'Follow-up for skin lesion monitoring',
      status: 'confirmed',
      duration: '30 min',
      insurance: 'TK - Techniker Krankenkasse'
    },
    {
      id: '2',
      type: 'oncologist',
      doctor: 'Prof. Dr. Michael Braun',
      date: '2024-06-25',
      time: '10:00',
      location: 'Munich Medical Center',
      address: 'Maximilianstraße 45, 80539 München',
      notes: 'Initial consultation for mole examination',
      status: 'pending',
      duration: '45 min',
      insurance: 'TK - Techniker Krankenkasse'
    },
    {
      id: '3',
      type: 'dermatologist',
      doctor: 'Dr. Michael Braun',
      date: '2024-06-10',
      time: '10:00',
      location: 'Munich Medical Center',
      address: 'Maximilianstraße 45, 80539 München',
      notes: 'Initial consultation - mole examination',
      outcome: 'Benign finding, 6-month follow-up recommended',
      status: 'completed',
      duration: '45 min',
      insurance: 'TK - Techniker Krankenkasse',
      followUpRequired: true,
      followUpDate: '2024-12-10'
    }
  ];

  const today = new Date();
  const upcoming = appointments.filter(apt => {
    const aptDate = new Date(apt.date);
    return aptDate >= today && apt.status !== 'cancelled';
  });

  const past = appointments.filter(apt => {
    const aptDate = new Date(apt.date);
    return aptDate < today || apt.status === 'completed';
  });

  switch (type) {
    case 'upcoming':
      return upcoming;
    case 'past':
      return past;
    default:
      return appointments;
  }
};

export const createAppointment = async (appointment: Omit<Appointment, 'id'>): Promise<Appointment> => {
  // In a real app, this would save to your database
  const newAppointment: Appointment = {
    ...appointment,
    id: Date.now().toString()
  };
  return newAppointment;
};

export const updateAppointment = async (id: string, updates: Partial<Appointment>): Promise<Appointment> => {
  // In a real app, this would update the database
  console.log(`Updating appointment ${id} with:`, updates);
  return { id, ...updates } as Appointment;
};

export const cancelAppointment = async (id: string): Promise<void> => {
  // In a real app, this would update the database
  console.log(`Cancelling appointment ${id}`);
};
