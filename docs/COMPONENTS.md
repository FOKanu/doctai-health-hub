# 🎨 Component Library

## Overview

The DoctAI Health Hub component library provides a comprehensive set of React components built with TypeScript, Tailwind CSS, and shadcn/ui. This documentation covers all available components, their props, usage examples, and best practices.

## 📋 Table of Contents

- [🏗️ Architecture](#️-architecture)
- [🎨 UI Components](#-ui-components)
- [🏥 Healthcare Components](#-healthcare-components)
- [📊 Analytics Components](#-analytics-components)
- [🔐 Compliance Components](#-compliance-components)
- [📱 Layout Components](#-layout-components)
- [🔧 Custom Hooks](#-custom-hooks)
- [🎯 Best Practices](#-best-practices)

---

## 🏗️ Architecture

### Component Structure

```
src/components/
├── 📁 ui/                    # Base UI components (shadcn/ui)
├── 📁 analytics/             # Analytics & reporting components
├── 📁 appointments/          # Appointment management
├── 📁 compliance/            # HIPAA compliance features
├── 📁 diet/                  # Nutrition & meal planning
├── 📁 fitness/               # Fitness tracking
├── 📁 findcare/              # Provider search
├── 📁 home/                  # Dashboard components
├── 📁 layout/                # Layout components
├── 📁 modals/                # Modal dialogs
├── 📁 notifications/         # Notification system
├── 📁 results/               # Results display
├── 📁 settings/              # User settings
├── 📁 telemedicine/          # Video consultations
└── 📁 treatments/            # Treatment management
```

### Design System

- **Colors**: Healthcare-focused color palette
- **Typography**: Accessible font hierarchy
- **Spacing**: Consistent 4px grid system
- **Icons**: Lucide React icon library
- **Animations**: Smooth, purposeful transitions

---

## 🎨 UI Components

### Button Component

```tsx
import { Button } from '@/components/ui/button';

// Basic usage
<Button>Click me</Button>

// Variants
<Button variant="default">Default</Button>
<Button variant="destructive">Delete</Button>
<Button variant="outline">Outline</Button>
<Button variant="secondary">Secondary</Button>
<Button variant="ghost">Ghost</Button>
<Button variant="link">Link</Button>

// Sizes
<Button size="sm">Small</Button>
<Button size="default">Default</Button>
<Button size="lg">Large</Button>

// With icons
<Button>
  <Plus className="mr-2 h-4 w-4" />
  Add Item
</Button>
```

**Props:**
```tsx
interface ButtonProps {
  variant?: 'default' | 'destructive' | 'outline' | 'secondary' | 'ghost' | 'link';
  size?: 'default' | 'sm' | 'lg' | 'icon';
  asChild?: boolean;
  disabled?: boolean;
  children: React.ReactNode;
}
```

### Card Component

```tsx
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';

<Card>
  <CardHeader>
    <CardTitle>Patient Information</CardTitle>
    <CardDescription>View and edit patient details</CardDescription>
  </CardHeader>
  <CardContent>
    <p>Patient content goes here</p>
  </CardContent>
</Card>
```

### Input Component

```tsx
import { Input } from '@/components/ui/input';

<Input
  type="email"
  placeholder="Enter email address"
  value={email}
  onChange={(e) => setEmail(e.target.value)}
/>
```

### Form Component

```tsx
import { Form, FormControl, FormField, FormItem, FormLabel, FormMessage } from '@/components/ui/form';

<Form {...form}>
  <form onSubmit={form.handleSubmit(onSubmit)}>
    <FormField
      control={form.control}
      name="email"
      render={({ field }) => (
        <FormItem>
          <FormLabel>Email</FormLabel>
          <FormControl>
            <Input placeholder="Enter email" {...field} />
          </FormControl>
          <FormMessage />
        </FormItem>
      )}
    />
  </form>
</Form>
```

---

## 🏥 Healthcare Components

### PatientCard Component

```tsx
import { PatientCard } from '@/components/patients/PatientCard';

<PatientCard
  patient={{
    id: 'pat_123',
    name: 'John Doe',
    age: 35,
    gender: 'male',
    lastVisit: '2024-01-15',
    nextAppointment: '2024-02-15'
  }}
  onEdit={(patientId) => handleEdit(patientId)}
  onView={(patientId) => handleView(patientId)}
/>
```

**Props:**
```tsx
interface PatientCardProps {
  patient: {
    id: string;
    name: string;
    age: number;
    gender: 'male' | 'female' | 'other';
    lastVisit?: string;
    nextAppointment?: string;
  };
  onEdit?: (patientId: string) => void;
  onView?: (patientId: string) => void;
}
```

### AppointmentCalendar Component

```tsx
import { AppointmentCalendar } from '@/components/appointments/AppointmentCalendar';

<AppointmentCalendar
  appointments={appointments}
  onDateSelect={(date) => handleDateSelect(date)}
  onAppointmentClick={(appointment) => handleAppointmentClick(appointment)}
  view="month"
/>
```

### MedicalRecordViewer Component

```tsx
import { MedicalRecordViewer } from '@/components/medical/MedicalRecordViewer';

<MedicalRecordViewer
  patientId="pat_123"
  recordType="all"
  onExport={(data) => handleExport(data)}
  canEdit={true}
/>
```

### VitalSignsChart Component

```tsx
import { VitalSignsChart } from '@/components/analytics/VitalSignsChart';

<VitalSignsChart
  data={vitalSignsData}
  timeRange="30d"
  metrics={['bloodPressure', 'heartRate', 'temperature']}
  onDataPointClick={(point) => handleDataPointClick(point)}
/>
```

---

## 📊 Analytics Components

### HealthScoreCard Component

```tsx
import { HealthScoreCard } from '@/components/analytics/HealthScoreCard';

<HealthScoreCard
  score={85}
  trend="improving"
  factors={[
    { name: 'Blood Pressure', status: 'good' },
    { name: 'Weight', status: 'improving' },
    { name: 'Exercise', status: 'needs_attention' }
  ]}
  onFactorClick={(factor) => handleFactorClick(factor)}
/>
```

### RiskAssessmentChart Component

```tsx
import { RiskAssessmentChart } from '@/components/analytics/RiskAssessmentChart';

<RiskAssessmentChart
  data={riskData}
  categories={['cardiovascular', 'diabetes', 'cancer']}
  timeRange="12m"
  onRiskClick={(risk) => handleRiskClick(risk)}
/>
```

### AppointmentAnalytics Component

```tsx
import { AppointmentAnalytics } from '@/components/analytics/AppointmentAnalytics';

<AppointmentAnalytics
  data={appointmentData}
  metrics={['attendance', 'duration', 'satisfaction']}
  period="90d"
  onMetricChange={(metric) => handleMetricChange(metric)}
/>
```

---

## 🔐 Compliance Components

### ComplianceDashboard Component

```tsx
import { ComplianceDashboard } from '@/components/compliance/ComplianceDashboard';

<ComplianceDashboard
  metrics={{
    overallScore: 95,
    auditLogs: 1250,
    recentBreaches: 0,
    activeSessions: 45,
    pendingRequests: 3
  }}
  onBreachClick={(breach) => handleBreachClick(breach)}
  onRequestClick={(request) => handleRequestClick(request)}
/>
```

### AuditLogViewer Component

```tsx
import { AuditLogViewer } from '@/components/compliance/AuditLogViewer';

<AuditLogViewer
  logs={auditLogs}
  filters={{
    startDate: '2024-01-01',
    endDate: '2024-01-31',
    userId: 'user_123',
    action: 'read'
  }}
  onFilterChange={(filters) => handleFilterChange(filters)}
  onExport={(data) => handleExport(data)}
/>
```

### AccessRequestModal Component

```tsx
import { AccessRequestModal } from '@/components/compliance/AccessRequestModal';

<AccessRequestModal
  isOpen={isModalOpen}
  onClose={() => setIsModalOpen(false)}
  resource={{
    type: 'patient',
    id: 'pat_123',
    name: 'John Doe'
  }}
  onSubmit={(request) => handleSubmit(request)}
/>
```

---

## 📱 Layout Components

### DashboardLayout Component

```tsx
import { DashboardLayout } from '@/components/layout/DashboardLayout';

<DashboardLayout
  sidebar={<AppSidebar />}
  header={<AppHeader />}
  footer={<AppFooter />}
>
  <div>Main content</div>
</DashboardLayout>
```

### AppSidebar Component

```tsx
import { AppSidebar } from '@/components/layout/AppSidebar';

<AppSidebar
  items={[
    { label: 'Dashboard', icon: Home, href: '/' },
    { label: 'Patients', icon: Users, href: '/patients' },
    { label: 'Appointments', icon: Calendar, href: '/appointments' },
    { label: 'Analytics', icon: BarChart, href: '/analytics' }
  ]}
  collapsed={isCollapsed}
  onToggle={() => setIsCollapsed(!isCollapsed)}
/>
```

### AppHeader Component

```tsx
import { AppHeader } from '@/components/layout/AppHeader';

<AppHeader
  user={currentUser}
  notifications={notifications}
  onLogout={() => handleLogout()}
  onNotificationClick={(notification) => handleNotificationClick(notification)}
/>
```

---

## 🔧 Custom Hooks

### usePatient Hook

```tsx
import { usePatient } from '@/hooks/usePatient';

const { patient, loading, error, updatePatient } = usePatient('pat_123');

if (loading) return <div>Loading...</div>;
if (error) return <div>Error: {error.message}</div>;

return (
  <div>
    <h1>{patient.name}</h1>
    <button onClick={() => updatePatient({ age: 36 })}>
      Update Age
    </button>
  </div>
);
```

### useAppointments Hook

```tsx
import { useAppointments } from '@/hooks/useAppointments';

const { appointments, loading, createAppointment, cancelAppointment } = useAppointments({
  patientId: 'pat_123',
  startDate: '2024-02-01',
  endDate: '2024-02-28'
});
```

### useAnalytics Hook

```tsx
import { useAnalytics } from '@/hooks/useAnalytics';

const { data, loading, refresh } = useAnalytics({
  type: 'patient',
  patientId: 'pat_123',
  period: '30d',
  metrics: ['appointments', 'medications', 'vitals']
});
```

---

## 🎯 Best Practices

### Component Design Principles

1. **Single Responsibility**: Each component should have one clear purpose
2. **Composition over Inheritance**: Use composition to build complex components
3. **Props Interface**: Always define clear prop interfaces
4. **Default Props**: Provide sensible defaults where appropriate
5. **Error Boundaries**: Wrap components in error boundaries

### Performance Optimization

```tsx
// Use React.memo for expensive components
const ExpensiveComponent = React.memo(({ data }) => {
  return <div>{/* Expensive rendering */}</div>;
});

// Use useMemo for expensive calculations
const expensiveValue = useMemo(() => {
  return calculateExpensiveValue(data);
}, [data]);

// Use useCallback for event handlers
const handleClick = useCallback((id) => {
  handleItemClick(id);
}, [handleItemClick]);
```

### Accessibility Guidelines

```tsx
// Always provide alt text for images
<img src="patient-photo.jpg" alt="Patient profile photo" />

// Use semantic HTML elements
<button aria-label="Close modal" onClick={onClose}>
  <X className="h-4 w-4" />
</button>

// Provide keyboard navigation
<div role="button" tabIndex={0} onKeyDown={handleKeyDown}>
  Clickable content
</div>
```

### Testing Components

```tsx
import { render, screen, fireEvent } from '@testing-library/react';
import { PatientCard } from './PatientCard';

test('renders patient information', () => {
  const patient = {
    id: 'pat_123',
    name: 'John Doe',
    age: 35
  };

  render(<PatientCard patient={patient} />);

  expect(screen.getByText('John Doe')).toBeInTheDocument();
  expect(screen.getByText('35')).toBeInTheDocument();
});

test('calls onEdit when edit button is clicked', () => {
  const onEdit = jest.fn();
  const patient = { id: 'pat_123', name: 'John Doe' };

  render(<PatientCard patient={patient} onEdit={onEdit} />);

  fireEvent.click(screen.getByRole('button', { name: /edit/i }));

  expect(onEdit).toHaveBeenCalledWith('pat_123');
});
```

### Styling Guidelines

```tsx
// Use Tailwind classes for styling
<div className="flex items-center justify-between p-4 bg-white rounded-lg shadow-sm">
  <h2 className="text-lg font-semibold text-gray-900">Patient Name</h2>
  <button className="px-3 py-1 text-sm bg-blue-500 text-white rounded hover:bg-blue-600">
    Edit
  </button>
</div>

// Use CSS modules for complex styles
import styles from './PatientCard.module.css';

<div className={styles.patientCard}>
  <div className={styles.header}>
    <h2 className={styles.title}>{patient.name}</h2>
  </div>
</div>
```

### State Management

```tsx
// Use local state for component-specific data
const [isEditing, setIsEditing] = useState(false);
const [formData, setFormData] = useState(initialData);

// Use context for shared state
const { user, updateUser } = useUserContext();

// Use custom hooks for complex logic
const { data, loading, error, refetch } = usePatientData(patientId);
```

---

## 📞 Support

For component-related questions:

1. **Documentation**: Check this guide for component usage
2. **Examples**: Review the component examples in the codebase
3. **Issues**: Report bugs via GitHub issues
4. **Discussions**: Ask questions in GitHub discussions

## 🔗 Related Resources

- [shadcn/ui Documentation](https://ui.shadcn.com/)
- [Tailwind CSS Documentation](https://tailwindcss.com/docs)
- [React Documentation](https://react.dev/)
- [TypeScript Documentation](https://www.typescriptlang.org/docs/)
