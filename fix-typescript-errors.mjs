#!/usr/bin/env node

import fs from 'fs';
import path from 'path';

// TypeScript error fixes
const fixes = [
  // Fix SelectionChangeHandler type errors
  {
    file: 'src/components/AppointmentsScreen.tsx',
    pattern: /onValueChange=\{setSortBy\}/g,
    replacement: 'onValueChange={(value) => setSortBy(value as "date" | "type" | "status")}'
  },
  {
    file: 'src/components/RiskAssessmentsScreen.tsx',
    pattern: /onValueChange=\{setSortBy\}/g,
    replacement: 'onValueChange={(value) => setSortBy(value as "date" | "risk" | "confidence")}'
  },
  {
    file: 'src/components/RiskAssessmentsScreen.tsx',
    pattern: /onValueChange=\{setFilterBy\}/g,
    replacement: 'onValueChange={(value) => setFilterBy(value as "all" | "high" | "medium")}'
  },
  {
    file: 'src/components/TotalScansScreen.tsx',
    pattern: /onValueChange=\{setSortBy\}/g,
    replacement: 'onValueChange={(value) => setSortBy(value as "date" | "riskLevel" | "confidence")}'
  },
  {
    file: 'src/components/TotalScansScreen.tsx',
    pattern: /onValueChange=\{setFilterBy\}/g,
    replacement: 'onValueChange={(value) => setFilterBy(value as "all" | "high" | "medium" | "low")}'
  },

  // Fix type imports
  {
    file: 'src/components/MedicationsScreen.tsx',
    pattern: /^/,
    replacement: 'import { Medication } from "@/types";\n'
  },
  {
    file: 'src/components/modals/ScheduleAppointmentModal.tsx',
    pattern: /^/,
    replacement: 'import { Appointment } from "@/types";\n'
  },

  // Fix function parameter type errors
  {
    file: 'src/components/TreatmentsScreen.tsx',
    pattern: /onAddTreatment=\{.*?\}/g,
    replacement: 'onAddTreatment={(treatment) => setTreatments([...treatments, treatment])}'
  },

  // Fix SyntheticEvent type errors in modals
  {
    file: 'src/components/modals/AddMedicationModal.tsx',
    pattern: /onAddMedication\(newMedication\)/g,
    replacement: 'onAddMedication?.(newMedication)'
  },
  {
    file: 'src/components/modals/AddTreatmentModal.tsx',
    pattern: /onAddTreatment\(newTreatment\)/g,
    replacement: 'onAddTreatment?.(newTreatment)'
  },

  // Fix variable hoisting issues
  {
    file: 'src/components/ScanScreen.tsx',
    pattern: /useEffect\(\(\) => \{\s*startCamera\(\);\s*return stopCamera;\s*\}, \[\]\);/g,
    replacement: 'useEffect(() => {\n    const start = async () => await startCamera();\n    start();\n    return () => stopCamera();\n  }, []);'
  },

  // Fix filter function parameters
  {
    file: 'src/components/findcare/FilterSidebar.tsx',
    pattern: /onSpecialtiesChange\(updatedSpecialties\)/g,
    replacement: 'onSpecialtiesChange?.(updatedSpecialties)'
  },
  {
    file: 'src/components/findcare/FilterSidebar.tsx',
    pattern: /onDistanceChange\(selectedDistance\)/g,
    replacement: 'onDistanceChange?.(selectedDistance)'
  },
  {
    file: 'src/components/findcare/FilterSidebar.tsx',
    pattern: /onDistanceChange\(distance, value\)/g,
    replacement: 'onDistanceChange?.(distance)'
  },
  {
    file: 'src/components/findcare/SelectFilterSection.tsx',
    pattern: /onChange\(value\)/g,
    replacement: 'onChange?.(value)'
  },

  // Fix DietPlanScreen profile save
  {
    file: 'src/components/DietPlanScreen.tsx',
    pattern: /onSave\(profile\)/g,
    replacement: 'onSave?.(profile)'
  },

  // Fix undefined function error in GoogleMapView
  {
    file: 'src/components/GoogleMapView.tsx',
    pattern: /searchNearbyPlaces\(/g,
    replacement: 'searchNearbyPlacesFunction('
  },

  // Fix notification manager
  {
    file: 'src/components/notifications/NotificationManager.tsx',
    pattern: /onSchedule\(medication\)/g,
    replacement: 'onSchedule?.(medication)'
  }
];

function applyFixes() {
  console.log('🔧 Applying TypeScript fixes...');
  
  fixes.forEach(fix => {
    const filePath = fix.file;
    
    if (!fs.existsSync(filePath)) {
      console.log(`⚠️  File not found: ${filePath}`);
      return;
    }
    
    let content = fs.readFileSync(filePath, 'utf8');
    
    if (fix.pattern.test(content)) {
      content = content.replace(fix.pattern, fix.replacement);
      fs.writeFileSync(filePath, content);
      console.log(`✅ Fixed: ${filePath}`);
    } else {
      console.log(`⏭️  No changes needed: ${filePath}`);
    }
  });
  
  console.log('🎉 TypeScript fixes completed!');
}

applyFixes();