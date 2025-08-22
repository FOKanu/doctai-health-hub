# 🧹 DOCTAI CLEANUP PROTOCOL

## 📊 **AUDIT SUMMARY**
- **Total Files:** 250+ TypeScript/React files
- **Lines of Code:** 36,923 lines
- **Current Project Size:** 385MB
- **Build Output:** 885KB
- **Estimated Reduction:** 65% (385MB → ~135MB)

---

## 🚨 **CRITICAL CLEANUP (HIGH IMPACT)**

### **Phase 1: Remove Entire Unused Directories**

#### **1.1 ML Models Directory (200MB+)**
```bash
# COMPLETELY REMOVE - Not used in production
rm -rf ml_models/
```
**Files to Remove:**
- `ct_scan_classifier/` (unused)
- `eeg_classifier/` (unused)
- `mri_classifier/` (unused)
- `skin_lesion_classifier/` (unused)
- `xray_classifier/` (unused)
- `vital_signs_analyzer/` (unused)
- `progression_tracker/` (unused)
- `VinBigData Xray unused ML/` (unused)
- `health_score_aggregator.py` (unused)
- `medical_image_manager.py` (unused)
- `prediction_schema.py` (unused)
- `run_local_prediction.py` (unused)

**Reason:** App uses Supabase + external APIs, not local ML models

#### **1.2 Unused Scripts Directory (1.7MB)**
```bash
# REMOVE - Not used in production
rm -rf scripts/
```
**Files to Remove:**
- `setup-google-cloud.sh` (457 lines - unused)
- `setup-database-simple.sql` (780 lines - redundant)
- `setup-database.sh` (243 lines - redundant)
- `test-google-cloud.js` (89 lines - unused)
- `test-google-cloud-storage.js` (182 lines - unused)

#### **1.3 Documentation Bloat (1.9MB)**
```bash
# REMOVE - Keep only essential docs
rm -rf docs/
```
**Files to Remove:**
- `API.md` (200+ lines)
- `COMPLIANCE.md` (500+ lines)
- `COMPONENTS.md` (400+ lines)
- `DEPLOYMENT.md` (300+ lines)
- `README.md` (200+ lines)
- `TROUBLESHOOTING.md` (300+ lines)

---

## 🔧 **SERVICE CLEANUP (MEDIUM IMPACT)**

### **Phase 2: Remove Unused Services**

#### **2.1 Google Cloud Storage Service (299 lines)**
```bash
# REMOVE - Using Supabase Storage instead
rm src/services/cloudHealthcare/googleCloudStorageService.ts
```

#### **2.2 Azure Health Bot Service**
```bash
# REMOVE - Not configured/enabled
rm src/services/cloudHealthcare/azureHealthBotService.ts
```

#### **2.3 Redundant Prediction Services**
```bash
# CONSOLIDATE - Multiple prediction services
# Keep: src/services/predictionService.ts (470 lines)
# Remove: src/services/prediction/modernPredictionService.ts
# Remove: src/services/prediction/hybridPredictionService.ts (645 lines)
```

---

## 🎨 **UI COMPONENT CLEANUP (LOW IMPACT)**

### **Phase 3: Remove Unused UI Components**

#### **3.1 Unused UI Components (20+ files)**
```bash
# REMOVE - Not imported anywhere
rm src/components/ui/accordion.tsx
rm src/components/ui/aspect-ratio.tsx
rm src/components/ui/breadcrumb.tsx
rm src/components/ui/carousel.tsx
rm src/components/ui/chart.tsx
rm src/components/ui/collapsible.tsx
rm src/components/ui/command.tsx
rm src/components/ui/context-menu.tsx
rm src/components/ui/drawer.tsx
rm src/components/ui/hover-card.tsx
rm src/components/ui/input-otp.tsx
rm src/components/ui/menubar.tsx
rm src/components/ui/navigation-menu.tsx
rm src/components/ui/pagination.tsx
rm src/components/ui/popover.tsx
rm src/components/ui/resizable.tsx
rm src/components/ui/scrollable-tabs.tsx
rm src/components/ui/sheet.tsx
rm src/components/ui/toggle-group.tsx
```

---

## ⚙️ **ENVIRONMENT VARIABLES CLEANUP**

### **Phase 4: Remove Redundant Environment Variables**

#### **4.1 Redundant Flags**
```bash
# REMOVE - Redundant with VITE_USE_CLOUD_HEALTHCARE
VITE_ENABLE_GOOGLE_HEALTHCARE=true

# REMOVE - Redundant with VITE_CLOUD_HEALTHCARE_FALLBACK
VITE_USE_CLOUD_FALLBACK=true
```

#### **4.2 Unused Variables**
```bash
# REMOVE - Not used in codebase
VITE_GOOGLE_CLOUD_STORAGE_BUCKET=your_storage_bucket_name
```

---

## 🧪 **TESTING CLEANUP**

### **Phase 5: Minimal Testing Infrastructure**
```bash
# KEEP - Only 3 test files exist
tests/e2e/home.spec.ts
src/components/__tests__/Button.test.tsx
tsconfig.test.json

# CONSIDER - Add more tests later
```

---

## 📋 **CLEANUP EXECUTION PLAN**

### **Week 1: Critical Cleanup**
```bash
# Day 1-2: Remove large directories
rm -rf ml_models/
rm -rf scripts/
rm -rf docs/

# Day 3-4: Remove unused services
rm src/services/cloudHealthcare/googleCloudStorageService.ts
rm src/services/cloudHealthcare/azureHealthBotService.ts

# Day 5: Test build and deployment
npm run build
npm run test
```

### **Week 2: UI Component Cleanup**
```bash
# Day 1-3: Remove unused UI components
# (List of 20+ files above)

# Day 4-5: Test UI functionality
npm run dev
# Manual testing of all screens
```

### **Week 3: Code Optimization**
```bash
# Day 1-2: Remove unused imports
# Use ESLint to find and remove unused imports

# Day 3-4: Consolidate redundant services
# Merge prediction services

# Day 5: Final testing and deployment
npm run build
npm run test
git commit -m "Major cleanup: removed unused files and services"
```

---

## ✅ **VERIFICATION CHECKLIST**

### **After Each Phase:**
- [ ] Build succeeds: `npm run build`
- [ ] Tests pass: `npm run test`
- [ ] Development server works: `npm run dev`
- [ ] No console errors
- [ ] All screens load properly
- [ ] Deployment works on Vercel

### **Final Verification:**
- [ ] Project size reduced by 65%
- [ ] Build size reduced by 43%
- [ ] No broken imports
- [ ] All features still work
- [ ] Documentation updated

---

## 📊 **EXPECTED RESULTS**

### **Size Reduction:**
- **Project Size:** 385MB → ~135MB (65% reduction)
- **Build Size:** 885KB → ~500KB (43% reduction)
- **Lines of Code:** 36,923 → ~25,000 (32% reduction)

### **Performance Benefits:**
- **Faster CI/CD pipelines**
- **Reduced storage costs**
- **Easier onboarding for new developers**
- **Better code maintainability**
- **Faster development builds**

---

## 🚨 **RISK MITIGATION**

### **Before Cleanup:**
1. **Create backup branch:**
   ```bash
   git checkout -b backup-before-cleanup
   git push origin backup-before-cleanup
   ```

2. **Document current state:**
   ```bash
   du -sh . > project-size-before.txt
   find . -type f | wc -l > file-count-before.txt
   ```

### **During Cleanup:**
1. **Test after each major removal**
2. **Keep backup of removed files for 1 week**
3. **Monitor deployment logs**

### **After Cleanup:**
1. **Verify all features work**
2. **Update documentation**
3. **Inform team of changes**

---

## 📝 **NOTES**

### **Files to Keep:**
- `README.md` (main project README)
- `package.json` and `package-lock.json`
- `tsconfig.json` files
- `vite.config.ts`
- `tailwind.config.ts`
- `.env.example`
- `.gitignore`

### **Files to Review:**
- `src/services/` - Check for other unused services
- `src/components/` - Check for other unused components
- `src/hooks/` - Check for unused hooks
- `src/utils/` - Check for unused utilities

---

## 🔄 **MAINTENANCE PLAN**

### **Monthly:**
- Run ESLint to find unused imports
- Check for new unused files
- Review bundle size

### **Quarterly:**
- Full audit of unused dependencies
- Review and update this protocol
- Clean up any new bloat

---

**Last Updated:** $(date)
**Audit Version:** 1.0
**Status:** Ready for Execution
