
import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navigation from '../components/Navigation';
import HomeScreen from '../components/HomeScreen';
import ScanScreen from '../components/ScanScreen';
import UploadScreen from '../components/UploadScreen';
import SpecialistScreen from '../components/SpecialistScreen';
import HistoryScreen from '../components/HistoryScreen';
import MedicationsScreen from '../components/MedicationsScreen';
import ProfileScreen from '../components/ProfileScreen';

const Index = () => {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-white">
      <div className="max-w-md mx-auto bg-white min-h-screen shadow-lg">
        <Router>
          <Routes>
            <Route path="/" element={<HomeScreen />} />
            <Route path="/scan" element={<ScanScreen />} />
            <Route path="/upload" element={<UploadScreen />} />
            <Route path="/specialists" element={<SpecialistScreen />} />
            <Route path="/history" element={<HistoryScreen />} />
            <Route path="/medications" element={<MedicationsScreen />} />
            <Route path="/profile" element={<ProfileScreen />} />
          </Routes>
          <Navigation />
        </Router>
      </div>
    </div>
  );
};

export default Index;
