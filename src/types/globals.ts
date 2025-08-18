// Global type suppression for development
// This file provides temporary type fixes for build errors

declare global {
  // Suppress console errors during development
  interface Console {
    suppressWarnings?: boolean;
  }

  // Add missing window properties
  interface Window {
    webkitSpeechRecognition?: any;
    SpeechRecognition?: any;
  }
}

// Export empty to make this a module
export {};