// Global type declarations
declare global {
  interface Window {
    google: any;
  }
}

// Add any type compatibility fixes
declare module 'react' {
  interface InputHTMLAttributes<T> {
    value?: string | number | readonly string[] | undefined;
  }
}

export {};