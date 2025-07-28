/// <reference types="vite/client" />

declare module '*.tsx' {
  const Component: React.ComponentType<any>;
  export default Component;
}

declare module '*.ts' {
  const module: any;
  export = module;
}

// Disable strict checking
declare global {
  interface Window {
    google: any;
  }
  
  var globalThis: any;
}
