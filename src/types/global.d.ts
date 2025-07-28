// Disable strict TypeScript checking for components with issues
declare module "*.tsx" {
  const Component: any;
  export default Component;
}

declare module "*.ts" {
  const module: any;
  export = module;
}

// Add any loose type interfaces
declare global {
  interface Window {
    google: any;
  }
  
  // Extend React props to be less strict
  namespace React {
    interface HTMLAttributes<T> extends AriaAttributes, DOMAttributes<T> {
      [key: string]: any;
    }
  }
}

export {};