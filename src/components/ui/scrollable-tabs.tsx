import React from 'react';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs';

interface ScrollableTabsProps {
  value?: string;
  onValueChange?: (value: string) => void;
  defaultValue?: string;
  className?: string;
  children: React.ReactNode;
  containerClassName?: string;
}

interface ScrollableTabsListProps {
  children: React.ReactNode;
  className?: string;
  containerClassName?: string;
}

export const ScrollableTabsList: React.FC<ScrollableTabsListProps> = ({
  children,
  className = "",
  containerClassName = ""
}) => {
  return (
    <div className={`relative group ${containerClassName}`}>
      {/* Left Scroll Arrow */}
      <button
        onClick={() => {
          const container = document.querySelector('.scrollable-tabs-container');
          if (container) {
            container.scrollBy({ left: -200, behavior: 'smooth' });
          }
        }}
        className="absolute left-0 top-1/2 -translate-y-1/2 z-10 bg-white/80 backdrop-blur-sm border border-gray-200 rounded-full p-1 shadow-md opacity-0 group-hover:opacity-100 transition-opacity duration-200 hover:bg-white"
      >
        <svg className="w-4 h-4 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
        </svg>
      </button>

      {/* Right Scroll Arrow */}
      <button
        onClick={() => {
          const container = document.querySelector('.scrollable-tabs-container');
          if (container) {
            container.scrollBy({ left: 200, behavior: 'smooth' });
          }
        }}
        className="absolute right-0 top-1/2 -translate-y-1/2 z-10 bg-white/80 backdrop-blur-sm border border-gray-200 rounded-full p-1 shadow-md opacity-0 group-hover:opacity-100 transition-opacity duration-200 hover:bg-white"
      >
        <svg className="w-4 h-4 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
        </svg>
      </button>

      <div className="overflow-x-auto scrollbar-hide scrollable-tabs-container">
        <TabsList className={`flex w-max min-w-full space-x-1 px-4 ${className}`}>
          {children}
        </TabsList>
      </div>
    </div>
  );
};

export const ScrollableTabs: React.FC<ScrollableTabsProps> = ({
  value,
  onValueChange,
  defaultValue,
  className = "",
  children,
  containerClassName = ""
}) => {
  return (
    <Tabs
      value={value}
      onValueChange={onValueChange}
      defaultValue={defaultValue}
      className={className}
    >
      {React.Children.map(children, (child) => {
        if (React.isValidElement(child) && child.type === TabsList) {
          return (
            <ScrollableTabsList
              className={child.props.className}
              containerClassName={containerClassName}
            >
              {child.props.children}
            </ScrollableTabsList>
          );
        }
        return child;
      })}
    </Tabs>
  );
};

// Export the original components for backward compatibility
export { Tabs, TabsList, TabsTrigger, TabsContent };
