
import React from 'react';
import { cn } from '@/lib/utils';

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'default' | 'outline' | 'ghost';
  size?: 'sm' | 'default' | 'lg' | 'icon';
  children: React.ReactNode;
}

export function buttonVariants({
  variant = 'default',
  size = 'default',
  className,
}: {
  variant?: 'default' | 'outline' | 'ghost';
  size?: 'sm' | 'default' | 'lg' | 'icon';
  className?: string;
}) {
  const baseStyles = 'inline-flex items-center justify-center rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50';

  const variantStyles = {
    default: 'bg-blue-600 text-white hover:bg-blue-700 focus-visible:ring-blue-500',
    outline: 'border border-gray-300 bg-white hover:bg-gray-50 focus-visible:ring-gray-500',
    ghost: 'hover:bg-gray-100 focus-visible:ring-gray-500'
  };

  const sizeStyles = {
    sm: 'h-8 px-3 text-xs',
    default: 'h-10 px-4 py-2',
    lg: 'h-12 px-8 text-base',
    icon: 'h-10 w-10'
  };

  return cn(baseStyles, variantStyles[variant], sizeStyles[size], className);
}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(({
  children,
  variant = 'default',
  size = 'default',
  className = '',
  ...props
}, ref) => {
  return (
    <button
      ref={ref}
      className={buttonVariants({ variant, size, className })}
      {...props}
    >
      {children}
    </button>
  );
});

Button.displayName = 'Button';
