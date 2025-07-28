import { cva } from "class-variance-authority"

export const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-xl text-sm font-medium ring-offset-background transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 [&_svg]:pointer-events-none [&_svg]:size-4 [&_svg]:shrink-0",
  {
    variants: {
      variant: {
        default: "bg-primary text-primary-foreground hover:bg-primary/90 btn-glow shadow-lg hover:shadow-xl",
        destructive:
          "bg-destructive text-destructive-foreground hover:bg-destructive/90 shadow-lg",
        outline:
          "border border-primary text-primary bg-card hover:bg-primary/10 hover:shadow-lg",
        secondary:
          "bg-secondary text-secondary-foreground hover:bg-secondary/90 shadow-lg",
        ghost: "hover:bg-accent/10 hover:text-accent-foreground",
        link: "text-primary underline-offset-4 hover:underline",
        medical: "bg-gradient-to-r from-primary to-primary-glow text-primary-foreground hover:shadow-glow rounded-full btn-glow",
        success: "bg-success text-success-foreground hover:bg-success/90 shadow-lg",
        warning: "bg-warning text-warning-foreground hover:bg-warning/90 shadow-lg",
      },
      size: {
        default: "h-10 px-4 py-2",
        sm: "h-9 rounded-md px-3",
        lg: "h-11 rounded-md px-8",
        icon: "h-10 w-10",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
)
