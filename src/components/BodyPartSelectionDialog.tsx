
import React from 'react';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group';
import { Label } from '@/components/ui/label';

export type BodyPart = 'Face' | 'Neck' | 'Chest' | 'Back' | 'Arm' | 'Leg' | 'Foot' | 'Other';

interface BodyPartSelectionDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onBodyPartSelect: (bodyPart: BodyPart) => void;
  selectedBodyPart: BodyPart | null;
}

const bodyParts: BodyPart[] = ['Face', 'Neck', 'Chest', 'Back', 'Arm', 'Leg', 'Foot', 'Other'];

export const BodyPartSelectionDialog: React.FC<BodyPartSelectionDialogProps> = ({
  open,
  onOpenChange,
  onBodyPartSelect,
  selectedBodyPart
}) => {
  const handleContinue = () => {
    if (selectedBodyPart) {
      onBodyPartSelect(selectedBodyPart);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Select Body Part</DialogTitle>
          <DialogDescription>
            Which body part are you scanning? This helps improve the accuracy of our analysis.
          </DialogDescription>
        </DialogHeader>
        
        <div className="py-4">
          <RadioGroup 
            value={selectedBodyPart || ''} 
            onValueChange={(value) => onBodyPartSelect(value as BodyPart)}
            className="grid grid-cols-2 gap-4"
          >
            {bodyParts.map((bodyPart) => (
              <div key={bodyPart} className="flex items-center space-x-2">
                <RadioGroupItem value={bodyPart} id={bodyPart} />
                <Label htmlFor={bodyPart} className="cursor-pointer">
                  {bodyPart}
                </Label>
              </div>
            ))}
          </RadioGroup>
        </div>

        <DialogFooter>
          <Button
            onClick={handleContinue}
            disabled={!selectedBodyPart}
            className="w-full"
          >
            Continue to Scan
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};
