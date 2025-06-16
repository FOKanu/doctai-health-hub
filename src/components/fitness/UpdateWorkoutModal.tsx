
import React, { useState } from 'react';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Checkbox } from '@/components/ui/checkbox';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent } from '@/components/ui/card';
import { Clock, Flame, Award, Dumbbell } from 'lucide-react';

interface UpdateWorkoutModalProps {
  isOpen: boolean;
  onClose: () => void;
  workout?: any;
}

export const UpdateWorkoutModal: React.FC<UpdateWorkoutModalProps> = ({ 
  isOpen, 
  onClose, 
  workout 
}) => {
  const [formData, setFormData] = useState({
    workoutType: workout?.title || '',
    duration: '',
    intensity: '',
    muscleGroups: [] as string[],
    exercises: '',
    calories: '',
    notes: '',
    rating: ''
  });

  const intensityLevels = [
    { value: 'light', label: 'Light', color: 'bg-green-100 text-green-800' },
    { value: 'moderate', label: 'Moderate', color: 'bg-yellow-100 text-yellow-800' },
    { value: 'vigorous', label: 'Vigorous', color: 'bg-orange-100 text-orange-800' },
    { value: 'intense', label: 'Intense', color: 'bg-red-100 text-red-800' }
  ];

  const muscleGroupOptions = [
    'Chest', 'Back', 'Shoulders', 'Arms', 'Legs', 'Glutes', 'Core', 'Cardio', 'Full Body'
  ];

  const handleMuscleGroupToggle = (muscleGroup: string) => {
    setFormData(prev => ({
      ...prev,
      muscleGroups: prev.muscleGroups.includes(muscleGroup)
        ? prev.muscleGroups.filter(mg => mg !== muscleGroup)
        : [...prev.muscleGroups, muscleGroup]
    }));
  };

  const calculatePoints = () => {
    const duration = parseInt(formData.duration) || 0;
    const intensityMultiplier = {
      'light': 1,
      'moderate': 1.5,
      'vigorous': 2,
      'intense': 2.5
    }[formData.intensity] || 1;
    
    return Math.round(duration * intensityMultiplier * 2);
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    console.log('Workout logged:', formData);
    console.log('Points earned:', calculatePoints());
    onClose();
    // Here you would typically save to your backend/database
  };

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Dumbbell className="w-5 h-5" />
            Log Workout Session
          </DialogTitle>
          <DialogDescription>
            Record your workout details to track progress and earn health points
          </DialogDescription>
        </DialogHeader>

        <form onSubmit={handleSubmit} className="space-y-6">
          {/* Workout Type */}
          <div className="space-y-2">
            <Label htmlFor="workoutType">Workout Type</Label>
            <Input
              id="workoutType"
              value={formData.workoutType}
              onChange={(e) => setFormData(prev => ({ ...prev, workoutType: e.target.value }))}
              placeholder="e.g., Upper Body Strength, HIIT Cardio"
            />
          </div>

          {/* Duration and Intensity */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="duration">Duration (minutes)</Label>
              <Input
                id="duration"
                type="number"
                value={formData.duration}
                onChange={(e) => setFormData(prev => ({ ...prev, duration: e.target.value }))}
                placeholder="30"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="intensity">Intensity Level</Label>
              <Select value={formData.intensity} onValueChange={(value) => setFormData(prev => ({ ...prev, intensity: value }))}>
                <SelectTrigger>
                  <SelectValue placeholder="Select intensity" />
                </SelectTrigger>
                <SelectContent>
                  {intensityLevels.map((level) => (
                    <SelectItem key={level.value} value={level.value}>
                      {level.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>

          {/* Muscle Groups */}
          <div className="space-y-2">
            <Label>Muscle Groups Worked</Label>
            <div className="grid grid-cols-3 gap-2">
              {muscleGroupOptions.map((muscle) => (
                <div key={muscle} className="flex items-center space-x-2">
                  <Checkbox
                    id={muscle}
                    checked={formData.muscleGroups.includes(muscle)}
                    onCheckedChange={() => handleMuscleGroupToggle(muscle)}
                  />
                  <label htmlFor={muscle} className="text-sm cursor-pointer">
                    {muscle}
                  </label>
                </div>
              ))}
            </div>
            {formData.muscleGroups.length > 0 && (
              <div className="flex flex-wrap gap-1 mt-2">
                {formData.muscleGroups.map((muscle) => (
                  <Badge key={muscle} variant="secondary" className="text-xs">
                    {muscle}
                  </Badge>
                ))}
              </div>
            )}
          </div>

          {/* Exercises */}
          <div className="space-y-2">
            <Label htmlFor="exercises">Exercises Performed</Label>
            <Textarea
              id="exercises"
              value={formData.exercises}
              onChange={(e) => setFormData(prev => ({ ...prev, exercises: e.target.value }))}
              placeholder="List the exercises you performed (e.g., Push-ups x 3 sets, Bench Press 10x3)"
              rows={3}
            />
          </div>

          {/* Calories and Rating */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="calories">Calories Burned (optional)</Label>
              <Input
                id="calories"
                type="number"
                value={formData.calories}
                onChange={(e) => setFormData(prev => ({ ...prev, calories: e.target.value }))}
                placeholder="320"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="rating">Workout Rating (1-5)</Label>
              <Select value={formData.rating} onValueChange={(value) => setFormData(prev => ({ ...prev, rating: value }))}>
                <SelectTrigger>
                  <SelectValue placeholder="Rate workout" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="1">1 - Poor</SelectItem>
                  <SelectItem value="2">2 - Fair</SelectItem>
                  <SelectItem value="3">3 - Good</SelectItem>
                  <SelectItem value="4">4 - Great</SelectItem>
                  <SelectItem value="5">5 - Excellent</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>

          {/* Notes */}
          <div className="space-y-2">
            <Label htmlFor="notes">Additional Notes</Label>
            <Textarea
              id="notes"
              value={formData.notes}
              onChange={(e) => setFormData(prev => ({ ...prev, notes: e.target.value }))}
              placeholder="How did you feel? Any observations or achievements?"
              rows={2}
            />
          </div>

          {/* Points Preview */}
          {formData.duration && formData.intensity && (
            <Card className="bg-gradient-to-r from-green-50 to-emerald-50 border-green-200">
              <CardContent className="p-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <Award className="w-5 h-5 text-green-600" />
                    <span className="font-medium text-green-800">Points to be earned:</span>
                  </div>
                  <div className="flex items-center space-x-4">
                    <div className="flex items-center space-x-1">
                      <Clock className="w-4 h-4 text-gray-500" />
                      <span className="text-sm">{formData.duration} min</span>
                    </div>
                    <div className="text-lg font-bold text-green-700">
                      +{calculatePoints()} points
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Action Buttons */}
          <div className="flex space-x-3 pt-4">
            <Button type="submit" className="flex-1">
              Log Workout
            </Button>
            <Button type="button" variant="outline" onClick={onClose}>
              Cancel
            </Button>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  );
};
