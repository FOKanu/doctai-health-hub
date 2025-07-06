import React, { useState } from 'react';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle, AlertDialogTrigger } from '@/components/ui/alert-dialog';
import { Cloud, Download, HardDrive, Trash2 } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';

interface DataStorageModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  type: 'sync' | 'export' | 'storage' | 'cache';
}

export const DataStorageModal = ({ open, onOpenChange, type }: DataStorageModalProps) => {
  const [cloudSync, setCloudSync] = useState(true);
  const [autoSync, setAutoSync] = useState(false);
  const [lastSync] = useState('2 hours ago');
  const [storageUsed] = useState(2.3);
  const [storageTotal] = useState(5);
  const [cacheSize] = useState('324 MB');
  const { toast } = useToast();

  const handleExportData = () => {
    toast({
      title: "Export Started",
      description: "Your data export is being prepared. You'll receive a download link shortly."
    });
    onOpenChange(false);
  };

  const handleClearCache = () => {
    toast({
      title: "Cache Cleared",
      description: `${cacheSize} of cached data has been cleared successfully.`
    });
    onOpenChange(false);
  };

  const getTitle = () => {
    switch (type) {
      case 'sync': return 'Sync Settings';
      case 'export': return 'Export Data';
      case 'storage': return 'Storage Usage';
      case 'cache': return 'Clear Cache';
      default: return 'Data & Storage';
    }
  };

  const renderContent = () => {
    switch (type) {
      case 'sync':
        return (
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <div className="space-y-1">
                <Label className="text-base">Cloud Sync</Label>
                <p className="text-sm text-muted-foreground">
                  Sync your data across all devices
                </p>
              </div>
              <Switch checked={cloudSync} onCheckedChange={setCloudSync} />
            </div>

            <div className="flex items-center justify-between">
              <div className="space-y-1">
                <Label className="text-base">Auto Sync</Label>
                <p className="text-sm text-muted-foreground">
                  Automatically sync when changes are made
                </p>
              </div>
              <Switch checked={autoSync} onCheckedChange={setAutoSync} disabled={!cloudSync} />
            </div>

            <div className="p-4 bg-medical-accent-light rounded-lg">
              <div className="flex items-center gap-2 mb-2">
                <Cloud className="h-4 w-4 text-medical-accent" />
                <span className="font-medium">Sync Status</span>
              </div>
              <p className="text-sm text-muted-foreground">
                Last sync: {lastSync}
              </p>
            </div>

            <div className="flex justify-end space-x-2">
              <Button variant="outline" onClick={() => onOpenChange(false)}>
                Close
              </Button>
              <Button onClick={() => toast({ title: "Sync initiated", description: "Your data is being synced now." })}>
                Sync Now
              </Button>
            </div>
          </div>
        );

      case 'export':
        return (
          <div className="space-y-6">
            <div className="flex items-center gap-3 p-4 bg-medical-accent-light rounded-lg">
              <Download className="h-6 w-6 text-medical-accent" />
              <div>
                <h3 className="font-medium">Export Your Health Data</h3>
                <p className="text-sm text-muted-foreground">
                  Download all your health records, scans, and medical data
                </p>
              </div>
            </div>

            <div className="space-y-3">
              <h4 className="font-medium">What's included:</h4>
              <ul className="space-y-1 text-sm text-muted-foreground">
                <li>• Medical scan results and images</li>
                <li>• Health analytics and reports</li>
                <li>• Appointment history</li>
                <li>• Medication records</li>
                <li>• Personal health profile</li>
              </ul>
            </div>

            <div className="p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
              <p className="text-sm text-yellow-800">
                <strong>Note:</strong> The export may take a few minutes to prepare. 
                You'll receive an email with the download link.
              </p>
            </div>

            <div className="flex justify-end space-x-2">
              <Button variant="outline" onClick={() => onOpenChange(false)}>
                Cancel
              </Button>
              <Button onClick={handleExportData}>
                Export Data
              </Button>
            </div>
          </div>
        );

      case 'storage':
        return (
          <div className="space-y-6">
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <Label className="text-base">Storage Usage</Label>
                <span className="text-sm font-medium">
                  {storageUsed} GB of {storageTotal} GB used
                </span>
              </div>
              <Progress value={(storageUsed / storageTotal) * 100} className="h-3" />
            </div>

            <div className="space-y-4">
              <div className="flex items-center justify-between p-3 bg-medical-accent-light rounded-lg">
                <div className="flex items-center gap-2">
                  <HardDrive className="h-4 w-4 text-medical-accent" />
                  <span className="font-medium">Medical Images</span>
                </div>
                <span className="text-sm">1.8 GB</span>
              </div>

              <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <div className="flex items-center gap-2">
                  <HardDrive className="h-4 w-4 text-gray-600" />
                  <span className="font-medium">Documents</span>
                </div>
                <span className="text-sm">0.3 GB</span>
              </div>

              <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <div className="flex items-center gap-2">
                  <HardDrive className="h-4 w-4 text-gray-600" />
                  <span className="font-medium">App Data</span>
                </div>
                <span className="text-sm">0.2 GB</span>
              </div>
            </div>

            <div className="flex justify-end">
              <Button onClick={() => onOpenChange(false)}>
                Close
              </Button>
            </div>
          </div>
        );

      case 'cache':
        return (
          <div className="space-y-6">
            <div className="flex items-center gap-3 p-4 bg-medical-accent-light rounded-lg">
              <Trash2 className="h-6 w-6 text-medical-accent" />
              <div>
                <h3 className="font-medium">Clear Cache</h3>
                <p className="text-sm text-muted-foreground">
                  Free up space by clearing temporary files
                </p>
              </div>
            </div>

            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="font-medium">Cache Size</span>
                <span className="text-medical-accent font-medium">{cacheSize}</span>
              </div>
              <p className="text-sm text-muted-foreground">
                Cached files help the app load faster but can be safely removed to free up storage space.
              </p>
            </div>

            <div className="p-3 bg-blue-50 border border-blue-200 rounded-lg">
              <p className="text-sm text-blue-800">
                <strong>Note:</strong> Clearing cache may cause the app to load more slowly 
                the first time you use it after clearing.
              </p>
            </div>

            <div className="flex justify-end space-x-2">
              <Button variant="outline" onClick={() => onOpenChange(false)}>
                Cancel
              </Button>
              <AlertDialog>
                <AlertDialogTrigger asChild>
                  <Button variant="destructive">Clear Cache</Button>
                </AlertDialogTrigger>
                <AlertDialogContent>
                  <AlertDialogHeader>
                    <AlertDialogTitle>Clear Cache?</AlertDialogTitle>
                    <AlertDialogDescription>
                      This will clear {cacheSize} of cached data. This action cannot be undone.
                    </AlertDialogDescription>
                  </AlertDialogHeader>
                  <AlertDialogFooter>
                    <AlertDialogCancel>Cancel</AlertDialogCancel>
                    <AlertDialogAction onClick={handleClearCache}>
                      Clear Cache
                    </AlertDialogAction>
                  </AlertDialogFooter>
                </AlertDialogContent>
              </AlertDialog>
            </div>
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>{getTitle()}</DialogTitle>
        </DialogHeader>
        {renderContent()}
      </DialogContent>
    </Dialog>
  );
};