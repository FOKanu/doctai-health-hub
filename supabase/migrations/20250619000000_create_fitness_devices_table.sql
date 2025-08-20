-- DoctAI Health Hub - Fitness Devices Table Migration
-- This creates the table to store connected fitness device information

-- Create fitness devices table
CREATE TABLE IF NOT EXISTS fitness_devices (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  type TEXT NOT NULL CHECK (type IN ('google_fit', 'fitbit', 'apple_health', 'samsung_health')),
  is_connected BOOLEAN DEFAULT true,
  last_sync TIMESTAMP WITH TIME ZONE,
  metrics TEXT[] DEFAULT '{}',
  access_token TEXT,
  refresh_token TEXT,
  token_expires_at TIMESTAMP WITH TIME ZONE,
  device_user_id TEXT, -- For Fitbit user ID
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_fitness_devices_user_id ON fitness_devices(user_id);
CREATE INDEX IF NOT EXISTS idx_fitness_devices_type ON fitness_devices(type);
CREATE INDEX IF NOT EXISTS idx_fitness_devices_connected ON fitness_devices(is_connected);

-- Create updated_at trigger
CREATE OR REPLACE FUNCTION update_fitness_devices_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_fitness_devices_updated_at
  BEFORE UPDATE ON fitness_devices
  FOR EACH ROW
  EXECUTE FUNCTION update_fitness_devices_updated_at();

-- Enable RLS
ALTER TABLE fitness_devices ENABLE ROW LEVEL SECURITY;

-- RLS Policies for fitness_devices table
CREATE POLICY "Users can view their own fitness devices"
  ON fitness_devices
  FOR SELECT
  USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own fitness devices"
  ON fitness_devices
  FOR INSERT
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own fitness devices"
  ON fitness_devices
  FOR UPDATE
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can delete their own fitness devices"
  ON fitness_devices
  FOR DELETE
  USING (auth.uid() = user_id);

-- Add comments for documentation
COMMENT ON TABLE fitness_devices IS 'Stores connected fitness device information for health data integration';
COMMENT ON COLUMN fitness_devices.type IS 'Type of fitness device/service (google_fit, fitbit, apple_health, samsung_health)';
COMMENT ON COLUMN fitness_devices.metrics IS 'Array of available metrics for this device';
COMMENT ON COLUMN fitness_devices.access_token IS 'OAuth access token for the device API';
COMMENT ON COLUMN fitness_devices.refresh_token IS 'OAuth refresh token for the device API';
COMMENT ON COLUMN fitness_devices.device_user_id IS 'User ID from the device service (e.g., Fitbit user ID)';
COMMENT ON COLUMN fitness_devices.metadata IS 'Additional device-specific metadata stored as JSON';
