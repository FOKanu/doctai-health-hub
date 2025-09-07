-- Create patient_profiles table for detailed patient information
CREATE TABLE public.patient_profiles (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID NOT NULL UNIQUE REFERENCES auth.users(id) ON DELETE CASCADE,
  
  -- Basic Information
  date_of_birth DATE,
  phone_number TEXT,
  address TEXT,
  city TEXT,
  state TEXT,
  zip_code TEXT,
  
  -- Emergency Contact
  emergency_contact_name TEXT,
  emergency_contact_phone TEXT,
  emergency_contact_relation TEXT,
  
  -- Medical History
  allergies TEXT[],
  current_medications TEXT[],
  medical_conditions TEXT[],
  surgeries TEXT[],
  family_history TEXT,
  
  -- Insurance Information
  has_insurance BOOLEAN DEFAULT FALSE,
  insurance_provider TEXT,
  policy_number TEXT,
  group_number TEXT,
  subscriber_name TEXT,
  subscriber_dob DATE,
  
  -- Metadata
  profile_completed_at TIMESTAMP WITH TIME ZONE,
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

-- Enable Row Level Security
ALTER TABLE public.patient_profiles ENABLE ROW LEVEL SECURITY;

-- Create RLS policies for patient profiles
CREATE POLICY "Users can view their own patient profile"
ON public.patient_profiles
FOR SELECT
USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own patient profile"
ON public.patient_profiles
FOR INSERT
WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own patient profile"
ON public.patient_profiles
FOR UPDATE
USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own patient profile"
ON public.patient_profiles
FOR DELETE
USING (auth.uid() = user_id);

-- Create trigger for automatic timestamp updates
CREATE TRIGGER update_patient_profiles_updated_at
  BEFORE UPDATE ON public.patient_profiles
  FOR EACH ROW
  EXECUTE FUNCTION public.update_updated_at_column();

-- Create indexes for better performance
CREATE INDEX idx_patient_profiles_user_id ON public.patient_profiles(user_id);
CREATE INDEX idx_patient_profiles_has_insurance ON public.patient_profiles(has_insurance);
CREATE INDEX idx_patient_profiles_created_at ON public.patient_profiles(created_at);