
-- Create enum for image types
CREATE TYPE image_type AS ENUM ('skin_lesion', 'ct_scan', 'mri', 'xray', 'eeg');

-- Create image_metadata table
CREATE TABLE image_metadata (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    url TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    type image_type NOT NULL,
    analysis_result JSONB,
    metadata JSONB,
    CONSTRAINT valid_url CHECK (url ~ '^https?://')
);

-- Create indexes
CREATE INDEX idx_image_metadata_user_id ON image_metadata(user_id);
CREATE INDEX idx_image_metadata_type ON image_metadata(type);
CREATE INDEX idx_image_metadata_created_at ON image_metadata(created_at);

-- Set up Row Level Security (RLS)
ALTER TABLE image_metadata ENABLE ROW LEVEL SECURITY;

-- Create policies
CREATE POLICY "Users can view their own images"
    ON image_metadata
    FOR SELECT
    USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own images"
    ON image_metadata
    FOR INSERT
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own images"
    ON image_metadata
    FOR UPDATE
    USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own images"
    ON image_metadata
    FOR DELETE
    USING (auth.uid() = user_id);

-- Create storage bucket for medical images
INSERT INTO storage.buckets (id, name, public)
VALUES ('medical-images', 'medical-images', false);

-- Set up storage policies
CREATE POLICY "Users can upload their own images"
    ON storage.objects
    FOR INSERT
    WITH CHECK (
        bucket_id = 'medical-images' AND
        auth.uid() = owner
    );

CREATE POLICY "Users can view their own images"
    ON storage.objects
    FOR SELECT
    USING (
        bucket_id = 'medical-images' AND
        auth.uid() = owner
    );

CREATE POLICY "Users can update their own images"
    ON storage.objects
    FOR UPDATE
    USING (
        bucket_id = 'medical-images' AND
        auth.uid() = owner
    );

CREATE POLICY "Users can delete their own images"
    ON storage.objects
    FOR DELETE
    USING (
        bucket_id = 'medical-images' AND
        auth.uid() = owner
    );
