import { supabase } from './supabaseClient';

export interface ImageMetadata {
  id: string;
  url: string;
  created_at: string;
  user_id: string;
  type: 'skin_lesion' | 'ct_scan' | 'mri' | 'xray' | 'eeg';
  analysis_result?: any;
  metadata?: {
    size: number;
    width: number;
    height: number;
    format: string;
    device_info?: {
      model: string;
      os: string;
      browser: string;
    };
  };
}

export const storageService = {
  async uploadImage(
    file: File | Blob,
    type: ImageMetadata['type'],
    metadata: Partial<ImageMetadata['metadata']> = {}
  ): Promise<ImageMetadata> {
    try {
      // Generate a unique file name
      const fileExt = file instanceof File ? file.name.split('.').pop() : 'jpg';
      const fileName = `${type}/${Date.now()}.${fileExt}`;

      // Upload file to Supabase Storage
      const { data: uploadData, error: uploadError } = await supabase.storage
        .from('medical-images')
        .upload(fileName, file, {
          cacheControl: '3600',
          upsert: false
        });

      if (uploadError) throw uploadError;

      // Get the public URL
      const { data: { publicUrl } } = supabase.storage
        .from('medical-images')
        .getPublicUrl(fileName);

      // Create metadata record in the database
      const { data: metadataRecord, error: metadataError } = await supabase
        .from('image_metadata')
        .insert({
          url: publicUrl,
          type,
          metadata: {
            ...metadata,
            size: file instanceof File ? file.size : 0,
            created_at: new Date().toISOString()
          }
        })
        .select()
        .single();

      if (metadataError) throw metadataError;

      return metadataRecord;
    } catch (error) {
      console.error('Error uploading image:', error);
      throw error;
    }
  },

  async getImageMetadata(imageId: string): Promise<ImageMetadata> {
    const { data, error } = await supabase
      .from('image_metadata')
      .select('*')
      .eq('id', imageId)
      .single();

    if (error) throw error;
    return data;
  },

  async getUserImages(userId: string): Promise<ImageMetadata[]> {
    const { data, error } = await supabase
      .from('image_metadata')
      .select('*')
      .eq('user_id', userId)
      .order('created_at', { ascending: false });

    if (error) throw error;
    return data;
  },

  async deleteImage(imageId: string): Promise<void> {
    // Get the image metadata first
    const { data: image } = await supabase
      .from('image_metadata')
      .select('url')
      .eq('id', imageId)
      .single();

    if (image) {
      // Extract the file path from the URL
      const filePath = image.url.split('/').pop();

      // Delete from storage
      await supabase.storage
        .from('medical-images')
        .remove([filePath]);

      // Delete metadata record
      await supabase
        .from('image_metadata')
        .delete()
        .eq('id', imageId);
    }
  },

  async updateImageMetadata(
    imageId: string,
    updates: Partial<ImageMetadata>
  ): Promise<ImageMetadata> {
    const { data, error } = await supabase
      .from('image_metadata')
      .update(updates)
      .eq('id', imageId)
      .select()
      .single();

    if (error) throw error;
    return data;
  }
};
