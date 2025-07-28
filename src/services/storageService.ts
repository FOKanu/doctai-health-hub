
import { supabase } from '@/integrations/supabase/client';
import type { Database } from '@/integrations/supabase/types';

type DbImageMetadata = Database['public']['Tables']['image_metadata']['Row'];
type DbImageMetadataInsert = Database['public']['Tables']['image_metadata']['Insert'];

export interface ImageMetadata {
  id: string;
  url: string;
  created_at: string;
  user_id: string;
  type: 'skin_lesion' | 'ct_scan' | 'mri' | 'xray' | 'eeg';
  analysis_result?: Record<string, unknown>;
  metadata?: {
    size: number;
    width?: number;
    height?: number;
    format?: string;
    device_info?: {
      model: string;
      os: string;
      browser: string;
    };
  };
}

// Helper function to convert database row to ImageMetadata
const convertDbRowToImageMetadata = (row: DbImageMetadata): ImageMetadata => {
  return {
    id: row.id,
    url: row.url,
    created_at: row.created_at,
    user_id: row.user_id || '',
    type: row.type,
    analysis_result: row.analysis_result,
    metadata: row.metadata as ImageMetadata['metadata']
  };
};

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
      const insertData: DbImageMetadataInsert = {
        url: publicUrl,
        type,
        metadata: {
          ...metadata,
          size: file instanceof File ? file.size : 0,
        }
      };

      const { data: metadataRecord, error: metadataError } = await supabase
        .from('image_metadata')
        .insert(insertData)
        .select()
        .single();

      if (metadataError) throw metadataError;

      return convertDbRowToImageMetadata(metadataRecord);
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
    return convertDbRowToImageMetadata(data);
  },

  async getUserImages(userId: string): Promise<ImageMetadata[]> {
    const { data, error } = await supabase
      .from('image_metadata')
      .select('*')
      .eq('user_id', userId)
      .order('created_at', { ascending: false });

    if (error) throw error;
    return data.map(convertDbRowToImageMetadata);
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
      const urlParts = image.url.split('/');
      const filePath = urlParts.slice(-2).join('/'); // Get type/filename

      // Delete from storage
      const { error: storageError } = await supabase.storage
        .from('medical-images')
        .remove([filePath]);

      if (storageError) {
        console.error('Error deleting from storage:', storageError);
      }

      // Delete metadata record
      const { error: dbError } = await supabase
        .from('image_metadata')
        .delete()
        .eq('id', imageId);

      if (dbError) {
        console.error('Error deleting metadata:', dbError);
        throw dbError;
      }
    }
  },

  async updateImageMetadata(
    imageId: string,
    updates: Partial<Pick<ImageMetadata, 'analysis_result' | 'metadata'>>
  ): Promise<ImageMetadata> {
    const { data, error } = await supabase
      .from('image_metadata')
      .update(updates)
      .eq('id', imageId)
      .select()
      .single();

    if (error) throw error;
    return convertDbRowToImageMetadata(data);
  }
};
