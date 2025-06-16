import { supabase } from '@/integrations/supabase/client';

const SCANS_TABLE_NAME = 'scans';
const SPOTS_TABLE_NAME = 'spots';

export interface PredictionResult {
  prediction: 'benign' | 'malignant';
  confidence: number;
  probabilities: {
    benign: number;
    malignant: number;
  };
  uploadedImageUrl: string;
  timestamp: string;
  imageId?: string;
}

export async function analyzePrediction(imageUri: string): Promise<PredictionResult> {
  try {
    // Upload image to Supabase Storage first
    const publicImageUrl = await uploadImageToSupabase(imageUri);

    const formData = new FormData();
    let formDataType = 'image/jpeg';
    if (imageUri.toLowerCase().endsWith('.png')) {
      formDataType = 'image/png';
    } else if (imageUri.toLowerCase().endsWith('.webp')) {
      formDataType = 'image/webp';
    }

    // Convert imageUri to blob first
    const response = await fetch(imageUri);
    const blob = await response.blob();
    
    formData.append('file', blob, 'photo.jpg');

    const apiResponse = await fetch(`${process.env.NEXT_PUBLIC_SCORING_API_URL}/predict`, {
      method: 'POST',
      body: formData,
    });

    const apiResponseJson = await apiResponse.json();
    if (apiResponseJson.status === 'error') {
      throw new Error(apiResponseJson.error || 'Prediction failed');
    }

    const prediction = apiResponseJson.predicted_class === 0 ? 'benign' : 'malignant';
    const malignantProbability = apiResponseJson.predicted_class === 1 ? apiResponseJson.confidence : (1 - apiResponseJson.confidence);
    const benignProbability = 1 - malignantProbability;

    return {
      prediction,
      confidence: apiResponseJson.confidence,
      probabilities: {
        benign: benignProbability,
        malignant: malignantProbability,
      },
      uploadedImageUrl: publicImageUrl,
      timestamp: new Date().toISOString(),
    };
  } catch (error) {
    console.error('Error predicting skin lesion:', error);
    throw error;
  }
}

export async function savePredictionToSupabase(result: PredictionResult, imageUri: string, spotId?: string) {
  try {
    const { data, error } = await supabase
      .from(SCANS_TABLE_NAME)
      .insert([
        {
          spot_id: spotId,
          image_url: result.uploadedImageUrl,
          prediction: result.prediction,
          confidence: result.confidence,
          benign_probability: result.probabilities.benign,
          malignant_probability: result.probabilities.malignant,
          scanned_at: result.timestamp,
        },
      ]);

    if (error) {
      console.error('Error saving prediction to Supabase:', {
        error,
        details: error.details,
        hint: error.hint,
        code: error.code
      });
      throw error;
    }

    return data;
  } catch (error) {
    console.error('Error in savePredictionToSupabase:', error);
    throw error;
  }
}

export async function getPredictionHistory(userId?: string) {
  let query = supabase
    .from(SCANS_TABLE_NAME)
    .select(`*,\
      spot:spot_id (
        user_id
      )`);

  if (userId) {
    query = query.eq('spot.user_id', userId);
  }

  const { data, error } = await query;
  if (error) {
    console.error('Error fetching prediction history:', error);
    return [];
  }
  return data;
}

async function uploadImageToSupabase(imageUri: string): Promise<string> {
  try {
    const response = await fetch(imageUri);
    const blob = await response.blob();
    const filename = `skin-scans/${Date.now()}-${Math.random().toString(36).substring(7)}.jpg`;

    const { data, error } = await supabase.storage
      .from('medical-images')
      .upload(filename, blob, {
        contentType: 'image/jpeg',
        cacheControl: '3600',
      });

    if (error) throw error;

    const { data: { publicUrl } } = supabase.storage
      .from('medical-images')
      .getPublicUrl(filename);

    return publicUrl;
  } catch (error) {
    console.error('Error uploading image to Supabase:', error);
    throw error;
  }
}
