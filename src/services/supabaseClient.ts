import { createClient } from '@supabase/supabase-js';

const supabaseUrl = 'https://vplnxxpdfevhxddffcrl.supabase.co';
const supabaseKey = import.meta.env.VITE_SUPABASE_KEY;

if (!supabaseKey) {
  throw new Error('Missing Supabase key environment variable');
}

export const supabase = createClient(supabaseUrl, supabaseKey);
