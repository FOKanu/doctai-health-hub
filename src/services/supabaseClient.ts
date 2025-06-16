
import { createClient } from '@supabase/supabase-js';

const supabaseUrl = 'https://vplnxxpdfevhxddffcrl.supabase.co';
const supabaseKey = import.meta.env.VITE_SUPABASE_KEY;

let supabase: any;

if (!supabaseKey) {
  if (import.meta.env.DEV) {
    console.warn('Missing Supabase key in development environment. Using mock implementation.');
    // Mock Supabase client for development when key is missing
    supabase = {
      storage: {
        from: (bucket: string) => ({
          upload: async (path: string, file: File | Blob, options?: any) => {
            console.log(`Mock upload to ${bucket}/${path}`);
            return { data: { path }, error: null };
          },
          getPublicUrl: (path: string) => ({
            data: { publicUrl: `https://mock-storage.supabase.co/${path}` }
          }),
          remove: async (paths: string[]) => {
            console.log('Mock delete:', paths);
            return { data: null, error: null };
          }
        })
      },
      from: (table: string) => ({
        insert: (data: any) => ({
          select: () => ({
            single: async () => {
              console.log(`Mock insert into ${table}:`, data);
              return { data: { id: Date.now().toString(), ...data }, error: null };
            }
          })
        }),
        select: (columns = '*') => ({
          eq: (column: string, value: any) => ({
            single: async () => {
              console.log(`Mock select from ${table} where ${column} = ${value}`);
              return { data: null, error: null };
            },
            order: (column: string, options?: any) => ({
              then: async (callback: any) => {
                console.log(`Mock select from ${table} ordered by ${column}`);
                return callback({ data: [], error: null });
              }
            })
          })
        }),
        update: (data: any) => ({
          eq: (column: string, value: any) => ({
            select: () => ({
              single: async () => {
                console.log(`Mock update ${table} where ${column} = ${value}:`, data);
                return { data: { ...data }, error: null };
              }
            })
          })
        }),
        delete: () => ({
          eq: (column: string, value: any) => ({
            then: async (callback: any) => {
              console.log(`Mock delete from ${table} where ${column} = ${value}`);
              return callback({ data: null, error: null });
            }
          })
        })
      }),
      auth: {
        signUp: async (credentials: any) => {
          console.log('Mock sign up:', credentials);
          return { data: { user: null, session: null }, error: null };
        },
        signInWithPassword: async (credentials: any) => {
          console.log('Mock sign in:', credentials);
          return { data: { user: null, session: null }, error: null };
        },
        signOut: async () => {
          console.log('Mock sign out');
          return { error: null };
        },
        getSession: async () => {
          console.log('Mock get session');
          return { data: { session: null }, error: null };
        }
      }
    };
  } else {
    throw new Error('Missing Supabase key environment variable');
  }
} else {
  supabase = createClient(supabaseUrl, supabaseKey);
}

export { supabase };
