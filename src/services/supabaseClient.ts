
// Mock Supabase client for development
// In a real app, this would contain the actual Supabase configuration

export const supabase = {
  storage: {
    from: (bucket: string) => ({
      upload: async (path: string, file: File | Blob, options?: Record<string, unknown>) => {
        // Mock upload implementation
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
    insert: (data: unknown) => ({
      select: () => ({
        single: async () => {
          console.log(`Mock insert into ${table}:`, data);
          return { data: { id: Date.now().toString(), ...data }, error: null };
        }
      })
    }),
    select: (columns = '*') => ({
      eq: (column: string, value: React.SyntheticEvent) => ({
        single: async () => {
          console.log(`Mock select from ${table} where ${column} = ${value}`);
          return { data: null, error: null };
        },
        order: (column: string, options?: Record<string, unknown>) => ({
          then: async (callback: () => void) => {
            console.log(`Mock select from ${table} ordered by ${column}`);
            return callback({ data: [], error: null });
          }
        })
      })
    }),
    update: (data: unknown) => ({
      eq: (column: string, value: React.SyntheticEvent) => ({
        select: () => ({
          single: async () => {
            console.log(`Mock update ${table} where ${column} = ${value}:`, data);
            return { data: { ...data }, error: null };
          }
        })
      })
    }),
    delete: () => ({
      eq: (column: string, value: React.SyntheticEvent) => ({
        then: async (callback: () => void) => {
          console.log(`Mock delete from ${table} where ${column} = ${value}`);
          return callback({ data: null, error: null });
        }
      })
    })
  })
};
