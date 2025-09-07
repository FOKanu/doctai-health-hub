export type Json =
  | string
  | number
  | boolean
  | null
  | { [key: string]: Json | undefined }
  | Json[]

export type Database = {
  // Allows to automatically instantiate createClient with right options
  // instead of createClient<Database, { PostgrestVersion: 'XX' }>(URL, KEY)
  __InternalSupabase: {
    PostgrestVersion: "12.2.3 (519615d)"
  }
  public: {
    Tables: {
      data_access_audit: {
        Row: {
          accessed_at: string | null
          id: number
          ip_address: unknown | null
          operation: string
          record_id: string | null
          table_name: string
          user_agent: string | null
          user_id: string | null
        }
        Insert: {
          accessed_at?: string | null
          id?: never
          ip_address?: unknown | null
          operation: string
          record_id?: string | null
          table_name: string
          user_agent?: string | null
          user_id?: string | null
        }
        Update: {
          accessed_at?: string | null
          id?: never
          ip_address?: unknown | null
          operation?: string
          record_id?: string | null
          table_name?: string
          user_agent?: string | null
          user_id?: string | null
        }
        Relationships: []
      }
      health_metrics_timeseries: {
        Row: {
          accuracy_score: number | null
          anonymization_timestamp: string | null
          created_at: string | null
          device_source: string | null
          id: string
          is_anonymized: boolean | null
          last_accessed_at: string | null
          metadata: Json | null
          metric_type: string | null
          recorded_at: string | null
          retention_expires_at: string | null
          user_id: string | null
          value: Json | null
        }
        Insert: {
          accuracy_score?: number | null
          anonymization_timestamp?: string | null
          created_at?: string | null
          device_source?: string | null
          id: string
          is_anonymized?: boolean | null
          last_accessed_at?: string | null
          metadata?: Json | null
          metric_type?: string | null
          recorded_at?: string | null
          retention_expires_at?: string | null
          user_id?: string | null
          value?: Json | null
        }
        Update: {
          accuracy_score?: number | null
          anonymization_timestamp?: string | null
          created_at?: string | null
          device_source?: string | null
          id?: string
          is_anonymized?: boolean | null
          last_accessed_at?: string | null
          metadata?: Json | null
          metric_type?: string | null
          recorded_at?: string | null
          retention_expires_at?: string | null
          user_id?: string | null
          value?: Json | null
        }
        Relationships: []
      }
      image_metadata: {
        Row: {
          analysis_result: Json | null
          anonymization_timestamp: string | null
          created_at: string | null
          id: string
          is_anonymized: boolean | null
          last_accessed_at: string | null
          metadata: Json | null
          retention_expires_at: string | null
          type: string | null
          updated_at: string | null
          url: string | null
          user_id: string | null
        }
        Insert: {
          analysis_result?: Json | null
          anonymization_timestamp?: string | null
          created_at?: string | null
          id: string
          is_anonymized?: boolean | null
          last_accessed_at?: string | null
          metadata?: Json | null
          retention_expires_at?: string | null
          type?: string | null
          updated_at?: string | null
          url?: string | null
          user_id?: string | null
        }
        Update: {
          analysis_result?: Json | null
          anonymization_timestamp?: string | null
          created_at?: string | null
          id?: string
          is_anonymized?: boolean | null
          last_accessed_at?: string | null
          metadata?: Json | null
          retention_expires_at?: string | null
          type?: string | null
          updated_at?: string | null
          url?: string | null
          user_id?: string | null
        }
        Relationships: []
      }
      patient_profiles: {
        Row: {
          address: string | null
          allergies: string[] | null
          city: string | null
          created_at: string
          current_medications: string[] | null
          date_of_birth: string | null
          emergency_contact_name: string | null
          emergency_contact_phone: string | null
          emergency_contact_relation: string | null
          family_history: string | null
          group_number: string | null
          has_insurance: boolean | null
          id: string
          insurance_provider: string | null
          medical_conditions: string[] | null
          phone_number: string | null
          policy_number: string | null
          profile_completed_at: string | null
          state: string | null
          subscriber_dob: string | null
          subscriber_name: string | null
          surgeries: string[] | null
          updated_at: string
          user_id: string
          zip_code: string | null
        }
        Insert: {
          address?: string | null
          allergies?: string[] | null
          city?: string | null
          created_at?: string
          current_medications?: string[] | null
          date_of_birth?: string | null
          emergency_contact_name?: string | null
          emergency_contact_phone?: string | null
          emergency_contact_relation?: string | null
          family_history?: string | null
          group_number?: string | null
          has_insurance?: boolean | null
          id?: string
          insurance_provider?: string | null
          medical_conditions?: string[] | null
          phone_number?: string | null
          policy_number?: string | null
          profile_completed_at?: string | null
          state?: string | null
          subscriber_dob?: string | null
          subscriber_name?: string | null
          surgeries?: string[] | null
          updated_at?: string
          user_id: string
          zip_code?: string | null
        }
        Update: {
          address?: string | null
          allergies?: string[] | null
          city?: string | null
          created_at?: string
          current_medications?: string[] | null
          date_of_birth?: string | null
          emergency_contact_name?: string | null
          emergency_contact_phone?: string | null
          emergency_contact_relation?: string | null
          family_history?: string | null
          group_number?: string | null
          has_insurance?: boolean | null
          id?: string
          insurance_provider?: string | null
          medical_conditions?: string[] | null
          phone_number?: string | null
          policy_number?: string | null
          profile_completed_at?: string | null
          state?: string | null
          subscriber_dob?: string | null
          subscriber_name?: string | null
          surgeries?: string[] | null
          updated_at?: string
          user_id?: string
          zip_code?: string | null
        }
        Relationships: []
      }
      patient_timelines: {
        Row: {
          anonymization_timestamp: string | null
          baseline_date: string | null
          condition_type: string | null
          confidence_score: number | null
          created_at: string | null
          id: string
          is_anonymized: boolean | null
          last_accessed_at: string | null
          metadata: Json | null
          notes: string | null
          retention_expires_at: string | null
          severity_score: number | null
          status: string | null
          updated_at: string | null
          user_id: string | null
        }
        Insert: {
          anonymization_timestamp?: string | null
          baseline_date?: string | null
          condition_type?: string | null
          confidence_score?: number | null
          created_at?: string | null
          id: string
          is_anonymized?: boolean | null
          last_accessed_at?: string | null
          metadata?: Json | null
          notes?: string | null
          retention_expires_at?: string | null
          severity_score?: number | null
          status?: string | null
          updated_at?: string | null
          user_id?: string | null
        }
        Update: {
          anonymization_timestamp?: string | null
          baseline_date?: string | null
          condition_type?: string | null
          confidence_score?: number | null
          created_at?: string | null
          id?: string
          is_anonymized?: boolean | null
          last_accessed_at?: string | null
          metadata?: Json | null
          notes?: string | null
          retention_expires_at?: string | null
          severity_score?: number | null
          status?: string | null
          updated_at?: string | null
          user_id?: string | null
        }
        Relationships: []
      }
      profiles: {
        Row: {
          avatar_url: string | null
          bio: string | null
          created_at: string | null
          display_name: string | null
          id: string
          role: string | null
          updated_at: string | null
          user_id: string
        }
        Insert: {
          avatar_url?: string | null
          bio?: string | null
          created_at?: string | null
          display_name?: string | null
          id?: string
          role?: string | null
          updated_at?: string | null
          user_id: string
        }
        Update: {
          avatar_url?: string | null
          bio?: string | null
          created_at?: string | null
          display_name?: string | null
          id?: string
          role?: string | null
          updated_at?: string | null
          user_id?: string
        }
        Relationships: []
      }
      provider_analytics: {
        Row: {
          created_at: string
          date_range_end: string
          date_range_start: string
          generated_at: string
          id: string
          provider_id: string
          report_data: Json
          report_type: string
          updated_at: string
        }
        Insert: {
          created_at?: string
          date_range_end: string
          date_range_start: string
          generated_at?: string
          id?: string
          provider_id: string
          report_data: Json
          report_type: string
          updated_at?: string
        }
        Update: {
          created_at?: string
          date_range_end?: string
          date_range_start?: string
          generated_at?: string
          id?: string
          provider_id?: string
          report_data?: Json
          report_type?: string
          updated_at?: string
        }
        Relationships: []
      }
      provider_appointments: {
        Row: {
          appointment_date: string
          appointment_type: string
          created_at: string
          duration_minutes: number
          id: string
          notes: string | null
          patient_id: string | null
          provider_id: string
          reason: string | null
          status: string
          updated_at: string
        }
        Insert: {
          appointment_date: string
          appointment_type: string
          created_at?: string
          duration_minutes?: number
          id?: string
          notes?: string | null
          patient_id?: string | null
          provider_id: string
          reason?: string | null
          status?: string
          updated_at?: string
        }
        Update: {
          appointment_date?: string
          appointment_type?: string
          created_at?: string
          duration_minutes?: number
          id?: string
          notes?: string | null
          patient_id?: string | null
          provider_id?: string
          reason?: string | null
          status?: string
          updated_at?: string
        }
        Relationships: []
      }
      provider_patients: {
        Row: {
          allergies: string[] | null
          contact_info: Json | null
          created_at: string
          current_medications: Json | null
          date_of_birth: string | null
          gender: string | null
          id: string
          last_visit_date: string | null
          medical_history: Json | null
          next_appointment_date: string | null
          patient_mrn: string
          patient_name: string
          provider_id: string
          risk_level: string | null
          updated_at: string
        }
        Insert: {
          allergies?: string[] | null
          contact_info?: Json | null
          created_at?: string
          current_medications?: Json | null
          date_of_birth?: string | null
          gender?: string | null
          id?: string
          last_visit_date?: string | null
          medical_history?: Json | null
          next_appointment_date?: string | null
          patient_mrn: string
          patient_name: string
          provider_id: string
          risk_level?: string | null
          updated_at?: string
        }
        Update: {
          allergies?: string[] | null
          contact_info?: Json | null
          created_at?: string
          current_medications?: Json | null
          date_of_birth?: string | null
          gender?: string | null
          id?: string
          last_visit_date?: string | null
          medical_history?: Json | null
          next_appointment_date?: string | null
          patient_mrn?: string
          patient_name?: string
          provider_id?: string
          risk_level?: string | null
          updated_at?: string
        }
        Relationships: []
      }
      provider_settings: {
        Row: {
          ai_settings: Json | null
          calendar_settings: Json | null
          created_at: string
          dashboard_layout: Json | null
          id: string
          notification_preferences: Json | null
          privacy_settings: Json | null
          provider_id: string
          theme_preferences: Json | null
          updated_at: string
        }
        Insert: {
          ai_settings?: Json | null
          calendar_settings?: Json | null
          created_at?: string
          dashboard_layout?: Json | null
          id?: string
          notification_preferences?: Json | null
          privacy_settings?: Json | null
          provider_id: string
          theme_preferences?: Json | null
          updated_at?: string
        }
        Update: {
          ai_settings?: Json | null
          calendar_settings?: Json | null
          created_at?: string
          dashboard_layout?: Json | null
          id?: string
          notification_preferences?: Json | null
          privacy_settings?: Json | null
          provider_id?: string
          theme_preferences?: Json | null
          updated_at?: string
        }
        Relationships: []
      }
      risk_progressions: {
        Row: {
          anonymization_timestamp: string | null
          condition_type: string | null
          confidence_score: number | null
          created_at: string | null
          factors: Json | null
          id: string
          is_anonymized: boolean | null
          last_accessed_at: string | null
          metadata: Json | null
          predicted_date: string | null
          probability: number | null
          recorded_at: string | null
          retention_expires_at: string | null
          risk_level: string | null
          user_id: string | null
        }
        Insert: {
          anonymization_timestamp?: string | null
          condition_type?: string | null
          confidence_score?: number | null
          created_at?: string | null
          factors?: Json | null
          id: string
          is_anonymized?: boolean | null
          last_accessed_at?: string | null
          metadata?: Json | null
          predicted_date?: string | null
          probability?: number | null
          recorded_at?: string | null
          retention_expires_at?: string | null
          risk_level?: string | null
          user_id?: string | null
        }
        Update: {
          anonymization_timestamp?: string | null
          condition_type?: string | null
          confidence_score?: number | null
          created_at?: string | null
          factors?: Json | null
          id?: string
          is_anonymized?: boolean | null
          last_accessed_at?: string | null
          metadata?: Json | null
          predicted_date?: string | null
          probability?: number | null
          recorded_at?: string | null
          retention_expires_at?: string | null
          risk_level?: string | null
          user_id?: string | null
        }
        Relationships: []
      }
      rls_policies: {
        Row: {
          description: string | null
          operation: string | null
          policy_definition: string | null
          policy_name: string | null
          table_name: string | null
        }
        Insert: {
          description?: string | null
          operation?: string | null
          policy_definition?: string | null
          policy_name?: string | null
          table_name?: string | null
        }
        Update: {
          description?: string | null
          operation?: string | null
          policy_definition?: string | null
          policy_name?: string | null
          table_name?: string | null
        }
        Relationships: []
      }
      scan_sequences: {
        Row: {
          analysis_type: string | null
          anonymization_timestamp: string | null
          baseline_image_id: string | null
          confidence_score: number | null
          created_at: string | null
          findings: Json | null
          id: string
          image_ids: string | null
          is_anonymized: boolean | null
          last_accessed_at: string | null
          metadata: Json | null
          progression_score: number | null
          recommendations: string | null
          retention_expires_at: string | null
          sequence_name: string | null
          updated_at: string | null
          user_id: string | null
        }
        Insert: {
          analysis_type?: string | null
          anonymization_timestamp?: string | null
          baseline_image_id?: string | null
          confidence_score?: number | null
          created_at?: string | null
          findings?: Json | null
          id: string
          image_ids?: string | null
          is_anonymized?: boolean | null
          last_accessed_at?: string | null
          metadata?: Json | null
          progression_score?: number | null
          recommendations?: string | null
          retention_expires_at?: string | null
          sequence_name?: string | null
          updated_at?: string | null
          user_id?: string | null
        }
        Update: {
          analysis_type?: string | null
          anonymization_timestamp?: string | null
          baseline_image_id?: string | null
          confidence_score?: number | null
          created_at?: string | null
          findings?: Json | null
          id?: string
          image_ids?: string | null
          is_anonymized?: boolean | null
          last_accessed_at?: string | null
          metadata?: Json | null
          progression_score?: number | null
          recommendations?: string | null
          retention_expires_at?: string | null
          sequence_name?: string | null
          updated_at?: string | null
          user_id?: string | null
        }
        Relationships: []
      }
      "Table setup verification": {
        Row: {
          check_type: string | null
          description: string | null
          expected_status: string | null
          expected_value: string | null
          table_name: string | null
        }
        Insert: {
          check_type?: string | null
          description?: string | null
          expected_status?: string | null
          expected_value?: string | null
          table_name?: string | null
        }
        Update: {
          check_type?: string | null
          description?: string | null
          expected_status?: string | null
          expected_value?: string | null
          table_name?: string | null
        }
        Relationships: []
      }
    }
    Views: {
      [_ in never]: never
    }
    Functions: {
      cleanup_expired_data: {
        Args: Record<PropertyKey, never>
        Returns: undefined
      }
      create_comprehensive_rls_policies: {
        Args: { p_table_name: string }
        Returns: undefined
      }
      debug_rls_policy: {
        Args: Record<PropertyKey, never>
        Returns: {
          current_user_id: string
          table_name: string
          user_id_type: string
        }[]
      }
      delete_user_data: {
        Args: { p_user_id: string }
        Returns: undefined
      }
      export_user_data: {
        Args: { p_user_id: string }
        Returns: {
          record_data: Json
          table_name: string
        }[]
      }
      secure_user_data_export: {
        Args: { p_user_id: string }
        Returns: {
          record_data: Json
          table_name: string
        }[]
      }
    }
    Enums: {
      image_type: "skin_lesion" | "ct_scan" | "mri" | "xray" | "eeg"
    }
    CompositeTypes: {
      [_ in never]: never
    }
  }
}

type DatabaseWithoutInternals = Omit<Database, "__InternalSupabase">

type DefaultSchema = DatabaseWithoutInternals[Extract<keyof Database, "public">]

export type Tables<
  DefaultSchemaTableNameOrOptions extends
    | keyof (DefaultSchema["Tables"] & DefaultSchema["Views"])
    | { schema: keyof DatabaseWithoutInternals },
  TableName extends DefaultSchemaTableNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof (DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"] &
        DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Views"])
    : never = never,
> = DefaultSchemaTableNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? (DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"] &
      DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Views"])[TableName] extends {
      Row: infer R
    }
    ? R
    : never
  : DefaultSchemaTableNameOrOptions extends keyof (DefaultSchema["Tables"] &
        DefaultSchema["Views"])
    ? (DefaultSchema["Tables"] &
        DefaultSchema["Views"])[DefaultSchemaTableNameOrOptions] extends {
        Row: infer R
      }
      ? R
      : never
    : never

export type TablesInsert<
  DefaultSchemaTableNameOrOptions extends
    | keyof DefaultSchema["Tables"]
    | { schema: keyof DatabaseWithoutInternals },
  TableName extends DefaultSchemaTableNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"]
    : never = never,
> = DefaultSchemaTableNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"][TableName] extends {
      Insert: infer I
    }
    ? I
    : never
  : DefaultSchemaTableNameOrOptions extends keyof DefaultSchema["Tables"]
    ? DefaultSchema["Tables"][DefaultSchemaTableNameOrOptions] extends {
        Insert: infer I
      }
      ? I
      : never
    : never

export type TablesUpdate<
  DefaultSchemaTableNameOrOptions extends
    | keyof DefaultSchema["Tables"]
    | { schema: keyof DatabaseWithoutInternals },
  TableName extends DefaultSchemaTableNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"]
    : never = never,
> = DefaultSchemaTableNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"][TableName] extends {
      Update: infer U
    }
    ? U
    : never
  : DefaultSchemaTableNameOrOptions extends keyof DefaultSchema["Tables"]
    ? DefaultSchema["Tables"][DefaultSchemaTableNameOrOptions] extends {
        Update: infer U
      }
      ? U
      : never
    : never

export type Enums<
  DefaultSchemaEnumNameOrOptions extends
    | keyof DefaultSchema["Enums"]
    | { schema: keyof DatabaseWithoutInternals },
  EnumName extends DefaultSchemaEnumNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[DefaultSchemaEnumNameOrOptions["schema"]]["Enums"]
    : never = never,
> = DefaultSchemaEnumNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[DefaultSchemaEnumNameOrOptions["schema"]]["Enums"][EnumName]
  : DefaultSchemaEnumNameOrOptions extends keyof DefaultSchema["Enums"]
    ? DefaultSchema["Enums"][DefaultSchemaEnumNameOrOptions]
    : never

export type CompositeTypes<
  PublicCompositeTypeNameOrOptions extends
    | keyof DefaultSchema["CompositeTypes"]
    | { schema: keyof DatabaseWithoutInternals },
  CompositeTypeName extends PublicCompositeTypeNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[PublicCompositeTypeNameOrOptions["schema"]]["CompositeTypes"]
    : never = never,
> = PublicCompositeTypeNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[PublicCompositeTypeNameOrOptions["schema"]]["CompositeTypes"][CompositeTypeName]
  : PublicCompositeTypeNameOrOptions extends keyof DefaultSchema["CompositeTypes"]
    ? DefaultSchema["CompositeTypes"][PublicCompositeTypeNameOrOptions]
    : never

export const Constants = {
  public: {
    Enums: {
      image_type: ["skin_lesion", "ct_scan", "mri", "xray", "eeg"],
    },
  },
} as const
