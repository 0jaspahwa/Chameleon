export interface HeroSection {
  title: string;
  subtitle: string;
  cta: string;
  targeted_products?: string[];
}

export interface ProductModule {
  title: string;
  products: string[];
}

export interface PersonalizationDetails {
  predicted_segment: number;
  segment_name?: string;
  assigned_business_tags: string[];
  prediction_source?: string;
  confidence?: number;
  is_cold_start?: boolean;
  target_category?: string;
}

export interface PersonalizeResponse {
  hero_section: HeroSection;
  product_modules: ProductModule[];
  personalization_details: PersonalizationDetails;
}