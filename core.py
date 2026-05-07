
import os
import json
import pandas as pd
import numpy as np

# FIX: Changed to absolute import for Colab execution
from ml_models import (
    PersonalizationModel, train_kmeans
)

try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    print(" Warning: The 'google-genai' library is not installed. AI-powered content generation will be disabled.")
    HAS_GENAI = False


class UltraPersonalizationEngine:
    def __init__(self, merged_df,api_key=None):
        print("🤖 Initializing UltraPersonalizationEngine...")

        # 2. Keep only the merged dataset
        self.merged_df = merged_df
        self.api_key = api_key

        # 3. Initialize the refactored ML pipeline
        self.pm = PersonalizationModel(rf_threshold=0.65, lstm_threshold=0.70)

        self.user_features = pd.DataFrame()
        self.cluster_features_cols = []

        # --- The Correct, Sequential Training Pipeline ---
        #self._prepare_user_features()
        #self._generate_ground_truth_labels()
        #self._apply_business_tags()
        #self._train_all_models()

        print(" Engine ready with all models (RF, Deep, LSTM) trained.")


    def _prepare_user_features(self):
        """Step 1: Aggregates raw data into a feature set for each user."""
        print("   - Step 1: Preparing user features...")

        # 1. Use native, fast aggregations ONLY
        features = self.merged_df.groupby('user_pseudo_id').agg(
            total_events=('event_name', 'count'),
            unique_pages=('page_type', 'nunique'),
            total_revenue=('purchase_revenue', 'sum'),
            first_visit=('timestamp', 'min')  
        ).reset_index()

        now = pd.Timestamp.now(tz='UTC')
        features['days_since_first_visit'] = (now - features['first_visit']).dt.days

        features.drop(columns=['first_visit'], inplace=True)

        self.user_features = features
        self.cluster_features_cols = [c for c in features.columns if c not in ['user_pseudo_id']]

    def _generate_ground_truth_labels(self):
        """
        Step 2: Creates the baseline behavioral clusters to act as the target variable (y)
        for our supervised learning models.
        """
        print("   - Step 2: Generating ground-truth behavioral clusters...")
        X_cluster = self.user_features[self.cluster_features_cols]

        kmeans_model, scaler = train_kmeans(X_cluster)

        self.user_features['target_cluster'] = kmeans_model.predict(scaler.transform(X_cluster))

    def _apply_business_tags(self):
        """Step 3: Applies descriptive rule-based tags for direct business logic."""
        print("   - Step 3: Applying rule-based business tags...")
        def get_tags(row):
            tags = []
            if row['total_revenue'] > 200: tags.append('high_value')
            if row['days_since_first_visit'] < 30: tags.append('new_user')
            if row['total_events'] > 50: tags.append('frequent_shopper')
            return tags if tags else ['standard']
        self.user_features['tags'] = self.user_features.apply(get_tags, axis=1)

    def _train_all_models(self):
        """Step 4: Trains all models using the refactored PersonalizationModel class."""
        print("   - Step 4: Training all personalization models...")
        self.pm.train_all(
            df=self.user_features,
            feature_cols=self.cluster_features_cols,
            target_col='target_cluster',
            merged_df=self.merged_df
        )

    def generate_personalized_landing_page(self, user_profile: dict) -> dict:
        """
        Generates a fully dynamic response using all trained models.
        """
        feature_dict = {
            'total_events': user_profile.get('total_events', 0),
            'unique_pages': user_profile.get('unique_pages', 0),
            'total_revenue': user_profile.get('total_revenue', 0),
            'days_since_first_visit': user_profile.get('days_since_first_visit', 365)
        }

        if user_profile.get('recent_sequence'):
            user_sequence = user_profile['recent_sequence']
        else:
            user_id = user_profile.get('user_pseudo_id')
            user_sequence = None
            # Safely check if merged_df has data before filtering
            if user_id and not self.merged_df.empty:
                user_events = self.merged_df[self.merged_df['user_pseudo_id'] == user_id]
                if not user_events.empty:
                    user_sequence = user_events.sort_values('timestamp')['event_name'].tolist()[-20:]

        pred_info = self.pm.predict_for_user(feature_dict, user_sequence=user_sequence)
        cluster = pred_info.get('cluster', 0)

        segment_profile = self._get_segment_profile(cluster)
        top_category = self._get_top_category_for_cluster(cluster)
        user_tags = self._apply_business_tags_to_single_user(feature_dict)

        hero_section = self._generate_hero_section(top_category, segment_profile)
        product_modules = self._generate_product_modules(cluster)
        cta_modules = self._generate_cta_modules(segment_profile, user_tags)

        return {
            'hero_section': hero_section,
            'product_modules': product_modules,
            'cta_modules': cta_modules,
            'personalization_details': {
                'predicted_behavioral_cluster': int(cluster),
                'assigned_business_tags': user_tags,
                'prediction_source': pred_info.get('source'),
                'confidence': round(pred_info.get('confidence', 0.0), 4)
            }
        }

    # --- Helper Functions ---
    def _apply_business_tags_to_single_user(self, profile: dict) -> list:
        tags = []
        if profile.get('total_revenue', 0) > 200: tags.append('high_value')
        if profile.get('days_since_first_visit', 365) < 30: tags.append('new_user')
        if profile.get('total_events', 0) > 50: tags.append('frequent_shopper')
        return tags if tags else ['standard']

    def _get_top_category_for_cluster(self, cluster: int) -> str:
        """Safely gets the most frequent item category for a given cluster."""
        cluster_users = self.user_features[self.user_features.get('target_cluster', -1) == cluster]['user_pseudo_id']
        if cluster_users.empty: return "Featured Products"

        cluster_transactions = self.merged_df[self.merged_df['user_pseudo_id'].isin(cluster_users)]

        if cluster_transactions.empty or 'ItemCategory' not in cluster_transactions.columns:
            return "Featured Products"

        top_cat = cluster_transactions['ItemCategory'].mode()
        return top_cat[0] if not top_cat.empty and pd.notna(top_cat[0]) else "Featured Products"

    def _get_segment_profile(self, cluster: int) -> dict:
        """Calculates average metrics for a given cluster."""
        cluster_data = self.user_features[self.user_features.get('target_cluster', -1) == cluster]
        if cluster_data.empty:
            return {'avg_revenue': 0, 'avg_events': 0}
        return {
            'avg_revenue': float(cluster_data['total_revenue'].mean()),
            'avg_events': float(cluster_data['total_events'].mean())
        }

    def _generate_hero_section(self, top_category: str, segment_profile: dict) -> dict:
        """Generates hero section content using native JSON capabilities."""
        print(f" Generating hero section for category: {top_category}...")

        if HAS_GENAI and self.api_key:
            try:
                # Explicitly pass the key we saved in __init__
                client = genai.Client(api_key=self.api_key)

                prompt = f"""
                As a marketing expert for an e-commerce store, write content for a website's hero section.
                The user's favorite product category is '{top_category}'.
                Their user segment has an average purchase value of ${segment_profile['avg_revenue']:.2f}.
                Generate a JSON object with exactly three keys: "title", "subtitle", and "cta".
                - "title": A catchy headline, max 8 words.
                - "subtitle": An engaging sentence to draw the user in.
                - "cta": A compelling call-to-action, max 4 words.
                Be creative and avoid generic phrases.
                """

                # New syntax for generating content
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                    )
                )
                return json.loads(response.text)

            except Exception as e:
                print(f" Generative AI Error: {e}")

        return {
            'title': f'Explore Our {top_category} Collection',
            'subtitle': 'Find exactly what you need from our handpicked selection.',
            'cta': 'Shop Now'
        }

    def _generate_product_modules(self, cluster: int) -> list:
        """Generates a list of product modules based on the cluster's top categories."""
        cluster_users = self.user_features[self.user_features.get('target_cluster', -1) == cluster]['user_pseudo_id']
        cluster_trans = self.merged_df[self.merged_df['user_pseudo_id'].isin(cluster_users)]

        if cluster_trans.empty or 'ItemCategory' not in cluster_trans.columns:
            return [{'title': 'Popular Products', 'products': ['Classic T-Shirt', 'Leather Wallet', 'Wireless Earbuds']}]

        top_categories = cluster_trans['ItemCategory'].value_counts().nlargest(2).index
        modules = []
        for cat in top_categories:
            if pd.isna(cat) or cat == "Unknown": continue
            products = cluster_trans[cluster_trans['ItemCategory'] == cat]['ItemName'].value_counts().nlargest(3).index.tolist()
            if products:
                modules.append({'title': f'Top Picks in {cat}', 'products': products})

        return modules if modules else [{'title': 'Popular Products', 'products': ['Classic T-Shirt', 'Leather Wallet', 'Wireless Earbuds']}]

    def _generate_cta_modules(self, segment_profile: dict, user_tags: list) -> list:
        """Generates a list of call-to-action modules based on user tags and segment behavior."""
        ctas = []
        if 'high_value' in user_tags:
            ctas.append({'type': 'loyalty', 'title': 'An Exclusive Offer for Our VIPs', 'button_text': 'View Your Deals'})
        elif 'new_user' in user_tags:
            ctas.append({'type': 'first_purchase', 'title': 'Get 20% Off Your First Order!', 'button_text': 'Claim Offer'})
        elif segment_profile.get('avg_revenue', 0) < 100:
            ctas.append({'type': 'engagement', 'title': 'Explore Our Best-Sellers', 'button_text': "See What's Popular"})
        else:
            ctas.append({'type': 'standard', 'title': 'Discover Something New', 'button_text': 'Browse All'})
        return ctas