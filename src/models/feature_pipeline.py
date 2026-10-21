import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder

class YouTubeFeaturePipeline(BaseEstimator, TransformerMixin):
    """
    Leakage-safe feature extraction and transformation pipeline for YouTube video engagement.
    Encodes:
    1. Structural and style features (length, word count, caps ratio, punctuation, density)
    2. Temporal cyclical features (hour_sin/cos, day_sin/cos, age)
    3. Channel authority features (channel frequency, out-of-fold smoothed target encoding)
    4. Categorical features (One-Hot Encoded category_id)
    5. Text n-gram features (TF-IDF unigrams & bigrams)
    """
    def __init__(self, tfidf_max_features=100, smoothing=10.0, random_state=42):
        self.tfidf_max_features = tfidf_max_features
        self.smoothing = smoothing
        self.random_state = random_state
        
        self.tfidf = None
        self.cat_encoder = None
        self.channel_counts = {}
        self.channel_target_map = {}
        self.global_target_mean = 1.0
        
        self.num_cols = [
            'title_length', 'title_word_count', 'description_length',
            'upload_hour', 'upload_day_num', 'upload_month', 'is_weekend', 'video_age_days',
            'title_caps_ratio', 'has_exclamation', 'has_question', 'has_digits',
            'title_character_density', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos'
        ]
        self.feature_names = []

    def _ensure_row_features(self, df):
        """Generates deterministic row-wise features if missing in the input dataframe."""
        df = df.copy()
        
        # Datetime
        if 'published_at' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['published_at']):
            df['published_at'] = pd.to_datetime(df['published_at'], errors='coerce')

        # Content lengths
        if 'title_length' not in df.columns:
            df['title_length'] = df['title'].fillna('').astype(str).str.len()
        if 'title_word_count' not in df.columns:
            df['title_word_count'] = df['title'].fillna('').astype(str).str.split().str.len()
        if 'description_length' not in df.columns:
            df['description_length'] = df['description'].fillna('').astype(str).str.len()

        # Style metrics
        if 'title_caps_ratio' not in df.columns:
            def calc_caps(s):
                letters = [c for c in str(s) if c.isalpha()]
                return sum(1 for c in letters if c.isupper()) / len(letters) if len(letters) > 0 else 0.0
            df['title_caps_ratio'] = df['title'].apply(calc_caps)

        if 'has_exclamation' not in df.columns:
            df['has_exclamation'] = df['title'].fillna('').astype(str).apply(lambda s: 1 if '!' in s else 0)
        if 'has_question' not in df.columns:
            df['has_question'] = df['title'].fillna('').astype(str).apply(lambda s: 1 if '?' in s else 0)
        if 'has_digits' not in df.columns:
            df['has_digits'] = df['title'].fillna('').astype(str).apply(lambda s: 1 if any(c.isdigit() for c in s) else 0)
        if 'title_character_density' not in df.columns:
            df['title_character_density'] = df['title_length'] / np.maximum(df['title_word_count'], 1)

        # Temporal features
        if 'upload_hour' not in df.columns and 'published_at' in df.columns:
            df['upload_hour'] = df['published_at'].dt.hour.fillna(12)
            df['upload_day_num'] = df['published_at'].dt.dayofweek.fillna(0)
            df['upload_month'] = df['published_at'].dt.month.fillna(6)
            df['is_weekend'] = df['upload_day_num'].apply(lambda x: 1 if x in [5, 6] else 0)

        # Cyclical features
        if 'hour_sin' not in df.columns and 'upload_hour' in df.columns:
            df['hour_sin'] = np.sin(2 * np.pi * df['upload_hour'] / 24.0)
            df['hour_cos'] = np.cos(2 * np.pi * df['upload_hour'] / 24.0)
        if 'day_sin' not in df.columns and 'upload_day_num' in df.columns:
            df['day_sin'] = np.sin(2 * np.pi * df['upload_day_num'] / 7.0)
            df['day_cos'] = np.cos(2 * np.pi * df['upload_day_num'] / 7.0)

        if 'video_age_days' not in df.columns:
            df['video_age_days'] = 100.0

        for col in self.num_cols:
            if col not in df.columns:
                df[col] = 0.0
            else:
                df[col] = df[col].fillna(0.0)

        return df

    def fit_transform(self, df, y):
        df = self._ensure_row_features(df)
        y = np.asarray(y)
        self.global_target_mean = float(np.mean(y))

        # 1. Channel Frequency
        self.channel_counts = df['channel_title'].fillna('Unknown').value_counts().to_dict()
        channel_freq = df['channel_title'].fillna('Unknown').map(self.channel_counts).fillna(1).values.reshape(-1, 1)

        # 2. Out-of-fold Channel Target Encoding with Smoothing
        oof_channel_te = np.zeros(len(df))
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        for tr_idx, val_idx in skf.split(df, y):
            tr_ch = df.iloc[tr_idx]['channel_title'].fillna('Unknown')
            tr_y = y[tr_idx]
            
            stats = pd.DataFrame({'ch': tr_ch, 'y': tr_y}).groupby('ch').agg(['count', 'mean'])
            c_dict = stats[('y', 'count')].to_dict()
            m_dict = stats[('y', 'mean')].to_dict()
            
            val_ch = df.iloc[val_idx]['channel_title'].fillna('Unknown')
            val_vals = []
            for ch in val_ch:
                n = c_dict.get(ch, 0)
                m = m_dict.get(ch, self.global_target_mean)
                val_vals.append((n * m + self.smoothing * self.global_target_mean) / (n + self.smoothing))
            oof_channel_te[val_idx] = val_vals

        # Full channel target map for test/inference transform
        full_stats = pd.DataFrame({'ch': df['channel_title'].fillna('Unknown'), 'y': y}).groupby('ch').agg(['count', 'mean'])
        f_counts = full_stats[('y', 'count')].to_dict()
        f_means = full_stats[('y', 'mean')].to_dict()
        self.channel_target_map = {}
        for ch in df['channel_title'].fillna('Unknown').unique():
            n = f_counts.get(ch, 0)
            m = f_means.get(ch, self.global_target_mean)
            self.channel_target_map[ch] = (n * m + self.smoothing * self.global_target_mean) / (n + self.smoothing)
            
        channel_te = oof_channel_te.reshape(-1, 1)

        # 3. Categorical Category Encoding
        self.cat_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        cat_encoded = self.cat_encoder.fit_transform(df[['category_id']])

        # 4. Dense Matrix
        num_matrix = df[self.num_cols].values
        dense_matrix = np.hstack([num_matrix, channel_freq, channel_te, cat_encoded])

        # 5. TF-IDF Text Extraction
        self.tfidf = TfidfVectorizer(
            max_features=self.tfidf_max_features,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=2
        )
        tfidf_sparse = self.tfidf.fit_transform(df['title'].fillna('').astype(str))

        # Names
        cat_names = [f"cat_{c}" for c in self.cat_encoder.categories_[0]]
        tfidf_names = [f"tfidf_{w}" for w in self.tfidf.get_feature_names_out()]
        self.feature_names = self.num_cols + ['channel_freq', 'channel_target_enc'] + cat_names + tfidf_names

        return hstack([csr_matrix(dense_matrix), tfidf_sparse]).tocsr()

    def transform(self, df):
        df = self._ensure_row_features(df)

        # 1. Channel Frequency
        channel_freq = df['channel_title'].fillna('Unknown').map(self.channel_counts).fillna(1).values.reshape(-1, 1)

        # 2. Channel Target Encoding
        channel_te = df['channel_title'].fillna('Unknown').map(self.channel_target_map).fillna(self.global_target_mean).values.reshape(-1, 1)

        # 3. Categorical Encoding
        cat_encoded = self.cat_encoder.transform(df[['category_id']])

        # 4. Dense Matrix
        num_matrix = df[self.num_cols].values
        dense_matrix = np.hstack([num_matrix, channel_freq, channel_te, cat_encoded])

        # 5. TF-IDF
        tfidf_sparse = self.tfidf.transform(df['title'].fillna('').astype(str))

        return hstack([csr_matrix(dense_matrix), tfidf_sparse]).tocsr()
