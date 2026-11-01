import pandas as pd
import numpy as np
from scipy import interpolate
from sklearn.preprocessing import RobustScaler

class WaterQualityPreprocessor:
    def __init__(self):
        self.scaler = RobustScaler()
        self.interpolation_methods = {
            'pH': 'linear',
            'DO': 'cubic',
            'COD': 'linear',
            'EC': 'linear',
            'NH': 'linear',
            'TUR': 'linear',
            'WT': 'cubic'
        }
        
    def load_and_clean_data(self, filepath):
        """Load and perform initial cleaning"""
        df = pd.read_csv(filepath, parse_dates=['timestamp'])
        df = df.set_index('timestamp')
        
        # Remove duplicate timestamps
        df = df[~df.index.duplicated(keep='first')]
        
        # Sort by timestamp
        df = df.sort_index()
        
        return df
    
    def handle_missing_values(self, df):
        """Advanced missing value handling"""
        df_clean = df.copy()
        
        for column in df.columns:
            missing_mask = df_clean[column].isnull()
            
            if missing_mask.sum() > 0:
                # Use appropriate interpolation method for each parameter
                method = self.interpolation_methods.get(column, 'linear')
                
                if method == 'cubic' and missing_mask.sum() < len(df) * 0.1:
                    # Cubic interpolation for smooth parameters (low missing rate)
                    df_clean[column] = df_clean[column].interpolate(
                        method='cubic', limit_direction='both'
                    )
                else:
                    # Linear interpolation as fallback
                    df_clean[column] = df_clean[column].interpolate(
                        method='linear', limit_direction='both'
                    )
                
                # Forward/backward fill for remaining NaN values
                df_clean[column] = df_clean[column].fillna(method='bfill').fillna(method='ffill')
        
        return df_clean
    
    def resample_data(self, df, frequency='1min'):
        """Resample to consistent time intervals"""
        return df.resample(frequency).mean()
    
    def remove_outliers(self, df, method='iqr', threshold=1.5):
        """Remove statistical outliers"""
        df_clean = df.copy()
        
        for column in df.columns:
            if method == 'iqr':
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                # Replace outliers with boundary values (winsorizing)
                df_clean[column] = df_clean[column].clip(lower_bound, upper_bound)
            
            elif method == 'zscore':
                z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
                df_clean[column] = df_clean[column].mask(z_scores > threshold, 
                                                        df[column].median())
        
        return df_clean
    
    def preprocess(self, df):
        """Complete preprocessing pipeline (expects DataFrame)"""
        print("Handling missing values...")
        df = self.handle_missing_values(df)
        print("Resampling to consistent intervals...")
        df = self.resample_data(df)
        print("Removing outliers...")
        df = self.remove_outliers(df)
        print(f"Preprocessing complete. Final shape: {df.shape}")
        return df