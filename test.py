import sys
sys.path.insert(0, '.')
from src.data.preprocess import EnergyFeatureEngineer
import joblib
import holidays

eng = joblib.load('models/feature_engineer.pkl')
eng.__class__ = EnergyFeatureEngineer
eng.german_holidays = holidays.Germany()
joblib.dump(eng, 'models/feature_engineer.pkl')
print('Done — feature engineer re-saved successfully')