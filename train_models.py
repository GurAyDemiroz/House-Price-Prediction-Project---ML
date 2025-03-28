# Data manipulation and analysis
import pandas as pd
import numpy as np

# Scikit-learn imports
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict, StratifiedKFold, RepeatedKFold, RepeatedStratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, median_absolute_error
from sklearn.pipeline import Pipeline

# Linear models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge

# Tree-based models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor

# Other models
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Optimization
import optuna
import os
import joblib
from datetime import datetime
import warnings

# Custom imports
from pipeline import DataPreprocessingPipeline
from logger import get_logger

# Disable warnings
warnings.filterwarnings('ignore')

# Initialize logger
logger = get_logger("logs/model_tuning.log")

# Tolerance R2 function for evaluation
def tolerance_r2(y_test, y_pred, tolerance=100_000):
    """
    Calculates R² score where errors within the tolerance are considered perfect predictions.
    
    Args:
        y_test: True values
        y_pred: Predicted values
        tolerance: Absolute error tolerance (default: 100,000)
    
    Returns:
        Modified R² score
    """
    residuals = y_pred - y_test
    residuals[np.abs(residuals) <= tolerance] = 0
    ssr = np.sum(residuals**2)
    sst = np.sum((y_test - np.mean(y_test))**2)
    return 1 - ssr/sst

# Percentage tolerance R2 function
def tolerance_percentage_r2(y_true, y_pred, tolerance=0.05):
    """
    Calculates R² score where percentage errors within the tolerance are considered perfect predictions.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        tolerance: Percentage error tolerance (default: 5%)
    
    Returns:
        Modified R² score
    """
    residuals = y_pred - y_true
    residuals[(np.abs(residuals) / y_true) <= tolerance] = 0
    ssr = np.sum(residuals**2)
    sst = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ssr / sst)

# RMSLE metriği ekleniyor
def root_mean_squared_log_error(y_true, y_pred):
    """
    Root Mean Squared Logarithmic Error (RMSLE) hesaplar.
    Logaritmik fark, büyük değer aralıklarında daha dengeli bir metrik sağlar.
    
    Args:
        y_true: Gerçek değerler
        y_pred: Tahmin edilen değerler
        
    Returns:
        RMSLE skoru
    """
    # Negatif değerleri önlemek için 1 ekliyoruz
    y_true, y_pred = np.maximum(y_true, 0) + 1, np.maximum(y_pred, 0) + 1
    return np.sqrt(np.mean(np.power(np.log(y_pred) - np.log(y_true), 2)))

# MAPE metriği ekleniyor
def mean_absolute_percentage_error(y_true, y_pred):
    """
    Mean Absolute Percentage Error (MAPE) hesaplar.
    
    Args:
        y_true (array-like): Gerçek değerler.
        y_pred (array-like): Tahmin edilen değerler.
        
    Returns:
        float: MAPE değeri (yüzde cinsinden).
    """
    # Sıfır bölme hatalarını önlemek için küçük bir epsilon değeri ekliyoruz.
    epsilon = 1e-10  
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    return mape

class HousePriceModel:
    def __init__(self, data_path, random_state=42):
        """
        Initialize the House Price Model trainer
        
        Args:
            data_path: Path to the dataset
            random_state: Random seed for reproducibility
        """
        self.data_path = data_path
        self.random_state = random_state
        
        # Model sınıflarını tanımla
        self.model_classes = {
            'LinearRegression': LinearRegression,
            'Ridge': Ridge,
            'Lasso': Lasso,
            'ElasticNet': ElasticNet,
            'RandomForest': RandomForestRegressor,
            'GradientBoosting': GradientBoostingRegressor,
            'XGBoost': XGBRegressor,
            'LightGBM': LGBMRegressor,
            'CatBoost': CatBoostRegressor,
            'BayesianRidge': BayesianRidge,
            'DecisionTree': DecisionTreeRegressor
        }
        
        # Encoder ve Scaler sınıf değişkenleri
        self.encoder = None
        self.scaler = None
        self.best_model = None
        self.best_params = None
        self.best_score = float('-inf')
        self.model_performances = {}
        
        # Orijinal feature isimleri
        self.numeric_features = []
        self.categorical_features = []

    def load_data(self):
        """Load and process the house price dataset"""
        logger.info("Loading data from %s...", self.data_path)
        try:
            house_df = pd.read_csv(self.data_path)
            data = DataPreprocessingPipeline(house_df=house_df).get_final_df()
            
            # Columns to exclude from features
            drop_columns = ['house_description', 'property_type',
                           'usage_status', 'credit_eligible', 'deed_status', 
                           'exchange', 'house_url', "house_price"]
            
            self.X = data.drop(columns=drop_columns, axis=1)
            self.y = data['house_price']
            
            logger.info("Data loaded successfully with %d samples and %d features",
                      len(self.X), self.X.shape[1])
            logger.info("Price range: %.2f - %.2f", self.y.min(), self.y.max())
            logger.info("Average price: %.2f", self.y.mean())
            
            # Kategorik ve sayısal özellikleri sakla (orijinal featureler)
            self.categorical_features = ['heating', 'province', 'district', 'neighborhood', 'furnished']
            self.numeric_features = [col for col in self.X.columns if col not in self.categorical_features]
            
            logger.info("Categorical features: %s", self.categorical_features)
            logger.info("Numeric features: %s", self.numeric_features)
            
        except Exception as e:
            logger.error("Error loading data: %s", str(e), exc_info=True)
            raise

    def preprocess_data(self):
        """Preprocess the data for modeling"""
        logger.info("Preprocessing data...")
        try:
            # Veriyi sadece train/test olarak böl
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y, test_size=0.2, random_state=self.random_state
            )
            
            # Sayısal özellikleri ölçeklendir
            self.scaler = StandardScaler()
            X_train_numeric = X_train[self.numeric_features]
            X_test_numeric = X_test[self.numeric_features]
            
            X_train_numeric_scaled = pd.DataFrame(
                self.scaler.fit_transform(X_train_numeric),
                columns=self.numeric_features,
                index=X_train_numeric.index
            )
            
            X_test_numeric_scaled = pd.DataFrame(
                self.scaler.transform(X_test_numeric),
                columns=self.numeric_features,
                index=X_test_numeric.index
            )
            
            # Kategorik özellikleri kodla
            X_train_cat = X_train[self.categorical_features]
            X_test_cat = X_test[self.categorical_features]
            
            self.encoder = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)
            X_train_encoded = self.encoder.fit_transform(X_train_cat)
            X_test_encoded = self.encoder.transform(X_test_cat)
            
            feature_names = self.encoder.get_feature_names_out(self.categorical_features)
            
            X_train_encoded_df = pd.DataFrame(
                X_train_encoded,
                columns=feature_names,
                index=X_train.index
            )
            
            X_test_encoded_df = pd.DataFrame(
                X_test_encoded,
                columns=feature_names,
                index=X_test.index
            )
            
            # Özellikleri birleştir
            self.X_train = pd.concat([X_train_numeric_scaled, X_train_encoded_df], axis=1)
            self.X_test = pd.concat([X_test_numeric_scaled, X_test_encoded_df], axis=1)
            self.y_train = y_train
            self.y_test = y_test
            
            # Validation referanslarını kaldır, train veriyi cross-validation için kullanacağız
            self.X_val = None
            self.y_val = None
            
            # Aykırı değer tespiti kaldırıldı - pipeline'da zaten işleniyor
            
            logger.info("Data preprocessing completed successfully")
            logger.info("Training set: %d samples, Test set: %d samples", 
                      len(self.X_train), len(self.X_test))
            logger.info("Total features after encoding: %d", self.X_train.shape[1])
            
        except Exception as e:
            logger.error("Error in preprocessing data: %s", str(e), exc_info=True)
            raise

    def train_and_evaluate_base_models(self):
        """Train and evaluate all base models before hyperparameter tuning"""
        logger.info("Training and evaluating base models...")
        
        results = []
        
        # RepeatedKFold ile çapraz doğrulama (Stratified yerine)
        n_splits = 5
        n_repeats = 2
        cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=self.random_state)
        
        for model_name in self.model_classes.keys():
            try:
                logger.info("Training %s with default parameters...", model_name)
                
                # Model oluştur, random_state kullanmadan
                model = self.model_classes[model_name]()
                
                # Train modeli
                model.fit(self.X_train, self.y_train)
                
                # Sadece eğitim seti ve test seti üzerinde değerlendir
                y_pred_train = model.predict(self.X_train)
                y_pred_test = model.predict(self.X_test)
                
                # Metrikler hesapla
                train_r2 = r2_score(self.y_train, y_pred_train)
                test_r2 = r2_score(self.y_test, y_pred_test)
                train_rmse = np.sqrt(mean_squared_error(self.y_train, y_pred_train))
                test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
                train_tpr2 = tolerance_percentage_r2(self.y_train, y_pred_train)
                test_tpr2 = tolerance_percentage_r2(self.y_test, y_pred_test)
                
                # Yeni metrikler ekleniyor
                train_rmsle = root_mean_squared_log_error(self.y_train, y_pred_train)
                test_rmsle = root_mean_squared_log_error(self.y_test, y_pred_test)
                train_mape = mean_absolute_percentage_error(self.y_train, y_pred_train)
                test_mape = mean_absolute_percentage_error(self.y_test, y_pred_test)
                
                # Cross-validation with RepeatedKFold
                cv_scores = []
                cv_rmsle_scores = []
                cv_mape_scores = []
                
                # Price bins olmadan split et
                for fold_idx, (train_idx, val_idx) in enumerate(cv.split(self.X_train)):
                    X_cv_train = self.X_train.iloc[train_idx]
                    y_cv_train = self.y_train.iloc[train_idx]
                    X_cv_val = self.X_train.iloc[val_idx]
                    y_cv_val = self.y_train.iloc[val_idx]
                    
                    # Her CV fold için yeni bir model örneği oluştur
                    cv_model = self.model_classes[model_name]()
                    
                    cv_model.fit(X_cv_train, y_cv_train)
                    y_cv_pred = cv_model.predict(X_cv_val)
                    
                    # TPR2 skorunu hesapla
                    tpr2_score = tolerance_percentage_r2(y_cv_val, y_cv_pred)
                    rmsle_score = root_mean_squared_log_error(y_cv_val, y_cv_pred)
                    mape_score = mean_absolute_percentage_error(y_cv_val, y_cv_pred)
                    
                    cv_scores.append(tpr2_score)
                    cv_rmsle_scores.append(rmsle_score)
                    cv_mape_scores.append(mape_score)
                
                cv_mean = np.mean(cv_scores)
                cv_std = np.std(cv_scores)
                cv_rmsle_mean = np.mean(cv_rmsle_scores)
                cv_mape_mean = np.mean(cv_mape_scores)
                
                # Sonuçları sakla
                result = {
                    'model_name': model_name,
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'train_rmse': train_rmse,
                    'test_rmse': test_rmse,
                    'train_tpr2': train_tpr2,
                    'test_tpr2': test_tpr2,
                    'train_rmsle': train_rmsle,
                    'test_rmsle': test_rmsle,
                    'train_mape': train_mape,
                    'test_mape': test_mape,
                    'cv_mean': cv_mean,
                    'cv_std': cv_std,
                    'cv_rmsle': cv_rmsle_mean,
                    'cv_mape': cv_mape_mean,
                    'cv_repeats': n_repeats,
                    'cv_splits': n_splits
                }
                
                results.append(result)
                self.model_performances[model_name] = result
                
                logger.info("  %s Performance:", model_name)
                logger.info("    Train R²: %.4f, Test R²: %.4f", train_r2, test_r2)
                logger.info("    Train TPR²: %.4f, Test TPR²: %.4f", train_tpr2, test_tpr2)
                logger.info("    Train RMSE: %.4f, Test RMSE: %.4f", train_rmse, test_rmse)
                logger.info("    Train RMSLE: %.4f, Test RMSLE: %.4f", train_rmsle, test_rmsle)
                logger.info("    Train MAPE: %.2f%%, Test MAPE: %.2f%%", train_mape, test_mape)
                logger.info("    CV TPR² (%dx%d): %.4f ± %.4f", n_repeats, n_splits, cv_mean, cv_std)
                logger.info("    CV RMSLE: %.4f, CV MAPE: %.2f%%", cv_rmsle_mean, cv_mape_mean)
                
            except Exception as e:
                logger.error("Error training %s: %s", model_name, str(e), exc_info=True)
        
        # Cross-validation sonuçlarına göre sırala
        results.sort(key=lambda x: x['cv_mean'], reverse=True)
        
        logger.info("\nBase Model Performance Summary (sorted by CV TPR²):")
        for i, result in enumerate(results):
            logger.info("%d. %s: CV TPR² = %.4f ± %.4f, Test TPR² = %.4f", 
                      i+1, result['model_name'], result['cv_mean'], 
                      result['cv_std'], result['test_tpr2'])
        
        # En iyi base model
        self.best_base_model = results[0]['model_name']
        self.best_base_score = results[0]['cv_mean']  # CV sonuçlarını kullan
        
        return results

    def get_hyperparameter_space(self, trial, model_name):
        """
        Model için hiperparametre uzayını tanımlar
        
        Args:
            trial: Optuna trial nesnesi
            model_name: Model ismi
            
        Returns:
            Model parametreleri sözlüğü
        """
        model_params = {}
        
        if model_name == 'LinearRegression':
            model_params = {
                'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
                'positive': trial.suggest_categorical('positive', [True, False]),
                'n_jobs': -1
            }
            
        elif model_name == 'Ridge':
            model_params = {
                'alpha': trial.suggest_float('alpha', 1e-5, 10, log=True),
                'solver': trial.suggest_categorical('solver', ['auto', 'svd', 'cholesky', 'sparse_cg'])
            }
            
        elif model_name == 'Lasso':
            model_params = {
                'alpha': trial.suggest_float('alpha', 1e-5, 100, log=True),
                'selection': trial.suggest_categorical('selection', ['cyclic', 'random'])
            }
            
        elif model_name == 'ElasticNet':
            model_params = {
                'alpha': trial.suggest_float('alpha', 1e-5, 100, log=True),
                'l1_ratio': trial.suggest_float('l1_ratio', 0.0, 1.0),
                'selection': trial.suggest_categorical('selection', ['cyclic', 'random'])
            }
            
        elif model_name == 'RandomForest':
            model_params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
                'max_depth': trial.suggest_int('max_depth', 3, 40),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
                'n_jobs': -1
                # random_state kaldırıldı
            }
            
        elif model_name == 'GradientBoosting':
            model_params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
            }
            
        elif model_name == 'XGBoost':
            model_params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 2000),  # Üst sınır düşürüldü
                'max_depth': trial.suggest_int('max_depth', 3, 10),             # Üst sınır düşürüldü (8 -> 6)
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),  # Üst sınır düşürüldü
                'subsample': trial.suggest_float('subsample', 0.6, 0.9),        # Alt sınır yükseltildi
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
                'gamma': trial.suggest_float('gamma', 0.0, 5.0),                # Bölünme kayıp eşiği eklendi
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10), # Minimum yaprak ağırlığı eklendi
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),        # L1 regularizasyon eklendi
                'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0),     # L2 regularizasyon eklendi
                'n_jobs': -1
            }
            
        elif model_name == 'LightGBM':
            model_params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'n_jobs': -1
                # random_state kaldırıldı
            }
            
        elif model_name == 'CatBoost':
            model_params = {
                'iterations': trial.suggest_int('iterations', 100, 3000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'depth': trial.suggest_int('depth', 4, 10),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                'random_strength': trial.suggest_float('random_strength', 0.1, 10),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 10),
                'border_count': trial.suggest_int('border_count', 32, 255),
                'verbose': False
                # random_seed kaldırıldı
            }
            
        elif model_name == 'BayesianRidge':
            model_params = {
                'alpha_1': trial.suggest_float('alpha_1', 1e-7, 1e-2, log=True),
                'alpha_2': trial.suggest_float('alpha_2', 1e-7, 1e-2, log=True),
                'lambda_1': trial.suggest_float('lambda_1', 1e-7, 1e-2, log=True),
                'lambda_2': trial.suggest_float('lambda_2', 1e-7, 1e-2, log=True)
            }
            
        elif model_name == 'DecisionTree':
            model_params = {
                'max_depth': trial.suggest_int('max_depth', 3, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2'])
                # random_state kaldırıldı
            }
            
        return model_params

    def create_model(self, model_name, params):
        """
        Model ismini ve parametreleri alarak model örneği oluşturur
        
        Args:
            model_name: Model ismi
            params: Model parametreleri
            
        Returns:
            Model nesnesi
        """
        if model_name == 'XGBoost':
            model = XGBRegressor(**params)
        elif model_name == 'LightGBM':
            model = LGBMRegressor(**params)
        elif model_name == 'CatBoost':
            model = CatBoostRegressor(**params)
        else:
            model = self.model_classes[model_name](**params)
            
        return model

    def process_special_params(self, model_name, best_params):
        """
        Özel parametre işleme için yardımcı metod (MLP artık bulunmuyor)
        
        Args:
            model_name: Model ismi
            best_params: Optimizasyon sonrası parametre sözlüğü
        
        Returns:
            İşlenmiş parametre sözlüğü
        """
        # MLP özel işleme kaldırıldı
        return best_params.copy()

    def get_adaptive_cv(self, model_name):
        """
        Model tipine bakılmaksızın sabit CV stratejisi kullanır
        
        Args:
            model_name: Model ismi (artık kullanılmıyor ama arayüz tutarlılığı için tutuldu)
        
        Returns:
            (cv_folds, cv_object) - fold sayısı ve CV objesi
        """
        # Tüm modeller için sabit fold ve tekrar sayısı kullan
        n_splits = 5  # Tüm modeller için 10 split
        n_repeats = 2  # Tüm modeller için 3 tekrar
        
        # RepeatedKFold kullanalım (Stratified yerine)
        cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=self.random_state)
        
        logger.info(f"Using RepeatedKFold with {n_splits} folds and {n_repeats} repeats for all models")
        
        # Price bins artık gerekli değil, None döndürüyoruz
        return n_splits, cv, None, n_repeats

    def get_pruner(self, model_name):
        """
        Model tipine göre uyarlanabilir pruner
        
        Args:
            model_name: Model ismi
        
        Returns:
            Optuna pruner
        """
        # Hesaplama yoğun modeller için daha agresif pruning
        complex_models = ['XGBoost', 'LightGBM', 'CatBoost', 'RandomForest']  # MLP kaldırıldı
        
        if model_name in complex_models:
            # Hesaplama yoğun modeller için daha agresif (erken sonlandırma)
            return optuna.pruners.HyperbandPruner(
                min_resource=3, 
                max_resource=20, 
                reduction_factor=3
            )
        else:
            # Basit modeller için daha toleranslı
            return optuna.pruners.MedianPruner(
                n_startup_trials=5, 
                n_warmup_steps=5
            )

    def tune_model(self, n_trials=200):
        """
        Model hiperparametrelerini Optuna ile ayarla, her seferinde bir model
        
        Args:
            n_trials: Her model için optimizasyon denemesi sayısı
        """
        logger.info("Starting hyperparameter tuning with Optuna, one model at a time...")
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            all_best_models = {}
            
            for model_name in self.model_classes.keys():
                logger.info("\n" + "="*50)
                logger.info(f"Optimizing {model_name} (Trials: {n_trials})")
                logger.info("="*50)
                
                # Model tipine göre uyarlanabilir CV ve pruner
                n_splits, cv, _, n_repeats = self.get_adaptive_cv(model_name)
                pruner = self.get_pruner(model_name)
                logger.info(f"Using {n_splits}-fold CV with {pruner.__class__.__name__} for {model_name}")
                
                # Model için özel objective function tanımla
                def model_objective(trial):
                    # Modelin hiperparametre uzayını al
                    model_params = self.get_hyperparameter_space(trial, model_name)
                    model = self.create_model(model_name, model_params)
                    
                    # Uyarlanabilir CV stratejisi kullan (price_bins olmadan)
                    cv_scores = []
                    cv_tpr2_scores = []
                    cv_mape_scores = []
                    for train_idx, val_idx in cv.split(self.X_train):
                        X_fold_train = self.X_train.iloc[train_idx]
                        y_fold_train = self.y_train.iloc[train_idx]
                        X_fold_val = self.X_train.iloc[val_idx]
                        y_fold_val = self.y_train.iloc[val_idx]
                        
                        try:
                            # Bu fold için modeli eğit
                            model.fit(X_fold_train, y_fold_train)
                            y_pred = model.predict(X_fold_val)
                            
                            # TPR2 ve MAPE skorlarını hesapla
                            tpr2_score = tolerance_percentage_r2(y_fold_val, y_pred)
                            mape_score = mean_absolute_percentage_error(y_fold_val, y_pred)
                            
                            if np.isnan(tpr2_score) or np.isinf(tpr2_score):
                                cv_tpr2_scores.append(-1.0)  # Geçersiz skor için ceza
                            else:
                                cv_tpr2_scores.append(tpr2_score)
                            
                            if np.isnan(mape_score) or np.isinf(mape_score):
                                cv_mape_scores.append(float('inf'))  # Geçersiz MAPE için ceza
                            else:
                                cv_mape_scores.append(mape_score)
                        
                        except Exception as e:
                            logger.warning(f"Model {model_name} failed during CV fold: {str(e)}")
                            cv_tpr2_scores.append(-1.0)  # Hata durumunda ceza
                            cv_mape_scores.append(float('inf'))  # Hata durumunda ceza
                        
                        if hasattr(trial, 'report'):
                            # HyperbandPruner için ara skor raporlaması
                            for step, score in enumerate(cv_tpr2_scores):
                                trial.report(score, step)
                                if trial.should_prune():
                                    raise optuna.exceptions.TrialPruned()
                    
                    if not cv_tpr2_scores or np.mean(cv_tpr2_scores) < -0.5:
                        # Skorlar geçersizse, kötü bir değer döndür
                        return -0.5
                    
                    # Sadece TPR² metriğine odaklan, kombine metrik kullanma
                    avg_tpr2 = np.mean(cv_tpr2_scores)
                    return avg_tpr2
                
                # Bu model için çalışma oluştur
                study = optuna.create_study(
                    direction='maximize',
                    pruner=pruner,
                    study_name=f"{model_name}_tuning_{timestamp}"
                )
                
                # Bu model için optimizasyonu çalıştır
                try:
                    study.optimize(model_objective, n_trials=n_trials)
                    best_trial = study.best_trial
                    best_params = best_trial.params.copy()
                    best_score = best_trial.value
                    
                    logger.info(f"\nBest parameters for {model_name}:")
                    for param, value in best_params.items():
                        logger.info(f"  {param}: {value}")
                    logger.info(f"Best score (TPR²): {best_score:.4f}")
                    
                    best_params = self.process_special_params(model_name, best_params)
                    best_model = self.create_model(model_name, best_params)
                    
                    logger.info(f"Evaluating {model_name} with best parameters using cross-validation...")
                    
                    # Cross-validation ile değerlendirme
                    cv_scores = []
                    for train_idx, val_idx in cv.split(self.X_train):
                        X_fold_train = self.X_train.iloc[train_idx]
                        y_fold_train = self.y_train.iloc[train_idx]
                        X_fold_val = self.X_train.iloc[val_idx]
                        y_fold_val = self.y_train.iloc[val_idx]
                        
                        best_fold_model = self.create_model(model_name, best_params)
                        best_fold_model.fit(X_fold_train, y_fold_train)
                        y_fold_pred = best_fold_model.predict(X_fold_val)
                        score = tolerance_percentage_r2(y_fold_val, y_fold_pred)
                        cv_scores.append(score)
                    
                    cv_val_score = np.mean(cv_scores)
                    cv_val_std = np.std(cv_scores)
                    
                    logger.info(f"Best parameters CV TPR²: {cv_val_score:.4f} ± {cv_val_std:.4f}")
                    
                    best_model.fit(self.X_train, self.y_train)
                    
                    # Eğitim ve test setleri üzerinde performans değerlendirmesi
                    train_pred = best_model.predict(self.X_train)
                    test_pred = best_model.predict(self.X_test)
                    train_score = tolerance_percentage_r2(self.y_train, train_pred)
                    test_score = tolerance_percentage_r2(self.y_test, test_pred)
                    train_mape = mean_absolute_percentage_error(self.y_train, train_pred)
                    test_mape = mean_absolute_percentage_error(self.y_test, test_pred)
                    
                    logger.info(f"Train TPR²: {train_score:.4f}")
                    logger.info(f"Test TPR²: {test_score:.4f}")
                    logger.info(f"Train MAPE: {train_mape:.2f}%")
                    logger.info(f"Test MAPE: {test_mape:.2f}%")
                    
                    # MAPE değerini CV üzerinden hesapla (price_bins olmadan)
                    cv_mape_scores = []
                    for train_idx, val_idx in cv.split(self.X_train):
                        X_fold_train = self.X_train.iloc[train_idx]
                        y_fold_train = self.y_train.iloc[train_idx]
                        X_fold_val = self.X_train.iloc[val_idx]
                        y_fold_val = self.y_train.iloc[val_idx]
                        
                        best_fold_model = self.create_model(model_name, best_params)
                        best_fold_model.fit(X_fold_train, y_fold_train)
                        y_fold_pred = best_fold_model.predict(X_fold_val)
                        mape_score = mean_absolute_percentage_error(y_fold_val, y_fold_pred)
                        cv_mape_scores.append(mape_score)
                    
                    cv_mape_mean = np.mean(cv_mape_scores)
                    cv_mape_std = np.std(cv_mape_scores)
                    test_mape = mean_absolute_percentage_error(self.y_test, test_pred)
                    
                    logger.info(f"Best parameters CV MAPE: {cv_mape_mean:.2f}% ± {cv_mape_std:.2f}%")
                    logger.info(f"Test MAPE: {test_mape:.2f}%")
                    
                    val_score = cv_val_score
                    
                    # Train performansını da sakla
                    all_best_models[model_name] = {
                        'model': best_model,
                        'params': best_params,
                        'cv_score': best_score,
                        'val_score': val_score,
                        'val_score_std': cv_val_std,
                        'train_score': train_score,
                        'test_score': test_score,
                        'cv_mape': cv_mape_mean,
                        'cv_mape_std': cv_mape_std,
                        'train_mape': train_mape,
                        'test_mape': test_mape,
                    }
                
                except Exception as e:
                    logger.error(f"Error optimizing {model_name}: {e}", exc_info=True)
            
            if all_best_models:
                # CV skoruna göre genel en iyi modeli bul
                best_model_name = max(all_best_models.items(), key=lambda x: x[1]['val_score'])[0]
                best_model_info = all_best_models[best_model_name]
                
                logger.info("OVERALL BEST MODEL COMPARISON")
                logger.info("\n" + "="*50)
                logger.info("OVERALL BEST MODEL COMPARISON")
                logger.info("="*50)
                
                logger.info("\nModel Performance Summary (sorted by CV TPR²):")
                sorted_models = sorted(all_best_models.items(), key=lambda x: x[1]['val_score'], reverse=True)
                for i, (name, info) in enumerate(sorted_models):
                    logger.info(f"{i+1}. {name}: CV TPR² = {info['val_score']:.4f} ± {info['val_score_std']:.4f}, Train TPR² = {info['train_score']:.4f}, Test TPR² = {info['test_score']:.4f}")
                    logger.info(f"   Overfitting Ratio: {info['train_score']/info['test_score']:.2f}x, Train-CV Gap: {info['train_score']-info['val_score']:.4f}")
                
                logger.info(f"\nSelected Best Model: {best_model_name}")
                logger.info(f"  CV TPR²: {best_model_info['val_score']:.4f}")
                logger.info(f"  Test TPR²: {best_model_info['test_score']:.4f}")
                
                best_model = best_model_info['model']
                
                # No need to retrain, model is already trained on the full training set
                y_pred_test = best_model.predict(self.X_test)
                test_r2 = r2_score(self.y_test, y_pred_test)
                test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
                test_mae = mean_absolute_error(self.y_test, y_pred_test)
                test_tpr2 = tolerance_percentage_r2(self.y_test, y_pred_test)
                
                logger.info("\nFinal Best Model Performance on Test Set:")
                logger.info(f"  Model: {best_model_name}")
                logger.info(f"  R²: {test_r2:.4f}")
                logger.info(f"  RMSE: {test_rmse:.2f}")
                logger.info(f"  MAE: {test_mae:.2f}")
                logger.info(f"  TPR²: {test_tpr2:.4f}")
                
                self.best_model = best_model
                self.best_params = best_model_info['params']
                self.best_score = test_tpr2
                
                return best_model, self.best_params, test_tpr2
            else:
                logger.error("No models were successfully tuned.")
                return None, None, None
        
        except Exception as e:
            logger.error(f"Error during model tuning: {e}", exc_info=True)
            raise

    def evaluate_model(self):
        """Evaluate the best model with comprehensive metrics"""
        if self.best_model is None:
            logger.error("No best model available. Run tune_model() first.")
            return
        
        logger.info("\nComprehensive Evaluation of Best Model:")
        try:
            logger.info("Model Type: %s", self.best_model.__class__.__name__)
            
            y_pred_train = self.best_model.predict(self.X_train)
            y_pred_test = self.best_model.predict(self.X_test)
            
            metrics = {
                'R² (Train)': r2_score(self.y_train, y_pred_train),
                'R² (Test)': r2_score(self.y_test, y_pred_test),
                'RMSE (Train)': np.sqrt(mean_squared_error(self.y_train, y_pred_train)),
                'RMSE (Test)': np.sqrt(mean_squared_error(self.y_test, y_pred_test)),
                'RMSLE (Train)': root_mean_squared_log_error(self.y_train, y_pred_train),
                'RMSLE (Test)': root_mean_squared_log_error(self.y_test, y_pred_test),
                'MAE (Train)': mean_absolute_error(self.y_train, y_pred_train),
                'MAE (Test)': mean_absolute_error(self.y_test, y_pred_test),
                'MAPE (Train)': mean_absolute_percentage_error(self.y_train, y_pred_train),
                'MAPE (Test)': mean_absolute_percentage_error(self.y_test, y_pred_test),
                'Median AE (Train)': median_absolute_error(self.y_train, y_pred_train),
                'Median AE (Test)': median_absolute_error(self.y_test, y_pred_test),
                'TPR² 5% (Train)': tolerance_percentage_r2(self.y_train, y_pred_train, 0.05),
                'TPR² 5% (Test)': tolerance_percentage_r2(self.y_test, y_pred_test, 0.05),
                'TPR² 7% (Train)': tolerance_percentage_r2(self.y_train, y_pred_train, 0.07),
                'TPR² 7% (Test)': tolerance_percentage_r2(self.y_test, y_pred_test, 0.07),
                'TPR² 10% (Train)': tolerance_percentage_r2(self.y_train, y_pred_train, 0.1),
                'TPR² 10% (Test)': tolerance_percentage_r2(self.y_test, y_pred_test, 0.1),
                'Absolute R² 100k (Train)': tolerance_r2(self.y_train, y_pred_train, 100_000),
                'Absolute R² 100k (Test)': tolerance_r2(self.y_test, y_pred_test, 100_000),
            }
            
            train_errors = np.abs(self.y_train - y_pred_train)
            test_errors = np.abs(self.y_test - y_pred_test)
            train_pct_errors = train_errors / self.y_train * 100
            test_pct_errors = test_errors / self.y_test * 100
            
            error_stats = {
                'Mean Pct Error (Train)': np.mean(train_pct_errors),
                'Mean Pct Error (Test)': np.mean(test_pct_errors),
                'Median Pct Error (Train)': np.median(train_pct_errors),
                'Median Pct Error (Test)': np.median(test_pct_errors),
                '25th Pct Error (Train)': np.percentile(train_pct_errors, 25),
                '25th Pct Error (Test)': np.percentile(test_pct_errors, 25),
                '75th Pct Error (Train)': np.percentile(train_pct_errors, 75),
                '75th Pct Error (Test)': np.percentile(test_pct_errors, 75),
                '90th Pct Error (Train)': np.percentile(train_pct_errors, 90),
                '90th Pct Error (Test)': np.percentile(test_pct_errors, 90),
                '95th Pct Error (Train)': np.percentile(train_pct_errors, 95),
                '95th Pct Error (Test)': np.percentile(test_pct_errors, 95)
            }
            
            logger.info("\nStandard Metrics:")
            for name, value in metrics.items():
                logger.info("  %s: %.4f", name, value)
            
            logger.info("\nError Distribution (% of true price):")
            for name, value in error_stats.items():
                logger.info("  %s: %.2f%%", name, value)
            
            price_ranges = [
                (0, 3_500_000),
                (3_500_000, float('inf')),
            ]
            
            logger.info("\nPerformance by Price Range:")
            for min_price, max_price in price_ranges:
                range_mask = (self.y_test >= min_price) & (self.y_test < max_price)
                if np.sum(range_mask) == 0:
                    continue
                
                y_range = self.y_test[range_mask]
                y_pred_range = y_pred_test[range_mask]
                
                r2_range = r2_score(y_range, y_pred_range) if len(y_range) > 1 else float('nan')
                rmse_range = np.sqrt(mean_squared_error(y_range, y_pred_range))
                rmsle_range = root_mean_squared_log_error(y_range, y_pred_range)
                mae_range = mean_absolute_error(y_range, y_pred_range)
                mape_range = np.mean(np.abs((y_range - y_pred_range) / y_range)) * 100
                tpr2_range = tolerance_percentage_r2(y_range, y_pred_range, 0.05)
                
                logger.info(f"  Price Range: {min_price:,} - {max_price if max_price < float('inf') else 'inf'}")
                logger.info(f"    Samples: {len(y_range)}")
                logger.info(f"    R²: {r2_range:.4f}")
                logger.info(f"    RMSE: {rmse_range:.2f}")
                logger.info(f"    RMSLE: {rmsle_range:.4f}")
                logger.info(f"    MAE: {mae_range:.2f}")
                logger.info(f"    MAPE: {mape_range:.2f}%")
                logger.info(f"    TPR² (5%): {tpr2_range:.4f}")
            
            return metrics
        
        except Exception as e:
            logger.error("Error during model evaluation: %s", str(e), exc_info=True)
            raise

    def make_prediction_examples(self, n_samples=10):
        """
        Show prediction examples from the test set
        
        Args:
            n_samples: Number of examples to show
        """
        if self.best_model is None:
            logger.error("No best model available. Run tune_model() first.")
            return
        
        logger.info("\nPrediction Examples:")
        try:
            test_indices = np.random.choice(self.X_test.index, min(n_samples, len(self.X_test)), replace=False)
            for i, idx in enumerate(test_indices):
                X_sample = self.X_test.loc[[idx]]
                y_true = self.y_test.loc[idx]
                y_pred = self.best_model.predict(X_sample)[0]
                abs_error = abs(y_true - y_pred)
                pct_error = abs_error / y_true * 100
                
                logger.info("\nExample %d:", i+1)
                logger.info("  True Price: %s TL", f"{y_true:,.2f}")
                logger.info("  Predicted: %s TL", f"{y_pred:,.2f}")
                logger.info("  Absolute Error: %s TL", f"{abs_error:,.2f}")
                logger.info("  Percentage Error: %.2f%%", pct_error)
        
        except Exception as e:
            logger.error("Error generating prediction examples: %s", str(e), exc_info=True)
            raise

    def save_best_model(self):
        """
        Save the best model, scaler, encoder and feature information
        """
        if self.best_model is None:
            logger.error("No best model available to save.")
            return
        
        try:
            # 1. Zaman damgalı klasör oluşturma
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_dir = os.path.join("models", f"model_{timestamp}")
            os.makedirs(model_dir, exist_ok=True)
            
            # 2. Dosya yollarını tanımlama
            model_path = os.path.join(model_dir, "model.joblib")
            scaler_path = os.path.join(model_dir, "scaler.joblib")
            encoder_path = os.path.join(model_dir, "encoder.joblib")
            info_path = os.path.join(model_dir, "feature_info.joblib")
            
            # 3. Model bilgilerini hazırlama - kategorik değerleri de ekleyelim
            feature_info = {
                'model_type': self.best_model.__class__.__name__,
                'parameters': self.best_params,
                'numeric_features': self.numeric_features,
                'categorical_features': self.categorical_features,
                'feature_order': list(self.X_train.columns),
                'column_order': list(self.X_train.columns),  # column_order da ekleyelim
                'performance': {
                    'test_r2': r2_score(self.y_test, self.best_model.predict(self.X_test)),
                    'test_tpr2': tolerance_percentage_r2(self.y_test, self.best_model.predict(self.X_test)),
                    'test_rmse': np.sqrt(mean_squared_error(self.y_test, self.best_model.predict(self.X_test))),
                },
                'timestamp': timestamp
            }
            
            # Kategorik değerleri ekleyelim
            feature_info['categorical_values'] = {}
            for i, feature in enumerate(self.categorical_features):
                if hasattr(self.encoder, 'categories_') and i < len(self.encoder.categories_):
                    feature_info['categorical_values'][feature] = list(self.encoder.categories_[i])
                    logger.info(f"Added {len(self.encoder.categories_[i])} values for '{feature}'")
            
            # ÖZELLİK KATKILARINI HESAPLA VE KAYDET
            feature_info['feature_effects'] = self._calculate_feature_effects()
            
            # Özellikle mahalleler için etki değerlerini loglayalım
            neighborhood_effects = {}
            for feature, effect in feature_info['feature_effects'].items():
                if 'neighborhood_' in feature:
                    neighborhood_name = feature.replace('neighborhood_', '')
                    neighborhood_effects[neighborhood_name] = effect
            
            if neighborhood_effects:
                logger.info(f"Saved effect values for {len(neighborhood_effects)} neighborhoods")
            else:
                logger.warning("No neighborhood effects were calculated or saved")
            
            # 4. Bileşenleri kaydetme
            joblib.dump(self.best_model, model_path)       # Model nesnesi
            joblib.dump(self.scaler, scaler_path)          # Sayısal veri ölçekleyici
            joblib.dump(self.encoder, encoder_path)        # Kategorik veri kodlayıcı
            joblib.dump(feature_info, info_path)           # Meta veri ve model bilgileri
            
            # 5. Tüm model karşılaştırmalarını CSV olarak kaydetme
            if hasattr(self, 'model_performances'):
                performance_df = pd.DataFrame.from_dict(self.model_performances, orient='index')
                summary_path = os.path.join(model_dir, "model_comparison.csv")
                performance_df.to_csv(summary_path)
                logger.info("  Model comparison saved to: %s", summary_path)
            
            logger.info("\nModel saved successfully:")
            logger.info("  Directory: %s", model_dir)
            logger.info("  Model Type: %s", self.best_model.__class__.__name__)
            logger.info("  Test TPR²: %.4f", feature_info['performance']['test_tpr2'])
            
            return model_dir
        
        except Exception as e:
            logger.error("Error saving model: %s", str(e), exc_info=True)
            raise

    def _calculate_feature_effects(self):
        """
        Model tipine göre özelliklerin fiyatlara etkisini hesaplar
        
        Returns:
            Özellik-etki çiftlerini içeren sözlük
        """
        feature_effects = {}
        model_type = self.best_model.__class__.__name__
        
        try:
            if model_type in ['LinearRegression', 'Ridge', 'Lasso', 'ElasticNet', 'BayesianRidge']:
                # Doğrusal modeller için katsayıları doğrudan kullan
                for feature, coef in zip(self.X_train.columns, self.best_model.coef_):
                    feature_effects[feature] = float(coef)
                    
            elif model_type in ['RandomForestRegressor', 'GradientBoostingRegressor', 'DecisionTreeRegressor']:
                # Ağaç tabanlı modeller için feature importance kullan
                for feature, imp in zip(self.X_train.columns, self.best_model.feature_importances_):
                    feature_effects[feature] = float(imp)
                    
            elif model_type == 'XGBRegressor':
                # XGBoost için gain based feature importance 
                if hasattr(self.best_model, 'get_booster'):
                    importance_dict = self.best_model.get_booster().get_score(importance_type='gain')
                    # Feature isimleri farklı olabilir - eşleştirme yapalım
                    for f_idx, feature in enumerate(self.X_train.columns):
                        f_id = f'f{f_idx}'
                        if f_id in importance_dict:
                            feature_effects[feature] = float(importance_dict[f_id])
                        else:
                            feature_effects[feature] = 0.0
                else:
                    # Fallback olarak feature_importances_ kullan
                    for feature, imp in zip(self.X_train.columns, self.best_model.feature_importances_):
                        feature_effects[feature] = float(imp)
                        
            elif model_type == 'LGBMRegressor':
                # LightGBM için feature importance
                if hasattr(self.best_model, 'feature_importances_'):
                    for feature, imp in zip(self.X_train.columns, self.best_model.feature_importances_):
                        feature_effects[feature] = float(imp)
                        
            elif model_type == 'CatBoostRegressor':
                # CatBoost için feature importance
                if hasattr(self.best_model, 'get_feature_importance'):
                    importances = self.best_model.get_feature_importance()
                    for feature, imp in zip(self.X_train.columns, importances):
                        feature_effects[feature] = float(imp)
            
            # Özellikle ilçe ve mahallelerle ilgili etkileri grupla
            district_effects = {}
            neighborhood_effects = {}
            
            # Diğer kategorik değişkenleri de grupla
            categorical_groups = {cat: {} for cat in self.categorical_features}
            
            # Tüm özellik etkilerini kategorilere göre grupla
            for feature, effect in feature_effects.items():
                for cat in self.categorical_features:
                    prefix = f"{cat}_"
                    if feature.startswith(prefix):
                        value = feature[len(prefix):]
                        categorical_groups[cat][value] = effect
                        
                        if cat == 'district':
                            district_effects[value] = effect
                        elif cat == 'neighborhood':
                            neighborhood_effects[value] = effect
            
            # Etki değerlerini sayı ve ortalamasını logla
            logger.info(f"Calculated effects for {len(neighborhood_effects)} neighborhoods")
            logger.info(f"Calculated effects for {len(district_effects)} districts")
            
            if neighborhood_effects:
                avg_effect = sum(neighborhood_effects.values()) / len(neighborhood_effects)
                logger.info(f"Average neighborhood effect: {avg_effect:.6f}")
                
                # En etkili mahalleleri göster
                sorted_neighborhoods = sorted(neighborhood_effects.items(), key=lambda x: x[1], reverse=True)
                top_5 = sorted_neighborhoods[:5]
                bottom_5 = sorted_neighborhoods[-5:]
                logger.info(f"Top 5 neighborhoods by effect: {top_5}")
                logger.info(f"Bottom 5 neighborhoods by effect: {bottom_5}")
            
            # Model tipine göre normalize et
            if model_type in ['RandomForestRegressor', 'GradientBoostingRegressor', 'XGBRegressor', 'LGBMRegressor', 'CatBoostRegressor']:
                # Ağaç modelleri için önemleri normalize et (sıralama değerlerini korur)
                for cat, values in categorical_groups.items():
                    if values:
                        max_effect = max(values.values())
                        if max_effect > 0:
                            for value in values:
                                values[value] = values[value] / max_effect
            
            # Gruplandırılmış etkileri ana sözlüğe ekle
            feature_effects['_grouped'] = {
                'district': district_effects,
                'neighborhood': neighborhood_effects,
                **{cat: vals for cat, vals in categorical_groups.items() 
                   if cat not in ['district', 'neighborhood']}
            }
            
            logger.info(f"Calculated feature effects for {len(feature_effects)} features")
            return feature_effects
            
        except Exception as e:
            logger.error(f"Error calculating feature effects: {str(e)}", exc_info=True)
            # Boş sözlük döndür
            return {}

if __name__ == "__main__":
    try:
        logger.info("Starting House Price Prediction Model Training and Tuning")
        
        # Create output directories if they don't exist
        os.makedirs('models', exist_ok=True)
        
        # Initialize model
        house_price_model = HousePriceModel("data/all_data.csv", random_state=42)
        
        # Load and preprocess data
        house_price_model.load_data()
        house_price_model.preprocess_data()
        
        # Train and evaluate base models
        house_price_model.train_and_evaluate_base_models()
        
        # Tune the models
        house_price_model.tune_model(n_trials=200)
        
        # Evaluate the best model
        house_price_model.evaluate_model()
        
        # Show prediction examples
        house_price_model.make_prediction_examples(n_samples=10)
        
        # Save the best model
        model_dir = house_price_model.save_best_model()
        logger.info("Best model saved to: %s", model_dir)
        
        logger.info("\nModel training and tuning completed successfully")
    
    except Exception as e:
        logger.error("Error in house price model training: %s", str(e), exc_info=True)
        raise