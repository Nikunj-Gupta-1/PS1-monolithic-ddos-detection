import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class OptimizedDDoSModelTrainer:
    """
    Enhanced DDoS detection model trainer with comprehensive output and visualizations
    """
    
    def __init__(self, binary_classification=True):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_selector = SelectKBest(score_func=f_classif, k=25)
        self.model = None
        self.feature_names = None
        self.binary_classification = binary_classification
        self.training_metrics = {}
        
    def load_and_preprocess_data(self, file_path):
        """Enhanced data preprocessing with detailed progress reporting"""
        print("🔄 Loading and preprocessing dataset...")
        print("=" * 60)
        
        df = pd.read_csv(file_path)
        
        # Convert to binary classification if specified
        if self.binary_classification:
            print("📊 Converting to binary classification (Attack vs Benign)")
            original_classes = df[' Label'].value_counts()
            df[' Label'] = df[' Label'].apply(
                lambda x: 'BENIGN' if x == 'BENIGN' else 'ATTACK'
            )
            print(f"   ✓ Converted {len(original_classes)} classes to 2 classes")
        
        print(f"\n📈 Dataset Overview:")
        print(f"   Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        
        # Display class distribution with percentages
        class_counts = df[' Label'].value_counts()
        print(f"\n📊 Class Distribution:")
        for class_name, count in class_counts.items():
            percentage = (count / len(df)) * 100
            print(f"   {class_name}: {count:,} samples ({percentage:.1f}%)")
        
        # Clean data
        print(f"\n🧹 Cleaning data...")
        original_shape = df.shape
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        
        if original_shape[1] != df.shape[1]:
            print(f"   ✓ Removed {original_shape[1] - df.shape[1]} unnamed columns")
        
        # Separate features and labels
        X = df.drop([' Label'], axis=1)
        y = df[' Label']
        
        # Keep only numeric columns
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        X = X[numeric_columns]
        print(f"   ✓ Using {len(numeric_columns)} numeric features")
        
        # Balance the dataset
        print(f"\n⚖️  Balancing dataset...")
        X_balanced, y_balanced = self.balance_dataset(X, y)
        
        balanced_counts = pd.Series(y_balanced).value_counts()
        print(f"   ✓ Balanced dataset shape: {X_balanced.shape}")
        print(f"   ✓ Balanced distribution:")
        for class_name, count in balanced_counts.items():
            percentage = (count / len(y_balanced)) * 100
            print(f"     {class_name}: {count:,} samples ({percentage:.1f}%)")
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y_balanced)
        
        print(f"\n✅ Preprocessing complete!")
        print("=" * 60)
        
        return X_balanced, y_encoded, y_balanced
    
    def balance_dataset(self, X, y):
        """Create properly balanced dataset with progress reporting"""
        if self.binary_classification:
            target_samples = min(50000, min(pd.Series(y).value_counts()))
            print(f"   Target samples per class: {target_samples:,}")
            
            undersampler = RandomUnderSampler(
                sampling_strategy={
                    'BENIGN': target_samples,
                    'ATTACK': target_samples
                },
                random_state=42
            )
            
            X_balanced, y_balanced = undersampler.fit_resample(X, y)
        else:
            print("   Using SMOTE for multi-class balancing...")
            smote = SMOTE(random_state=42, k_neighbors=3)
            X_balanced, y_balanced = smote.fit_resample(X, y)
        
        return X_balanced, y_balanced
    
    def train_model(self, X, y, test_size=0.2):
        """Train model with comprehensive evaluation and reporting"""
        print("\n🚀 Starting model training...")
        print("=" * 60)
        
        # Split data
        print("📊 Splitting data with stratification...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"   Training set: {len(X_train):,} samples")
        print(f"   Test set: {len(X_test):,} samples")
        
        # Feature selection
        print(f"\n🎯 Selecting best features...")
        X_train_selected = self.feature_selector.fit_transform(X_train, y_train)
        X_test_selected = self.feature_selector.transform(X_test)
        
        selected_indices = self.feature_selector.get_support()
        self.feature_names = X.columns[selected_indices].tolist()
        print(f"   ✓ Selected {len(self.feature_names)} most important features")
        
        # Feature scaling
        print(f"\n⚖️  Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train_selected)
        X_test_scaled = self.scaler.transform(X_test_selected)
        print(f"   ✓ Features scaled using StandardScaler")
        
        # Model configuration
        print(f"\n🔧 Configuring ensemble models...")
        rf = RandomForestClassifier(
            n_estimators=1000, max_depth=20, min_samples_split=10,
            min_samples_leaf=4, class_weight='balanced', random_state=42, n_jobs=-1
        )
        
        xgb = XGBClassifier(
            n_estimators=1000, max_depth=10, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, scale_pos_weight=1,
            random_state=42, n_jobs=-1, eval_metric='logloss'
        )
        
        lgb = LGBMClassifier(
            n_estimators=1000, max_depth=10, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, class_weight='balanced',
            random_state=42, n_jobs=-1, verbose=-1
        )
        
        self.model = VotingClassifier(
            estimators=[('RandomForest', rf), ('XGBoost', xgb), ('LightGBM', lgb)],
            voting='soft', n_jobs=-1
        )
        
        print(f"   ✓ Ensemble configured: RandomForest + XGBoost + LightGBM")
        
        # Cross-validation
        print(f"\n🔄 Performing cross-validation...")
        cv_scores = cross_val_score(
            self.model, X_train_scaled, y_train,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='accuracy', n_jobs=-1
        )
        
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        print(f"   Cross-validation results:")
        print(f"     Mean Accuracy: {cv_mean:.4f} ({cv_mean*100:.2f}%)")
        print(f"     Std Deviation: ±{cv_std:.4f}")
        print(f"     95% Confidence: {cv_mean:.4f} ± {cv_std * 1.96:.4f}")
        
        # Train final model
        print(f"\n🎯 Training final ensemble model...")
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)
        
        # Store training metrics
        self.training_metrics = {
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'test_accuracy': accuracy_score(y_test, y_pred),
            'feature_count': len(self.feature_names),
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }
        
        # Comprehensive evaluation
        self.evaluate_model_performance(y_test, y_pred, y_pred_proba)
        
        # Create visualizations
        self.create_performance_visualizations(y_test, y_pred)
        
        print(f"\n✅ Model training completed!")
        print("=" * 60)
        
        return X_test_scaled, y_test, y_pred
    
    def evaluate_model_performance(self, y_true, y_pred, y_pred_proba):
        """Comprehensive model evaluation with clear metrics"""
        print(f"\n🎯 MODEL PERFORMANCE SUMMARY")
        print("=" * 60)
        
        # Overall accuracy
        accuracy = accuracy_score(y_true, y_pred)
        print(f"🎯 Overall Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Performance interpretation
        if accuracy >= 0.99:
            performance_level = "🌟 EXCELLENT"
            interpretation = "Outstanding performance - ready for production"
        elif accuracy >= 0.95:
            performance_level = "✅ VERY GOOD"
            interpretation = "High performance - suitable for deployment"
        elif accuracy >= 0.90:
            performance_level = "⚠️  GOOD"
            interpretation = "Adequate performance - consider improvements"
        else:
            performance_level = "❌ NEEDS IMPROVEMENT"
            interpretation = "Poor performance - requires optimization"
        
        print(f"📊 Performance Level: {performance_level}")
        print(f"💡 Interpretation: {interpretation}")
        
        # Class-specific metrics
        precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)
        class_names = self.label_encoder.classes_
        
        print(f"\n📊 Detailed Performance by Class:")
        print("-" * 50)
        
        for i, class_name in enumerate(class_names):
            print(f"\n{class_name.upper()}:")
            print(f"  Precision: {precision[i]:.4f} ({precision[i]*100:.2f}%)")
            print(f"  Recall:    {recall[i]:.4f} ({recall[i]*100:.2f}%)")
            print(f"  F1-Score:  {f1[i]:.4f} ({f1[i]*100:.2f}%)")
            print(f"  Support:   {support[i]:,} samples")
        
        # Average metrics
        precision_macro = np.mean(precision)
        recall_macro = np.mean(recall)
        f1_macro = np.mean(f1)
        
        print(f"\n📈 Summary Metrics:")
        print(f"  Average Precision: {precision_macro:.4f} ({precision_macro*100:.2f}%)")
        print(f"  Average Recall:    {recall_macro:.4f} ({recall_macro*100:.2f}%)")
        print(f"  Average F1-Score:  {f1_macro:.4f} ({f1_macro*100:.2f}%)")
        
        # AUC score for binary classification
        if len(class_names) == 2:
            auc_score = roc_auc_score(y_true, y_pred_proba[:, 1])
            print(f"  AUC Score:         {auc_score:.4f} ({auc_score*100:.2f}%)")
        
        print("=" * 60)
    
    def create_performance_visualizations(self, y_true, y_pred):
        """Create clear performance visualizations"""
        class_names = self.label_encoder.classes_
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names, ax=axes[0])
        axes[0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('True Label', fontsize=12)
        axes[0].set_xlabel('Predicted Label', fontsize=12)
        
        # Performance metrics bar chart
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
        
        x = np.arange(len(class_names))
        width = 0.25
        
        axes[1].bar(x - width, precision, width, label='Precision', alpha=0.8, color='skyblue')
        axes[1].bar(x, recall, width, label='Recall', alpha=0.8, color='lightgreen')
        axes[1].bar(x + width, f1, width, label='F1-Score', alpha=0.8, color='salmon')
        
        axes[1].set_xlabel('Classes', fontsize=12)
        axes[1].set_ylabel('Score', fontsize=12)
        axes[1].set_title('Performance Metrics by Class', fontsize=14, fontweight='bold')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(class_names)
        axes[1].legend()
        axes[1].set_ylim(0, 1.1)
        
        # Add percentage labels on bars
        for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
            axes[1].text(i - width, p + 0.01, f'{p:.3f}', ha='center', va='bottom', fontsize=10)
            axes[1].text(i, r + 0.01, f'{r:.3f}', ha='center', va='bottom', fontsize=10)
            axes[1].text(i + width, f + 0.01, f'{f:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('ddos_model_performance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Performance visualization saved as 'ddos_model_performance.png'")
    
    def save_model(self, model_path=__import__('os').path.abspath(__import__('os').path.join(__import__('os').path.dirname(__file__), '../raw data collected for project/optimized_ddos_model.pkl'))):
        """Save model with comprehensive metadata"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_selector': self.feature_selector,
            'feature_names': self.feature_names,
            'binary_classification': self.binary_classification,
            'training_metrics': self.training_metrics
        }
        
        joblib.dump(model_data, model_path)
        
        # Save detailed summary
        with open('model_training_summary.txt', 'w') as f:
            f.write("DDoS Detection Model Training Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Model Type: {'Binary' if self.binary_classification else 'Multi-class'} Classification\n")
            f.write(f"Test Accuracy: {self.training_metrics['test_accuracy']:.4f} ({self.training_metrics['test_accuracy']*100:.2f}%)\n")
            f.write(f"Cross-validation Mean: {self.training_metrics['cv_mean']:.4f}\n")
            f.write(f"Cross-validation Std: {self.training_metrics['cv_std']:.4f}\n")
            f.write(f"Features Used: {self.training_metrics['feature_count']}\n")
            f.write(f"Training Samples: {self.training_metrics['training_samples']:,}\n")
            f.write(f"Test Samples: {self.training_metrics['test_samples']:,}\n")
            f.write(f"Classes: {list(self.label_encoder.classes_)}\n")
            f.write(f"Selected Features: {self.feature_names}\n")
        
        print(f"💾 Model saved to: {model_path}")
        print(f"📄 Training summary saved to: model_training_summary.txt")

# Enhanced usage function
def train_optimized_model():
    """Train the optimized DDoS detection model with clear output"""
    print("🚀 Starting DDoS Detection Model Training")
    print("=" * 60)
    
    trainer = OptimizedDDoSModelTrainer(binary_classification=True)
    
    # Load and preprocess data
    X, y_encoded, y_original = trainer.load_and_preprocess_data(__import__('os').path.abspath(__import__('os').path.join(__import__('os').path.dirname(__file__), '../raw data collected for project/CIC_DDoS2019_To_Use.csv')))
    
    # Train model
    X_test, y_test, y_pred = trainer.train_model(X, y_encoded)
    
    # Save model
    trainer.save_model(__import__('os').path.abspath(__import__('os').path.join(__import__('os').path.dirname(__file__), '../raw data collected for project/optimized_ddos_model.pkl')))
    
    print(f"\n🎉 TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"📁 Files Created:")
    print(f"   ✓ optimized_ddos_model.pkl (trained model)")
    print(f"   ✓ model_training_summary.txt (detailed summary)")
    print(f"   ✓ ddos_model_performance.png (performance visualization)")
    print("=" * 60)
    
    return trainer

if __name__ == "__main__":
    trainer = train_optimized_model()
