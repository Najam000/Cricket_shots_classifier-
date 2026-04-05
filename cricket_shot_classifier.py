"""
╔══════════════════════════════════════════════════════════════════════╗
║          Cricket Shot Classifier  —  PyCharm Project                ║
║  Dataset : https://www.kaggle.com/datasets/aneesh10/cricket-shot-   ║
║            dataset/data                                             ║
║  Author  : Najam Ul Hassan                                          ║
║  Trainer : Engr. Aamir Jamil                                        ║
║  Course  : Artificial Intelligence (ML; DL; Communication)          ║
║  Batch   : Batch II                                                 ║
╚══════════════════════════════════════════════════════════════════════╝

Run:  python cricket_shot_classifier.py
"""

# ══════════════════════════════════════════════════════════════════════
# SECTION 1 — Imports & Setup
# ══════════════════════════════════════════════════════════════════════
import os
import warnings
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

warnings.filterwarnings('ignore')

from PIL import Image
from skimage.feature import hog
from skimage.color import rgb2gray
from skimage.transform import resize

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    train_test_split, cross_val_score,
    GridSearchCV, StratifiedKFold
)
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
)

sns.set_theme(style='whitegrid', palette='Set2')
plt.rcParams['figure.figsize'] = (10, 6)
np.random.seed(42)

print('=' * 60)
print('  🏏  Cricket Shot Classifier')
print('  Author  : Najam Ul Hassan')
print('  Trainer : Engr. Aamir Jamil')
print('  Course  : AI (ML; DL; Communication) | Batch II')
print('=' * 60)
print('✅  All imports successful!\n')

# ══════════════════════════════════════════════════════════════════════
# SECTION 2 — Configuration & Dataset Loading
# ══════════════════════════════════════════════════════════════════════
# ──  Update DATASET_DIR to your local path after downloading from Kaggle
#    https://www.kaggle.com/datasets/aneesh10/cricket-shot-dataset/data
# ──────────────────────────────────────────────────────────────────────
DATASET_DIR = './cricket-shot-dataset'   # ← change if needed
IMG_SIZE    = (64, 64)
CLASSES     = ['pull_shot', 'leg_glance_flick', 'drive', 'sweep']
OUTPUT_DIR  = './outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_hog_features(img_array, img_size=(64, 64)):
    """RGB image array  →  HOG feature vector (~1 764 dims)."""
    img_resized = resize(img_array, img_size, anti_aliasing=True)
    img_gray    = rgb2gray(img_resized) if img_resized.ndim == 3 else img_resized
    return hog(
        img_gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        feature_vector=True,
    )


def load_dataset(dataset_dir, classes, img_size=(64, 64)):
    """Walk class folders, extract HOG features, return X, y arrays."""
    X, y, class_counts = [], [], {}
    for label, cls in enumerate(classes):
        cls_dir = os.path.join(dataset_dir, cls)
        if not os.path.isdir(cls_dir):
            print(f'  ⚠️  Folder not found: {cls_dir}')
            continue
        count = 0
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                try:
                    img  = np.array(Image.open(os.path.join(cls_dir, fname)).convert('RGB'))
                    feat = extract_hog_features(img, img_size)
                    X.append(feat)
                    y.append(label)
                    count += 1
                except Exception as e:
                    print(f'    Skip {fname}: {e}')
        class_counts[cls] = count
        print(f'  {cls:<22} →  {count} images')
    return np.array(X), np.array(y), class_counts


print('📂  Loading dataset …')
X, y, class_counts = load_dataset(DATASET_DIR, CLASSES, IMG_SIZE)
print(f'\n✅  {X.shape[0]} samples | {X.shape[1]} HOG features | {len(CLASSES)} classes\n')

# ══════════════════════════════════════════════════════════════════════
# SECTION 3 — Exploratory Data Analysis (EDA)
# ══════════════════════════════════════════════════════════════════════
print('📊  Running EDA …')

counts = list(class_counts.values())
labels = [c.replace('_', ' ').title() for c in class_counts.keys()]

# 3a. Class distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].bar(labels, counts, color=sns.color_palette('Set2', 4), edgecolor='black')
axes[0].set_title('Class Distribution', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Number of Images')
for i, v in enumerate(counts):
    axes[0].text(i, v + 10, str(v), ha='center', fontweight='bold')
axes[1].pie(
    counts, labels=labels, autopct='%1.1f%%',
    colors=sns.color_palette('Set2', 4), startangle=140,
    wedgeprops={'edgecolor': 'white', 'linewidth': 1.5},
)
axes[1].set_title('Class Proportion', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'eda_class_distribution.png'), dpi=150, bbox_inches='tight')
plt.show()

# 3b. Sample images
if all(os.path.isdir(os.path.join(DATASET_DIR, c)) for c in CLASSES):
    fig, axes = plt.subplots(4, 5, figsize=(16, 12))
    fig.suptitle('Sample Cricket Shot Images per Class', fontsize=16, fontweight='bold')
    for row, cls in enumerate(CLASSES):
        cls_dir = os.path.join(DATASET_DIR, cls)
        imgs = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for col, fname in enumerate(np.random.choice(imgs, min(5, len(imgs)), replace=False)):
            img = mpimg.imread(os.path.join(cls_dir, fname))
            axes[row][col].imshow(img)
            axes[row][col].axis('off')
            if col == 0:
                axes[row][col].set_ylabel(
                    cls.replace('_', ' ').title(),
                    fontsize=10, fontweight='bold', rotation=0, labelpad=80, va='center',
                )
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'eda_sample_images.png'), dpi=150, bbox_inches='tight')
    plt.show()

# 3c. HOG feature statistics
df_feat = pd.DataFrame({
    'class':    y,
    'mean_hog': X.mean(axis=1),
    'std_hog':  X.std(axis=1),
    'max_hog':  X.max(axis=1),
})
df_feat['class_name'] = df_feat['class'].map(
    {i: c.replace('_', ' ').title() for i, c in enumerate(CLASSES)}
)
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, col, title in zip(axes,
                           ['mean_hog', 'std_hog', 'max_hog'],
                           ['Mean HOG Value', 'Std HOG Value', 'Max HOG Value']):
    for cls_name in df_feat['class_name'].unique():
        ax.hist(df_feat[df_feat['class_name'] == cls_name][col], bins=30, alpha=0.6, label=cls_name)
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Value')
    ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'eda_hog_stats.png'), dpi=150, bbox_inches='tight')
plt.show()
print('✅  EDA complete\n')

# ══════════════════════════════════════════════════════════════════════
# SECTION 4 — Preprocessing
# ══════════════════════════════════════════════════════════════════════
print('⚙️   Preprocessing …')

# Train/test split (80 / 20) — stratified
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f'  Train : {X_train.shape[0]} samples')
print(f'  Test  : {X_test.shape[0]} samples')

# StandardScaler — fit on TRAIN only (no data leakage)
scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f'  Feature mean after scaling : {X_train_sc.mean():.4f}  (≈ 0.0)')
print(f'  Feature std  after scaling : {X_train_sc.std():.4f}   (≈ 1.0)')
print('✅  Preprocessing complete — no data leakage!\n')

# ══════════════════════════════════════════════════════════════════════
# SECTION 5 — Train & Compare All Models (Baseline)
# ══════════════════════════════════════════════════════════════════════
print('🤖  Training baseline models …')

all_models = {
    'SVM (RBF)'           : SVC(kernel='rbf', random_state=42, probability=True),
    'Random Forest'       : RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting'   : GradientBoostingClassifier(n_estimators=100, random_state=42),
    'Extra Trees'         : ExtraTreesClassifier(n_estimators=100, random_state=42),
    'Logistic Regression' : LogisticRegression(max_iter=1000, random_state=42),
    'K-Nearest Neighbors' : KNeighborsClassifier(n_neighbors=5),
    'Naive Bayes'         : GaussianNB(),
}

results = []
print(f'\n  {"Model":<25} {"Train Acc":>10} {"Test Acc":>10}')
print('  ' + '─' * 48)
for name, clf in all_models.items():
    clf.fit(X_train_sc, y_train)
    tr = accuracy_score(y_train, clf.predict(X_train_sc))
    te = accuracy_score(y_test,  clf.predict(X_test_sc))
    results.append({'Model': name, 'Train Accuracy': tr, 'Test Accuracy': te})
    print(f'  {name:<25} {tr:>10.4f} {te:>10.4f}')

df_results = pd.DataFrame(results).sort_values('Test Accuracy', ascending=False)
print(f'\n  Top 3:\n{df_results.head(3).to_string(index=False)}\n')

# Visualise
fig, ax = plt.subplots(figsize=(12, 6))
x, w = np.arange(len(df_results)), 0.35
ax.bar(x - w/2, df_results['Train Accuracy'], w, label='Train Acc', color='steelblue', alpha=0.85, edgecolor='black')
bars2 = ax.bar(x + w/2, df_results['Test Accuracy'],  w, label='Test Acc',  color='coral',    alpha=0.85, edgecolor='black')
for b in bars2:
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005,
            f'{b.get_height():.3f}', ha='center', va='bottom', fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(df_results['Model'], rotation=25, ha='right')
ax.set_ylabel('Accuracy')
ax.set_title('Baseline Model Comparison — Cricket Shot Classification', fontsize=14, fontweight='bold')
ax.legend()
ax.set_ylim(0, 1.1)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'baseline_comparison.png'), dpi=150, bbox_inches='tight')
plt.show()

# ══════════════════════════════════════════════════════════════════════
# SECTION 6 — Cross-Validation (Top 3 Models)
# ══════════════════════════════════════════════════════════════════════
print('🔁  5-Fold Cross-Validation on Top 3 models …\n')

top3_names  = df_results['Model'].head(3).tolist()
top3_models = {n: all_models[n] for n in top3_names}
cv          = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_results = []
print(f'  {"Model":<25} {"CV Mean":>9} {"± Std":>8}')
print('  ' + '─' * 46)
for name, clf in top3_models.items():
    scores = cross_val_score(clf, X_train_sc, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
    cv_results.append({'Model': name, 'CV Mean': scores.mean(), 'CV Std': scores.std(), 'CV Scores': scores})
    print(f'  {name:<25} {scores.mean():>9.4f} {scores.std():>8.4f}')
    print(f'    Folds: {np.round(scores, 4)}')

fig, ax = plt.subplots(figsize=(9, 5))
bp = ax.boxplot([r['CV Scores'] for r in cv_results], patch_artist=True)
for patch, color in zip(bp['boxes'], sns.color_palette('Set2', 3)):
    patch.set_facecolor(color)
ax.set_xticklabels(top3_names, rotation=15, ha='right')
ax.set_ylabel('CV Accuracy')
ax.set_title('5-Fold CV Distribution — Top 3 Models', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'cv_boxplot.png'), dpi=150, bbox_inches='tight')
plt.show()
print()

# ══════════════════════════════════════════════════════════════════════
# SECTION 7 — Hyperparameter Tuning with GridSearchCV
# ══════════════════════════════════════════════════════════════════════
best_cv_name = max(cv_results, key=lambda r: r['CV Mean'])['Model']
print(f'🔧  Tuning: {best_cv_name}')

if 'SVM' in best_cv_name:
    param_grid = {
        'C':      [0.1, 1, 10, 100],
        'gamma':  ['scale', 'auto', 0.001, 0.01],
        'kernel': ['rbf', 'poly'],
    }
    base_clf = SVC(probability=True, random_state=42)
elif 'Random Forest' in best_cv_name:
    param_grid = {
        'n_estimators':      [100, 200, 300],
        'max_depth':         [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf':  [1, 2],
    }
    base_clf = RandomForestClassifier(random_state=42)
else:
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1, 0.2],
        'max_depth': [3, 5],
    }
    base_clf = GradientBoostingClassifier(random_state=42)

grid_search = GridSearchCV(
    estimator  = base_clf,
    param_grid = param_grid,
    cv         = StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring    = 'accuracy',
    n_jobs     = -1,
    verbose    = 1,
)
grid_search.fit(X_train_sc, y_train)
print(f'\n  ✅  Best Params : {grid_search.best_params_}')
print(f'      Best CV Acc : {grid_search.best_score_:.4f}\n')

# Visualise GridSearch
cv_df  = pd.DataFrame(grid_search.cv_results_)
top_n  = cv_df.nlargest(15, 'mean_test_score')
fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(range(len(top_n)), top_n['mean_test_score'],
       yerr=top_n['std_test_score'], capsize=4,
       color='mediumseagreen', edgecolor='black', alpha=0.8)
ax.axhline(grid_search.best_score_, color='red', linestyle='--',
           label=f'Best: {grid_search.best_score_:.4f}')
ax.set_xlabel('Parameter Combination Rank')
ax.set_ylabel('Mean CV Accuracy')
ax.set_title(f'GridSearchCV Results — {best_cv_name}', fontsize=14, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'gridsearch_results.png'), dpi=150, bbox_inches='tight')
plt.show()

# ══════════════════════════════════════════════════════════════════════
# SECTION 8 — Final Evaluation — Best Model
# ══════════════════════════════════════════════════════════════════════
print('📈  Final evaluation …\n')

best_model   = grid_search.best_estimator_
y_pred       = best_model.predict(X_test_sc)
test_acc     = accuracy_score(y_test, y_pred)
class_labels = [c.replace('_', ' ').title() for c in CLASSES]

print('=' * 55)
print(f'  FINAL TEST ACCURACY : {test_acc:.4f}  ({test_acc * 100:.2f} %)')
print('=' * 55)
print('\nClassification Report:')
print(classification_report(y_test, y_pred, target_names=class_labels))

# Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
cm      = confusion_matrix(y_test, y_pred)
cm_norm = confusion_matrix(y_test, y_pred, normalize='true')
ConfusionMatrixDisplay(cm,      display_labels=class_labels).plot(ax=axes[0], cmap='Blues',  colorbar=False)
ConfusionMatrixDisplay(cm_norm, display_labels=class_labels).plot(ax=axes[1], cmap='Greens', colorbar=False)
axes[0].set_title('Confusion Matrix (Counts)',     fontweight='bold')
axes[1].set_title('Confusion Matrix (Normalised)', fontweight='bold')
for ax in axes:
    ax.tick_params(axis='x', rotation=30)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
plt.show()

# Per-class accuracy bar
per_class_acc = cm_norm.diagonal()
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(class_labels, per_class_acc, color=sns.color_palette('Set2', 4), edgecolor='black')
for b in bars:
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
            f'{b.get_height() * 100:.1f}%', ha='center', fontweight='bold')
ax.set_ylabel('Accuracy')
ax.set_ylim(0, 1.15)
ax.set_title('Per-Class Accuracy', fontsize=14, fontweight='bold')
ax.axhline(test_acc, color='red', linestyle='--', label=f'Overall: {test_acc:.3f}')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'per_class_accuracy.png'), dpi=150, bbox_inches='tight')
plt.show()

# ══════════════════════════════════════════════════════════════════════
# SECTION 9 — Save Model & Scaler with Pickle
# ══════════════════════════════════════════════════════════════════════
MODEL_PATH  = 'cricket_shot_model.pkl'
SCALER_PATH = 'cricket_shot_scaler.pkl'

with open(MODEL_PATH,  'wb') as f:
    pickle.dump(best_model, f)
with open(SCALER_PATH, 'wb') as f:
    pickle.dump(scaler, f)

print(f'\n✅  Model  saved  →  {MODEL_PATH}')
print(f'✅  Scaler saved  →  {SCALER_PATH}')

# Reload & verify
with open(MODEL_PATH,  'rb') as f:
    loaded_model  = pickle.load(f)
with open(SCALER_PATH, 'rb') as f:
    loaded_scaler = pickle.load(f)

verify_acc = accuracy_score(y_test, loaded_model.predict(loaded_scaler.transform(X_test)))
print(f'\n🔁  Reloaded model accuracy : {verify_acc:.4f}  (matches {test_acc:.4f})')

print('\n' + '=' * 60)
print('  PROJECT SUMMARY')
print('=' * 60)
print(f'  Dataset     : Cricket Shot Dataset (Kaggle — aneesh10)')
print(f'  Classes     : {CLASSES}')
print(f'  Best Model  : {best_cv_name}')
print(f'  Best Params : {grid_search.best_params_}')
print(f'  Test Acc    : {test_acc * 100:.2f} %')
print(f'  Outputs     : {OUTPUT_DIR}/')
print(f'  Model       : {MODEL_PATH}')
print(f'  Scaler      : {SCALER_PATH}')
print('=' * 60)
print('\n🏏  Done! Open outputs/ to view all saved plots.')
