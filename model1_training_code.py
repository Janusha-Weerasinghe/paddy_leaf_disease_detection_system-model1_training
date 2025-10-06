"""
Comprehensive Disease Detection pipeline:
- ResNet50 transfer learning
- Multi-scale block + CBAM attention
- Traditional augmentation + optional GAN-based augmentation (DCGAN)
- ROI detection helper
- Full evaluation (precision, recall, f1, per-class AUC)
- Grad-CAM
- TFLite conversion with quantization
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_fscore_support
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import itertools
import random
import warnings
warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ------------------------
# USER CONFIG
# ------------------------

# ---- DATASET PATHS ----
# ---- Download datasets directly with kagglehub ----
import kagglehub

# Download datasets and get paths (runs once; cached locally)
nizorogbezuode_rice_leaf_images_path = kagglehub.dataset_download('nizorogbezuode/rice-leaf-images')
print(f"Dataset path for nizorogbezuode/rice-leaf-images: {nizorogbezuode_rice_leaf_images_path}")

# dedeikhsandwisaputra_rice_leafs_disease_dataset_path = kagglehub.dataset_download('dedeikhsandwisaputra/rice-leafs-disease-dataset')
# print(f"Dataset path for dedeikhsandwisaputra/rice-leafs-disease-dataset: {dedeikhsandwisaputra_rice_leafs_disease_dataset_path}")

nirmalsankalana_rice_leaf_disease_image_path = kagglehub.dataset_download('nirmalsankalana/rice-leaf-disease-image')
print(f"Dataset path for nirmalsankalana/rice-leaf-disease-image: {nirmalsankalana_rice_leaf_disease_image_path}")

# anshulm257_rice_disease_dataset_path = kagglehub.dataset_download('anshulm257/rice-disease-dataset')
# print(f"Dataset path for anshulm257/rice-disease-dataset: {anshulm257_rice_disease_dataset_path}")
#
# hduytrng_mendeley_rice_disease_dataset_path = kagglehub.dataset_download('hduytrng/mendeley-rice-disease-dataset')
# print(f"Dataset path for hduytrng/mendeley-rice-disease-dataset: {hduytrng_mendeley_rice_disease_dataset_path}")

# ------------------------
# Define dataset paths dictionary mapping disease class to dataset folders

dataset_paths = {
    "Bacterialblight": [
        # os.path.join(hduytrng_mendeley_rice_disease_dataset_path, "Augmented Images", "Bacterial Leaf Blight"),
        os.path.join(nirmalsankalana_rice_leaf_disease_image_path, "Bacterialblight"),
        # os.path.join(dedeikhsandwisaputra_rice_leafs_disease_dataset_path, "RiceLeafsDisease", "train", "bacterial_leaf_blight"),
    ],
    "Blast": [
        # os.path.join(hduytrng_mendeley_rice_disease_dataset_path, "Augmented Images", "Leaf Blast"),
        os.path.join(nirmalsankalana_rice_leaf_disease_image_path, "Blast"),
        # os.path.join(dedeikhsandwisaputra_rice_leafs_disease_dataset_path, "RiceLeafsDisease", "train", "leaf_blast"),
    ],
    "Brownspot": [
        #os.path.join(hduytrng_mendeley_rice_disease_dataset_path, "Augmented Images", "Brown Spot"),
        os.path.join(nirmalsankalana_rice_leaf_disease_image_path, "Brownspot"),
        #os.path.join(dedeikhsandwisaputra_rice_leafs_disease_dataset_path, "RiceLeafsDisease", "train", "brown_spot"),
    ],
    "Healthy": [
        #os.path.join(hduytrng_mendeley_rice_disease_dataset_path, "Augmented Images", "Healthy Rice Leaf"),
        os.path.join(nizorogbezuode_rice_leaf_images_path, "rice_images", "_Healthy"),
        #os.path.join(dedeikhsandwisaputra_rice_leafs_disease_dataset_path, "RiceLeafsDisease", "train", "healthy"),
    ],
}


CLASS_NAMES = list(dataset_paths.keys())
IMG_SIZE = 224 # Image height/width used for model input
BATCH_SIZE = 16 # Number of samples trained per update
EPOCHS = 2 # Number of times to train on the full dataset
SEED = 42 # Makes results reproducible by controlling randomness
np.random.seed(SEED)
tf.random.set_seed(SEED)
AUTO = tf.data.AUTOTUNE

# GAN augmentation options
USE_GAN_AUG = False   # Set True to enable GAN-based augmentation (heavy)
GAN_EPOCHS = 1      # Keep low for experimentation; increase for better quality
NUM_SYNTHETIC = 200   # number of synthetic images to generate if GAN used

# Output artifacts
MODEL_SAVE = "paddy_disease_detector_resnet50_full.keras"
TFLITE_SAVE = "paddy_disease_detector_resnet50_full.tflite"
CHECKPOINT_PATH = "best_disease_detector_full.h5"
LABELS_FILE = "disease_labels.txt"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
plot_path = os.path.join(BASE_DIR, "cnn_training_curve.png")
cm_path = os.path.join(BASE_DIR, "confusion_matrix.png")
final_pdf_path = os.path.join(BASE_DIR, "model_results_summary.pdf")
os.makedirs(os.path.join(BASE_DIR, "plots"), exist_ok=True)

# ------------------------
# Helpers: load images
# ------------------------
def load_images_from_dirs(dirs, resize=(IMG_SIZE, IMG_SIZE)):
    images = []
    for d in dirs:
        if not os.path.exists(d):
            # skip missing directories; user might not have all sources
            continue
        for root, _, files in os.walk(d):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    fp = os.path.join(root, file)
                    img = cv2.imread(fp)
                    if img is None:
                        continue
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, resize)
                    images.append(img)
    return images

# Load dataset
all_images = []
all_labels = []
for idx, cls in enumerate(CLASS_NAMES):
    imgs = load_images_from_dirs(dataset_paths[cls])
    all_images.extend(imgs)
    all_labels.extend([idx]*len(imgs))

all_images = np.array(all_images, dtype=np.float32)
all_labels = np.array(all_labels)
print(f"Loaded images: {len(all_images)} across {len(CLASS_NAMES)} classes")

if len(all_images) == 0:
    raise SystemExit("No images found - please update dataset_paths to valid folders.")

# Normalize to [0,1]
all_images = all_images / 255.0

# Shuffle
perm = np.random.permutation(len(all_images))
all_images = all_images[perm]
all_labels = all_labels[perm]

# Train/Val/Test split 70/20/10
n = len(all_images)
train_end = int(n * 0.7)
val_end = int(n * 0.9)

train_images = all_images[:train_end]
train_labels = all_labels[:train_end]
val_images = all_images[train_end:val_end]
val_labels = all_labels[train_end:val_end]
test_images = all_images[val_end:]
test_labels = all_labels[val_end:]

print(f"Train: {len(train_images)}, Val: {len(val_images)}, Test: {len(test_images)}")

# ------------------------
# ROI helper: simple leaf crop (optional)
# ------------------------
def simple_roi_crop(rgb_image):
    """
    Attempts simple leaf extraction via HSV thresholding and largest contour.
    Returns cropped RGB resized image or original if detection fails.
    """
    img = (rgb_image*255).astype(np.uint8) if rgb_image.max() <= 1.0 else rgb_image.astype(np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # broad green-ish mask
    lower = np.array([10, 20, 20])
    upper = np.array([120, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    # morphological ops
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    c = max(contours, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(c)
    cropped = img[y:y+h, x:x+w]
    if cropped.size == 0:
        return cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return cv2.resize(cropped, (IMG_SIZE, IMG_SIZE))

# Example use: train_images = np.array([simple_roi_crop(im) for im in train_images])

# ------------------------
# Augmentation pipelines
# ------------------------
def basic_augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.15)
    image = tf.image.random_contrast(image, lower=0.85, upper=1.15)
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    # small rotations
    angle = tf.random.uniform([], -0.1, 0.1)
    image = tfa.image.rotate(image, angle) if 'tfa' in globals() else image
    return image, label

# Try to import tensorflow_addons for rotate; fallback if missing
try:
    import tensorflow_addons as tfa
except Exception:
    tfa = None
    print("tensorflow_addons not found — small rotation augmentation will be skipped. Install 'tensorflow_addons' for rotate augmentation.")

# Create tf.data datasets
train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
train_ds = train_ds.shuffle(2048, seed=SEED)

# Apply augmentation map if tfa available else limited
def tf_augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, 0.15)
    image = tf.image.random_contrast(image, 0.85, 1.15)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label

train_ds = train_ds.map(tf_augment, num_parallel_calls=AUTO).batch(BATCH_SIZE).prefetch(AUTO)
val_ds = tf.data.Dataset.from_tensor_slices((val_images, val_labels)).batch(BATCH_SIZE).prefetch(AUTO)
test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(BATCH_SIZE)

# ------------------------
# Optional: Simple DCGAN for augmentation
# ------------------------
if USE_GAN_AUG:
    print("=== GAN augmentation ENABLED ===")
    # We'll build a compact DCGAN for RGB 64x64 images, train from train_images downscaled
    class DCGAN:
        def __init__(self, img_shape=(64,64,3), latent_dim=100):
            self.img_shape = img_shape
            self.latent_dim = latent_dim
            self.generator = self.build_generator()
            self.discriminator = self.build_discriminator()
            self.discriminator.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
            self.discriminator.trainable = False
            z = Input(shape=(latent_dim,))
            img = self.generator(z)
            validity = self.discriminator(img)
            self.combined = Model(z, validity)
            self.combined.compile(optimizer=Adam(1e-4), loss='binary_crossentropy')

        def build_generator(self):
            z = Input(shape=(self.latent_dim,))
            x = layers.Dense(8*8*128, activation='relu')(z)
            x = layers.Reshape((8,8,128))(x)
            x = layers.UpSampling2D()(x)
            x = layers.Conv2D(128, kernel_size=3, padding='same', activation='relu')(x)
            x = layers.UpSampling2D()(x)
            x = layers.Conv2D(64, kernel_size=3, padding='same', activation='relu')(x)
            x = layers.UpSampling2D()(x)
            x = layers.Conv2D(3, kernel_size=3, padding='same', activation='tanh')(x)
            model = Model(z, x)
            return model

        def build_discriminator(self):
            img = Input(shape=self.img_shape)
            x = layers.Conv2D(32, kernel_size=3, strides=2, padding='same')(img)
            x = layers.LeakyReLU(0.2)(x)
            x = layers.Conv2D(64, kernel_size=3, strides=2, padding='same')(x)
            x = layers.LeakyReLU(0.2)(x)
            x = layers.Conv2D(128, kernel_size=3, strides=2, padding='same')(x)
            x = layers.LeakyReLU(0.2)(x)
            x = layers.Flatten()(x)
            x = layers.Dense(1, activation='sigmoid')(x)
            model = Model(img, x)
            return model

        def train(self, images, epochs=1, batch_size=32):
            # images scaled to [-1,1] and shape 64x64x3
            imgs = np.array([cv2.resize((im*255).astype(np.uint8), (64,64)) for im in images])
            imgs = imgs.astype(np.float32) / 127.5 - 1.0
            valid = np.ones((batch_size,1))
            fake = np.zeros((batch_size,1))
            steps = len(imgs) // batch_size
            for epoch in range(epochs):
                for _ in range(steps):
                    idx = np.random.randint(0, len(imgs), batch_size)
                    real = imgs[idx]
                    noise = np.random.normal(0,1,(batch_size, self.latent_dim))
                    gen_imgs = self.generator.predict(noise, verbose=0)
                    d_loss_real = self.discriminator.train_on_batch(real, valid)
                    d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
                    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                    noise = np.random.normal(0,1,(batch_size, self.latent_dim))
                    g_loss = self.combined.train_on_batch(noise, valid)
                if epoch % 10 == 0:
                    print(f"GAN epoch {epoch}/{epochs} d_loss={d_loss[0]:.4f} g_loss={g_loss:.4f}")

        def generate(self, n):
            noise = np.random.normal(0,1,(n, self.latent_dim))
            gen = self.generator.predict(noise)
            gen = (gen + 1.0) / 2.0  # back to [0,1]
            out = [cv2.resize((g*255).astype(np.uint8), (IMG_SIZE, IMG_SIZE)) for g in gen]
            return np.array(out) / 255.0

    # Train quick GAN (use small epochs for testing)
    gan = DCGAN(img_shape=(64,64,3), latent_dim=100)
    sample_for_gan = train_images[np.random.choice(len(train_images), min(1000, len(train_images)), replace=False)]
    gan.train(sample_for_gan, epochs=GAN_EPOCHS, batch_size=32)
    synth = gan.generate(NUM_SYNTHETIC)
    synth_labels = np.random.choice(train_labels, size=len(synth))  # assign random labels -> better: class-conditional GAN
    # Append synthetic images to training pool (simple approach)
    train_images = np.concatenate([train_images, synth], axis=0)
    train_labels = np.concatenate([train_labels, synth_labels], axis=0)
    # Rebuild tf dataset
    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    train_ds = train_ds.shuffle(2048, seed=SEED).map(tf_augment, num_parallel_calls=AUTO).batch(BATCH_SIZE).prefetch(AUTO)
    print("GAN augmentation added synthetic images to the training set.")

# ------------------------
# CBAM (Convolutional Block Attention Module)
# ------------------------
def cbam_block(input_feature, ratio=8):
    channel = input_feature.shape[-1]
    shared_layer_one = layers.Dense(channel//ratio, activation='relu', kernel_initializer='he_normal')
    shared_layer_two = layers.Dense(channel, kernel_initializer='he_normal')

    # Channel attention
    avg_pool = layers.GlobalAveragePooling2D()(input_feature)
    avg_pool = layers.Reshape((1,1,channel))(avg_pool)
    avg_fc = shared_layer_one(avg_pool)
    avg_fc = shared_layer_two(avg_fc)

    max_pool = layers.GlobalMaxPooling2D()(input_feature)
    max_pool = layers.Reshape((1,1,channel))(max_pool)
    max_fc = shared_layer_one(max_pool)
    max_fc = shared_layer_two(max_fc)

    cbam_feature = layers.Add()([avg_fc, max_fc])
    cbam_feature = layers.Activation('sigmoid')(cbam_feature)
    channel_refined = layers.Multiply()([input_feature, cbam_feature])

    # Spatial attention
    avg_pool_sp = layers.Lambda(lambda x: tf.reduce_mean(x, axis=3, keepdims=True))(channel_refined)
    max_pool_sp = layers.Lambda(lambda x: tf.reduce_max(x, axis=3, keepdims=True))(channel_refined)
    concat = layers.Concatenate(axis=3)([avg_pool_sp, max_pool_sp])
    cbam_spatial = layers.Conv2D(1, kernel_size=7, strides=1, padding='same', activation='sigmoid', kernel_initializer='he_normal')(concat)
    refined_feature = layers.Multiply()([channel_refined, cbam_spatial])
    return refined_feature

# ------------------------
# Multi-scale (Inception-like) block
# ------------------------
def multiscale_block(x, filters=64):
    branch1 = layers.Conv2D(filters, (1,1), padding='same', activation='relu')(x)
    branch3 = layers.Conv2D(filters, (1,1), padding='same', activation='relu')(x)
    branch3 = layers.Conv2D(filters, (3,3), padding='same', activation='relu')(branch3)
    branch5 = layers.Conv2D(filters, (1,1), padding='same', activation='relu')(x)
    branch5 = layers.Conv2D(filters, (5,5), padding='same', activation='relu')(branch5)
    branch_pool = layers.MaxPooling2D((3,3), strides=1, padding='same')(x)
    branch_pool = layers.Conv2D(filters, (1,1), padding='same', activation='relu')(branch_pool)
    out = layers.Concatenate(axis=-1)([branch1, branch3, branch5, branch_pool])
    out = layers.BatchNormalization()(out)
    out = layers.Activation('relu')(out)
    return out

# ------------------------
# Build model with ResNet50 + CBAM + multiscale
# ------------------------
def build_model(num_classes=len(CLASS_NAMES), img_size=IMG_SIZE):
    base = ResNet50(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
    base.trainable = False
    inp = Input(shape=(img_size, img_size, 3))
    x = base(inp, training=False)  # backbone features
    # add multi-scale block
    x = multiscale_block(x, filters=64)
    # attention
    x = cbam_block(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inp, outputs=out)
    return model

model = build_model()
# model.compile(optimizer=Adam(learning_rate=1e-4),
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# model.compile(
#     optimizer=Adam(1e-5),
#     loss='sparse_categorical_crossentropy',
#     metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
# )

from tensorflow.keras.metrics import SparseCategoricalAccuracy

model.compile(
    optimizer=Adam(1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=[SparseCategoricalAccuracy()]
)


# import tensorflow_addons as tfa
# from tensorflow.keras.optimizers import Adam
#
# model.compile(
#     optimizer=Adam(learning_rate=1e-4),
#     loss='sparse_categorical_crossentropy',
#     metrics=[
#         tf.keras.metrics.SparseCategoricalAccuracy()
#     ]
# )

model.summary()

# ------------------------
# Callbacks & Training
# ------------------------
early_stop = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1)
# checkpoint = ModelCheckpoint(CHECKPOINT_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)

from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint(
    'best_model.h5',
    monitor='val_sparse_categorical_accuracy',
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)

history = model.fit(train_ds,
                    epochs=EPOCHS,
                    validation_data=val_ds,
                    callbacks=[early_stop, checkpoint])

# Save training curves
import matplotlib.pyplot as plt

plt.figure(figsize=(6,4))
plt.plot(history.history['sparse_categorical_accuracy'], label='Train Accuracy')
plt.plot(history.history['val_sparse_categorical_accuracy'], label='Validation Accuracy')
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training Curve')
plt.xlabel('Epochs')
plt.ylabel('Accuracy / Loss')
plt.legend()
plt.tight_layout()
plt.savefig("cnn_training_curve.png")  # <- Make sure the filename matches plot_path
plt.close()

from tensorflow.keras.metrics import SparseCategoricalAccuracy

# Unfreeze last conv layers for fine-tuning
base_model = model.layers[1] if isinstance(model.layers[1], tf.keras.Model) else None
if base_model:
    for layer in base_model.layers[-30:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True

    model.compile(
        optimizer=Adam(1e-5),
        loss='sparse_categorical_crossentropy',
        metrics=[SparseCategoricalAccuracy()]
    )

    print("Starting fine-tuning...")

    history_fine = model.fit(
        train_ds,
        epochs=2,
        validation_data=val_ds,
        callbacks=[early_stop, checkpoint]
    )



# ------------------------
# Evaluation: predictions and metrics
# ------------------------

from sklearn.metrics import classification_report
# Get predictions on test set
y_pred_probs = model.predict(test_ds)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = test_labels

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# print("Classification Report:")
# print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))
#
# # Confusion matrix
# cm = confusion_matrix(y_true, y_pred)
# print("Confusion Matrix:\n", cm)
#
# AUC-ROC (one-vs-rest)
y_true_b = label_binarize(y_true, classes=list(range(len(CLASS_NAMES))))
try:
    aucs = {}
    for i, cname in enumerate(CLASS_NAMES):
        try:
            auc = roc_auc_score(y_true_b[:, i], y_pred_probs[:, i])
        except Exception:
            auc = float('nan')
        aucs[cname] = auc
    print("AUC per class:", aucs)
except Exception as e:
    print("AUC calculation failed:", e)

# # ==== 6. Confusion Matrix ====
# cm = confusion_matrix(y_true, y_pred)
# plt.figure(figsize=(5,4))
# plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
# plt.title("Confusion Matrix")
# plt.colorbar()
# tick_marks = np.arange(len(CLASS_NAMES))
# plt.xticks(tick_marks, CLASS_NAMES, rotation=45)
# plt.yticks(tick_marks, CLASS_NAMES)
# plt.ylabel('True label')
# plt.xlabel('Predicted label')
# for i in range(cm.shape[0]):
#     for j in range(cm.shape[1]):
#         plt.text(j, i, cm[i, j], ha="center", va="center", color="red")
# plt.tight_layout()
# plt.savefig(cm_path)
# plt.close()
# print(f"✅ Confusion matrix saved to: {cm_path}")
#
# # ==== 7. AUC per class ====
# y_true_b = label_binarize(y_true, classes=list(range(len(CLASS_NAMES))))
# aucs = {}
# for i, cname in enumerate(CLASS_NAMES):
#     try:
#         auc = roc_auc_score(y_true_b[:, i], y_pred_probs[:, i])
#     except Exception:
#         auc = float('nan')
#     aucs[cname] = auc
#
# # ==== 8. Classification Report ====
# report_text = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)
#
# # ==== 9. PDF Helper Function ====
# def add_text(c, text_lines, x, y, line_height=12, font="Courier", font_size=10):
#     text = c.beginText(x, y)
#     text.setFont(font, font_size)
#     for line in text_lines:
#         text.textLine(line)
#     c.drawText(text)
#     return y - len(text_lines)*line_height - 10
#
# # ==== 10. Generate PDF ====
# c = canvas.Canvas(final_pdf_path, pagesize=letter)
# width, height = letter
# current_y = height - 100
#
# # Title
# c.setFont("Helvetica-Bold", 16)
# c.drawString(50, current_y, "Rice Leaf Disease Detection - Model Results Summary")
# current_y -= 30
#
# # Training Curve
# if os.path.exists(plot_path):
#     c.setFont("Helvetica-Bold", 12)
#     c.drawString(50, current_y, "Training Curve:")
#     current_y -= 10
#     img = ImageReader(plot_path)
#     c.drawImage(img, 50, current_y-300, width=500, height=300, preserveAspectRatio=True, mask='auto')
#     current_y -= 310
# else:
#     c.setFont("Helvetica-Oblique", 10)
#     c.drawString(50, current_y, "Training curve image not found!")
#     current_y -= 20
# import matplotlib.pyplot as plt
# import os
#
# # Ensure you have a valid directory for saving the plots
# plot_dir = os.path.join(BASE_DIR, "plots")
# os.makedirs(plot_dir, exist_ok=True)
#
# # Plot Accuracy
# plt.figure(figsize=(6, 4))
# plt.plot(history.history['sparse_categorical_accuracy'], label='Train Accuracy')
# plt.plot(history.history['val_sparse_categorical_accuracy'], label='Validation Accuracy')
# plt.title('Model Accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.tight_layout()
# accuracy_path = os.path.join(plot_dir, "accuracy_curve.png")
# plt.savefig(accuracy_path)  # Save accuracy plot
# plt.close()
#
# # Plot Loss
# plt.figure(figsize=(6, 4))
# plt.plot(history.history['loss'], label='Train Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.title('Model Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.tight_layout()
# loss_path = os.path.join(plot_dir, "loss_curve.png")
# plt.savefig(loss_path)  # Save loss plot
# plt.close()
#
# print(f"Accuracy and Loss plots saved to {plot_dir}")
#
#
# # Confusion Matrix
# if os.path.exists(cm_path):
#     c.setFont("Helvetica-Bold", 12)
#     c.drawString(50, current_y, "Confusion Matrix:")
#     current_y -= 10
#     img = ImageReader(cm_path)
#     c.drawImage(img, 50, current_y-250, width=400, height=250, preserveAspectRatio=True, mask='auto')
#     current_y -= 260
# else:
#     c.setFont("Helvetica-Oblique", 10)
#     c.drawString(50, current_y, "Confusion matrix image not found!")
#     current_y -= 20
#
# # Classification Report
# c.setFont("Helvetica-Bold", 12)
# c.drawString(50, current_y, "Classification Report:")
# current_y -= 15
# lines = report_text.split("\n")
# current_y = add_text(c, lines, 50, current_y, line_height=10, font="Courier", font_size=9)
#
# # AUC per class
# c.setFont("Helvetica-Bold", 12)
# c.drawString(50, current_y, "AUC per Class:")
# current_y -= 15
# auc_lines = [f"{k}: {v:.4f}" for k, v in aucs.items()]
# add_text(c, auc_lines, 50, current_y, line_height=12, font="Courier", font_size=10)
#
# # # AUC per class
# # c.setFont("Helvetica", 12)
# # c.drawString(50, height - 900, "AUC per class:")
# # text = c.beginText(50, height - 920)
# # text.setFont("Courier", 10)
# # for k, v in aucs.items():
# #     text.textLine(f"{k}: {v:.4f}")
# # c.drawText(text)
#
# c.save()
# print(f"✅ Professional PDF saved to: {final_pdf_path}")


# ==== 6. Confusion Matrix ====
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(5, 4))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(len(CLASS_NAMES))
plt.xticks(tick_marks, CLASS_NAMES, rotation=45)
plt.yticks(tick_marks, CLASS_NAMES)
plt.ylabel('True label')
plt.xlabel('Predicted label')
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center", color="red")
plt.tight_layout()
plt.savefig(cm_path)
plt.close()
print(f"✅ Confusion matrix saved to: {cm_path}")


# ==== 7. AUC per class ====
y_true_b = label_binarize(y_true, classes=list(range(len(CLASS_NAMES))))
aucs = {}
for i, cname in enumerate(CLASS_NAMES):
    try:
        auc = roc_auc_score(y_true_b[:, i], y_pred_probs[:, i])
    except Exception:
        auc = float('nan')
    aucs[cname] = auc


# ==== 8. Classification Report ====
report_text = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)


# ==== 9. PDF Helper Function ====
def add_text(c, text_lines, x, y, line_height=12, font="Courier", font_size=10):
    text = c.beginText(x, y)
    text.setFont(font, font_size)
    for line in text_lines:
        text.textLine(line)
    c.drawText(text)
    return y - len(text_lines) * line_height - 10


# ==== 10. Generate PDF ====
c = canvas.Canvas(final_pdf_path, pagesize=letter)
width, height = letter
current_y = height - 100

# Title
c.setFont("Helvetica-Bold", 16)
c.drawString(50, current_y, "Rice Leaf Disease Detection - Model Results Summary")
current_y -= 30

# # Training Curve Image
# if os.path.exists(plot_path):
#     c.setFont("Helvetica-Bold", 12)
#     c.drawString(50, current_y, "Training Curve:")
#     current_y -= 10
#     img = ImageReader(plot_path)
#     c.drawImage(img, 50, current_y - 300, width=500, height=300, preserveAspectRatio=True, mask='auto')
#     current_y -= 310
# else:
#     c.setFont("Helvetica-Oblique", 10)
#     c.drawString(50, current_y, "Training curve image not found!")
#     current_y -= 20


# Ensure you have a valid directory for saving the plots
plot_dir = os.path.join(BASE_DIR, "plots")
os.makedirs(plot_dir, exist_ok=True)



# Plot Accuracy
plt.figure(figsize=(6, 4))
plt.plot(history.history['sparse_categorical_accuracy'], label='Train Accuracy')
plt.plot(history.history['val_sparse_categorical_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
accuracy_path = os.path.join(plot_dir, "accuracy_curve.png")
plt.savefig(accuracy_path)  # Save accuracy plot
plt.close()

# Plot Loss
plt.figure(figsize=(6, 4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
loss_path = os.path.join(plot_dir, "loss_curve.png")
plt.savefig(loss_path)  # Save loss plot
plt.close()

print(f"Accuracy and Loss plots saved to {plot_dir}")


# Confusion Matrix
if os.path.exists(cm_path):
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, current_y, "Confusion Matrix:")
    current_y -= 10
    img = ImageReader(cm_path)
    c.drawImage(img, 50, current_y - 250, width=400, height=250, preserveAspectRatio=True, mask='auto')
    current_y -= 260
else:
    c.setFont("Helvetica-Oblique", 10)
    c.drawString(50, current_y, "Confusion matrix image not found!")
    current_y -= 20

# Classification Report
c.setFont("Helvetica-Bold", 12)
c.drawString(50, current_y, "Classification Report:")
current_y -= 15
lines = report_text.split("\n")
current_y = add_text(c, lines, 50, current_y, line_height=10, font="Courier", font_size=9)

# AUC per class
c.setFont("Helvetica-Bold", 12)
c.drawString(50, current_y, "AUC per Class:")
current_y -= 15
auc_lines = [f"{k}: {v:.4f}" for k, v in aucs.items()]
add_text(c, auc_lines, 50, current_y, line_height=12, font="Courier", font_size=10)

# Save the PDF
c.save()
print(f"✅ Professional PDF saved to: {final_pdf_path}")

# ------------------------
# Save model and labels
# ------------------------
model.save(MODEL_SAVE)
with open(LABELS_FILE, 'w') as f:
    for label in CLASS_NAMES:
        f.write(label + '\n')
print(f"Saved Keras model -> {MODEL_SAVE} and labels -> {LABELS_FILE}")

# ------------------------
# Grad-CAM function and example usage
# ------------------------
import tensorflow.keras.backend as K

def get_gradcam_heatmap(model, image, last_conv_layer_name=None, pred_index=None):
    """
    image: numpy array shape (H,W,3) normalized [0,1]
    last_conv_layer_name: if None try to find last conv
    """
    img_array = np.expand_dims(image, axis=0)
    if last_conv_layer_name is None:
        # heuristics: find last conv layer
        for layer in reversed(model.layers):
            if isinstance(layer, layers.Conv2D):
                last_conv_layer_name = layer.name
                break
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()
    return heatmap, int(pred_index)

# Save Grad-CAM for first N test images
N = min(10, len(test_images))
os.makedirs("gradcam_examples", exist_ok=True)
for i in range(N):
    img = test_images[i]
    heatmap, pred_index = get_gradcam_heatmap(model, img)
    # overlay
    heatmap_resized = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    heatmap_rgb = plt.get_cmap('jet')(heatmap_resized)[:, :, :3]
    overlay = (0.4 * heatmap_rgb + 0.6 * img)
    overlay = np.clip(overlay, 0, 1)
    plt.imsave(f"gradcam_examples/img_{i}_pred_{CLASS_NAMES[pred_index]}.png", overlay)
print("Saved Grad-CAM overlays to folder: gradcam_examples")

# 8️⃣ Grad-CAM Visualization (Explainability)
# ============================================================
os.makedirs("gradcam_outputs", exist_ok=True)   # For the second part
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    # Ensure that the layer name exists
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        grads = tape.gradient(predictions[:, class_idx], conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # Pool gradients across spatial dimensions
        conv_outputs = conv_outputs[0]

        # Compute heatmap by averaging the gradients
        heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    # Normalize heatmap values to be between 0 and 1
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap


# Inspect layers to find the right one
for layer in model.layers:
    print(layer.name)

# Once you find the correct layer name (e.g., 'conv2d_5', 'conv5_block3_out', etc.), replace it in your code
last_conv_layer = "conv2d_5"  # Example, replace with correct layer

for imgs, labels in val_ds.take(1):
    for i in range(3):
        img_array = tf.expand_dims(imgs[i], axis=0)
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer)

        img = np.uint8(255 * imgs[i].numpy() / np.max(imgs[i]))  # Convert image to uint8
        heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # Resize heatmap to image size
        heatmap_resized = np.uint8(255 * heatmap_resized)  # Convert to uint8
        heatmap_resized = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)  # Apply colormap

        # Overlay heatmap on image
        overlay = cv2.addWeighted(img, 0.7, heatmap_resized, 0.3, 0)

        # Save the result to "gradcam_outputs" folder
        output_path = os.path.join("gradcam_outputs", f"gradcam_{i}.png")
        cv2.imwrite(output_path, overlay)

#print("Saved Grad-CAM overlays to folder: gradcam_outputs")
print("Grad-CAM overlays saved successfully.")


# ------------------------
# Convert to TFLite with quantization
# ------------------------
# Representative dataset generator
def representative_data_gen():
    for input_value, _ in tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(1).take(100):
        # input_value is shape (1, H, W, 3)
        yield [tf.cast(input_value, tf.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
try:
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    tflite_model = converter.convert()
    with open(TFLITE_SAVE, 'wb') as f:
        f.write(tflite_model)
    print(f"TFLite model saved: {TFLITE_SAVE}")
except Exception as e:
    print("TFLite conversion with full int8 failed (falling back to float32). Error:", e)
    # fallback
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(TFLITE_SAVE, 'wb') as f:
        f.write(tflite_model)
    print(f"Float32 TFLite model saved: {TFLITE_SAVE}")

    # Evaluate performance on validation data
    val_steps = val_ds.samples // BATCH_SIZE
    val_ds.reset()
    preds = model.predict(val_ds, steps=val_steps, verbose=1)
    pred_labels = np.argmax(preds, axis=1)
    true_labels = val_ds.classes[:val_steps * BATCH_SIZE]
    accuracy = np.mean(pred_labels == true_labels)

    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average=None, labels=range(4))

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    from matplotlib.backends.backend_pdf import PdfPages
    # Save classification report as a PDF file
    pdf_path = os.path.join(BASE_DIR, 'classification_report.pdf')
    with PdfPages(pdf_path) as pdf:
        fig, ax = plt.subplots(figsize=(8,6))
        ax.axis('off')
        txt = f"Validation Accuracy: {accuracy:.4f}\n\n"
        txt += "Class-wise Metrics:\n\n"
        for i, label in enumerate(CLASS_NAMES):
            txt += f"{label}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1-Score={f1[i]:.4f}\n"
        ax.text(0.01, 0.99, txt, verticalalignment='top', fontsize=12)
        pdf.savefig(fig)
        plt.close()
    print(f"Classification report saved as {pdf_path}")

print("All done.")
