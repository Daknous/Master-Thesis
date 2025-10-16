"""
Enhanced Ensemble Inference for Transformer Segmentation
Builds on your existing inference pipeline with ensemble capabilities
"""

import os, glob, time
import torch
import cv2
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
from pycocotools.coco import COCO
from PIL import Image
from skimage.morphology import remove_small_objects, remove_small_holes, opening, closing, disk
from skimage.measure import label
from itertools import combinations
import json

# --- Enhanced Configuration ---
IMG_SIZE = 1200
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENCODER = "resnet34"

# Scale configuration
PIXEL_TO_METER = 0.1317 * (3000 / 1200)  # ≈0.61734 m/px
mpp_x = mpp_y = PIXEL_TO_METER

# Paths
VAL_COCO_JSON = "/Users/zif/Documents/Substation_Master_thesis/Master-Thesis/Dataset_v2_filtered/test/_annotations.coco.json"
VAL_IMG_DIR = "/Users/zif/Documents/Substation_Master_thesis/Master-Thesis/Dataset_v2_filtered/test"

# Inference configuration
THRESHOLD = 0.5  # Will be optimized per model
MIN_SIZE = 100

# Transformer capacity categories based on area (square meters)
CAPACITY_CATEGORIES = {
    'Small (≤50 MVA)': (0, 100),      # 0-100 m²
    'Medium (50-150 MVA)': (100, 300), # 100-300 m²
    'Large (150-300 MVA)': (300, 600), # 300-600 m²
    'Extra Large (>300 MVA)': (600, float('inf'))  # >600 m²
}

class EnsembleInference:
    """Enhanced inference class with ensemble capabilities"""
    
    def __init__(self, model_paths, encoder='resnet34', device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.encoder = encoder
        self.models = {}
        self.preprocess_fn = smp.encoders.get_preprocessing_fn(encoder, 'imagenet')
        
        print(f"🖥️  Using device: {self.device}")
        self.load_models(model_paths)
    
    def load_models(self, model_paths):
        """Load multiple models for ensemble"""
        print(f"\n🔄 Loading {len(model_paths)} models for ensemble...")
        
        for i, model_info in enumerate(model_paths):
            if isinstance(model_info, dict):
                path = model_info['path']
                name = model_info.get('name', f'model_{i}')
                weight = model_info.get('weight', 1.0)
            else:
                path = model_info
                name = f'model_{i}'
                weight = 1.0
            
            try:
                model = smp.Unet(
                    encoder_name=self.encoder,
                    encoder_weights=None,  # Don't load ImageNet weights
                    in_channels=3,
                    classes=1
                )
                
                checkpoint = torch.load(path, map_location=self.device)
                model.load_state_dict(checkpoint)
                model.to(self.device).eval()
                
                self.models[name] = {
                    'model': model,
                    'weight': weight,
                    'path': path
                }
                
                print(f"✅ Loaded {name}: {os.path.basename(path)}")
                
            except Exception as e:
                print(f"❌ Failed to load {path}: {e}")
    
    def run_single_inference(self, image_pil, model, threshold=0.5):
        """Run inference with a single model"""
        # Convert PIL to numpy
        img_np = np.array(image_pil)
        
        # Apply preprocessing
        img_preprocessed = self.preprocess_fn(img_np)
        
        # Convert to tensor
        inp = torch.from_numpy(img_preprocessed.astype('float32')).permute(2, 0, 1).unsqueeze(0)
        inp = inp.to(self.device)
        
        with torch.no_grad():
            logits = model(inp)
            probs = torch.sigmoid(logits).squeeze(0).squeeze(0).cpu().numpy()
        
        return probs
    
    def run_ensemble_inference(self, image_pil, method='weighted_average', threshold=0.5):
        """Run ensemble inference with multiple models"""
        if len(self.models) == 1:
            # Single model case
            model_name = list(self.models.keys())[0]
            model = self.models[model_name]['model']
            probs = self.run_single_inference(image_pil, model, threshold)
            return (probs > threshold).astype(np.uint8)
        
        # Get predictions from all models
        all_predictions = []
        weights = []
        
        for model_name, model_data in self.models.items():
            model = model_data['model']
            weight = model_data['weight']
            
            probs = self.run_single_inference(image_pil, model, threshold)
            all_predictions.append(probs)
            weights.append(weight)
        
        # Ensemble combination
        if method == 'simple_average':
            ensemble_probs = np.mean(all_predictions, axis=0)
        elif method == 'weighted_average':
            weights = np.array(weights)
            weights = weights / weights.sum()  # Normalize
            ensemble_probs = np.average(all_predictions, axis=0, weights=weights)
        elif method == 'majority_voting':
            # Convert to binary first, then vote
            binary_preds = [(pred > threshold).astype(int) for pred in all_predictions]
            ensemble_probs = np.mean(binary_preds, axis=0)
            threshold = 0.5  # For majority vote
        else:
            raise ValueError(f"Unknown ensemble method: {method}")
        
        return (ensemble_probs > threshold).astype(np.uint8)
    
    def analyze_transformer_areas(self, mask, image_name):
        """Analyze individual transformer areas and categorize by capacity"""
        # Find connected components
        labeled_mask = label(mask, connectivity=2)
        num_transformers = labeled_mask.max()
        
        transformer_data = []
        
        for transformer_id in range(1, num_transformers + 1):
            # Extract single transformer
            single_transformer = (labeled_mask == transformer_id)
            
            # Calculate area metrics
            pixel_area = single_transformer.sum()
            real_area_m2 = pixel_area * mpp_x * mpp_y
            
            # Categorize by capacity
            capacity_category = self.categorize_transformer(real_area_m2)
            
            # Calculate centroid and bounding box
            y_coords, x_coords = np.where(single_transformer)
            centroid_x = x_coords.mean()
            centroid_y = y_coords.mean()
            
            min_y, max_y = y_coords.min(), y_coords.max()
            min_x, max_x = x_coords.min(), x_coords.max()
            bbox_width = max_x - min_x + 1
            bbox_height = max_y - min_y + 1
            
            transformer_data.append({
                'image': image_name,
                'transformer_id': transformer_id,
                'pixel_area': pixel_area,
                'area_m2': real_area_m2,
                'capacity_category': capacity_category,
                'centroid_x': centroid_x,
                'centroid_y': centroid_y,
                'bbox_width': bbox_width,
                'bbox_height': bbox_height,
                'aspect_ratio': bbox_width / bbox_height if bbox_height > 0 else 0
            })
        
        return transformer_data
    
    def categorize_transformer(self, area_m2):
        """Categorize transformer by capacity based on area"""
        for category, (min_area, max_area) in CAPACITY_CATEGORIES.items():
            if min_area <= area_m2 < max_area:
                return category
        return 'Unknown'

# Enhanced post-processors
def get_post_processors():
    """Define various post-processing methods"""
    return {
        "No PP": lambda m, *args: m,
        "Remove Small": lambda m, *args: remove_small_objects(
            m.astype(bool), min_size=MIN_SIZE
        ).astype(np.uint8),
        "Remove Small + Holes": lambda m, *args: remove_small_holes(
            remove_small_objects(m.astype(bool), min_size=MIN_SIZE),
            area_threshold=MIN_SIZE//2
        ).astype(np.uint8),
        "Morphological": lambda m, *args: closing(
            opening(m.astype(bool), disk(2)), disk(3)
        ).astype(np.uint8),
        "Combined": lambda m, *args: remove_small_holes(
            remove_small_objects(
                closing(opening(m.astype(bool), disk(2)), disk(3)),
                min_size=MIN_SIZE
            ),
            area_threshold=MIN_SIZE//2
        ).astype(np.uint8)
    }

def run_comprehensive_evaluation(model_configs, ensemble_methods=['single', 'weighted_average']):
    """Run comprehensive evaluation with multiple models and ensemble methods"""
    
    # Initialize ensemble inference
    ensemble_inference = EnsembleInference(model_configs, encoder=ENCODER, device=DEVICE)
    
    # Load COCO and image infos
    coco = COCO(VAL_COCO_JSON)
    img_infos = coco.loadImgs(coco.getImgIds())
    
    # Get post-processors
    post_processors = get_post_processors()
    
    # Storage for results
    records = []
    all_transformer_data = []
    
    print(f"\n📊 Evaluating on {len(img_infos)} images...")
    print(f"🔀 Testing ensemble methods: {ensemble_methods}")
    print(f"🛠️  Post-processors: {list(post_processors.keys())}")
    
    for i, info in enumerate(img_infos):
        if (i + 1) % 10 == 0:
            print(f"Processing {i+1}/{len(img_infos)} images...")
        
        coco_id = info["id"]
        base = os.path.splitext(info["file_name"])[0]
        
        # Find actual image file
        candidates = glob.glob(os.path.join(VAL_IMG_DIR, f"{base}*"))
        if not candidates:
            print(f"⚠️  No file for {base!r}, skipping.")
            continue
        
        img_path = candidates[0]
        img_name = os.path.basename(img_path)
        
        # Load image
        img = Image.open(img_path).convert("RGB")
        
        # Ground truth annotations
        ann_ids = coco.getAnnIds(imgIds=[coco_id], iscrowd=False)
        gt_count = len(ann_ids)
        
        # Ground truth area (only mask-based annotations)
        gt_area_px = 0
        valid_area_anns = 0
        
        for ann in coco.loadAnns(ann_ids):
            if ann.get("segmentation"):
                try:
                    mask_ann = coco.annToMask(ann)
                    gt_area_px += (mask_ann > 0).sum()
                    valid_area_anns += 1
                except Exception as e:
                    print(f"⚠️  ann {ann['id']} mask decode failed: {e}")
        
        gt_area_m2 = gt_area_px * mpp_x * mpp_y if valid_area_anns > 0 else np.nan
        
        # Test different ensemble methods
        for ensemble_method in ensemble_methods:
            if ensemble_method == 'single' and len(ensemble_inference.models) > 1:
                # Test each individual model
                for model_name, model_data in ensemble_inference.models.items():
                    single_model_inference = EnsembleInference(
                        [{'path': model_data['path'], 'name': model_name}],
                        encoder=ENCODER, device=DEVICE
                    )
                    
                    t0 = time.time()
                    mask_raw = single_model_inference.run_ensemble_inference(img, method='simple_average')
                    inference_time = time.time() - t0
                    
                    # Test post-processors
                    for pp_name, processor in post_processors.items():
                        method_name = f"{model_name}_{pp_name}"
                        
                        t0 = time.time()
                        try:
                            mask_processed = processor(mask_raw, mpp_x, mpp_y)
                        except TypeError:
                            mask_processed = processor(mask_raw)
                        pp_time = time.time() - t0
                        
                        # Analyze results
                        results = analyze_prediction(
                            mask_processed, gt_count, gt_area_m2, 
                            img_name, method_name, inference_time + pp_time
                        )
                        records.append(results)
                        
                        # Analyze individual transformers
                        transformer_data = ensemble_inference.analyze_transformer_areas(
                            mask_processed, img_name
                        )
                        for td in transformer_data:
                            td['method'] = method_name
                        all_transformer_data.extend(transformer_data)
            
            else:
                # Ensemble methods
                t0 = time.time()
                mask_raw = ensemble_inference.run_ensemble_inference(img, method=ensemble_method)
                inference_time = time.time() - t0
                
                # Test post-processors
                for pp_name, processor in post_processors.items():
                    method_name = f"Ensemble_{ensemble_method}_{pp_name}"
                    
                    t0 = time.time()
                    try:
                        mask_processed = processor(mask_raw, mpp_x, mpp_y)
                    except TypeError:
                        mask_processed = processor(mask_raw)
                    pp_time = time.time() - t0
                    
                    # Analyze results
                    results = analyze_prediction(
                        mask_processed, gt_count, gt_area_m2,
                        img_name, method_name, inference_time + pp_time
                    )
                    records.append(results)
                    
                    # Analyze individual transformers
                    transformer_data = ensemble_inference.analyze_transformer_areas(
                        mask_processed, img_name
                    )
                    for td in transformer_data:
                        td['method'] = method_name
                    all_transformer_data.extend(transformer_data)
    
    return records, all_transformer_data

def analyze_prediction(mask, gt_count, gt_area_m2, img_name, method_name, elapsed_time):
    """Analyze prediction results"""
    # Count analysis
    labeled_mask = label(mask, connectivity=2)
    pred_count = labeled_mask.max()
    pred_area_px = (labeled_mask > 0).sum()
    pred_area_m2 = pred_area_px * mpp_x * mpp_y
    
    # Metrics
    count_correct = int(pred_count == gt_count)
    count_err = abs(pred_count - gt_count)
    
    if not np.isnan(gt_area_m2) and gt_area_m2 > 0:
        area_abs_err = abs(pred_area_m2 - gt_area_m2)
        area_rel_err = area_abs_err / gt_area_m2
    else:
        area_abs_err = np.nan
        area_rel_err = np.nan
    
    return {
        "image": img_name,
        "method": method_name,
        "gt_count": gt_count,
        "pred_count": pred_count,
        "count_correct": count_correct,
        "count_abs_err": count_err,
        "gt_area_m2": gt_area_m2,
        "pred_area_m2": pred_area_m2,
        "area_abs_err": area_abs_err,
        "area_rel_err": area_rel_err,
        "time_s": elapsed_time
    }

def create_comprehensive_report(records, transformer_data, output_dir="ensemble_inference_results"):
    """Create comprehensive analysis report"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Main results DataFrame
    df = pd.DataFrame(records)
    
    # Summary statistics
    summary = (
        df.groupby("method")
        .agg(
            count_accuracy=("count_correct", "mean"),
            mae_count=("count_abs_err", "mean"),
            mae_area_m2=("area_abs_err", "mean"),
            mape_area=("area_rel_err", "mean"),
            avg_time_s=("time_s", "mean"),
            total_predictions=("pred_count", "sum"),
            total_gt=("gt_count", "sum")
        )
        .reset_index()
    )
    
    # Sort by count accuracy and area error
    summary['combined_score'] = summary['count_accuracy'] - 0.1 * summary['mape_area'].fillna(1)
    summary = summary.sort_values('combined_score', ascending=False)
    
    # Save results
    df.to_csv(os.path.join(output_dir, 'detailed_results.csv'), index=False)
    summary.to_csv(os.path.join(output_dir, 'summary_results.csv'), index=False)
    
    # Transformer analysis
    if transformer_data:
        transformer_df = pd.DataFrame(transformer_data)
        transformer_df.to_csv(os.path.join(output_dir, 'transformer_analysis.csv'), index=False)
        
        # Capacity distribution analysis
        capacity_summary = transformer_df.groupby(['method', 'capacity_category']).agg(
            count=('transformer_id', 'count'),
            avg_area_m2=('area_m2', 'mean'),
            std_area_m2=('area_m2', 'std')
        ).reset_index()
        capacity_summary.to_csv(os.path.join(output_dir, 'capacity_analysis.csv'), index=False)
    
    # Print summary
    print("\n" + "="*80)
    print("🏆 ENSEMBLE INFERENCE RESULTS SUMMARY")
    print("="*80)
    print(f"{'Method':<40} {'Count Acc':<10} {'MAE Count':<10} {'MAPE Area':<12} {'Time (s)':<8}")
    print("-"*80)
    
    for _, row in summary.head(10).iterrows():
        print(f"{row['method']:<40} {row['count_accuracy']:<10.3f} "
              f"{row['mae_count']:<10.1f} {row['mape_area']:<12.3f} {row['avg_time_s']:<8.3f}")
    
    if transformer_data:
        print(f"\n📊 TRANSFORMER CAPACITY DISTRIBUTION:")
        capacity_dist = transformer_df['capacity_category'].value_counts()
        for category, count in capacity_dist.items():
            print(f"  {category}: {count} transformers")
    
    print(f"\n💾 Results saved to: {output_dir}")
    
    return summary, transformer_df if transformer_data else None

# Example usage configuration
def main():
    """Main execution function"""
    
    # Define your model configurations
    # Update these paths based on your models from the training log
    model_configs = [
        {
            'path': '/Users/zif/Documents/Substation_Master_thesis/Master-Thesis/runs/final_experiments/model_best_epoch46_valIoU0.7494.pth',
            'name': 'Best_Model_191',
            'weight': 0.2  # Highest weight for best model and vise versa
        },
        {
            'path': '/Users/zif/Documents/Substation_Master_thesis/Master-Thesis/runs/final_experiments/model_best_epoch77_valIoU0.8014.pth', 
            'name': 'Model_186',
            'weight': 0.3
        },
        {
            'path': '/Users/zif/Documents/Substation_Master_thesis/Master-Thesis/runs/final_experiments/model_best_epoch83_valIoU0.7325.pth',
            'name': 'Model_185', 
            'weight': 0.2
        },
        {
            'path': '/Users/zif/Documents/Substation_Master_thesis/Master-Thesis/runs/final_experiments/model_best_epoch92_valIoU0.7404.pth',
            'name': 'Model_187',
            'weight': 0.2
        },
        {
            'path': '/Users/zif/Documents/Substation_Master_thesis/Master-Thesis/runs/final_experiments/model_best_epoch95_valIoU0.8109.pth',
            'name': 'Model_188',
            'weight': 0.3
        },
        {
            'path': '/Users/zif/Documents/Substation_Master_thesis/Master-Thesis/runs/final_experiments/model_best_epoch191_valIoU0.8449.pth',
            'name': 'Model_190',
            'weight': 0.4
        }
    ]
    
    # Test different ensemble methods
    ensemble_methods = ['single', 'weighted_average', 'simple_average', 'majority_voting']
    
    # Run comprehensive evaluation
    print("🚀 Starting comprehensive ensemble evaluation...")
    records, transformer_data = run_comprehensive_evaluation(model_configs, ensemble_methods)
    
    # Generate report
    summary, transformer_df = create_comprehensive_report(records, transformer_data)
    
    print("✅ Evaluation complete!")

if __name__ == "__main__":
    main()