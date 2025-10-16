#!/usr/bin/env python3
"""
Multi-resolution validation to determine optimal QGIS generation parameters
"""

import os
import torch
import numpy as np
from pathlib import Path
import json
import cv2
from tqdm import tqdm
import pandas as pd
from typing import Dict, List, Tuple

# Import your existing modules
import sys
sys.path.append('/Users/zif/clustermount/Master-Thesis')

import segmentation_models_pytorch as smp
from helper.preprocessing import get_preprocessing_fn, ENCODER
from helper.dataset_loader import SubstationDataset, get_validation_augmentation

class MultiResolutionValidator:
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        self.preprocessing_fn = get_preprocessing_fn()
        
        # Your training resolution
        self.training_resolution = 1200
        
    def _load_model(self, model_path):
        """Load your best model"""
        model = smp.Unet(
            encoder_name=ENCODER,
            encoder_weights='imagenet',
            in_channels=3,
            classes=1,
            decoder_attention_type="none"
        )
        
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint)
        model.to(self.device)
        model.eval()
        return model
    
    def analyze_resolution_impact(self, 
                                  images_3000: Dict[str, np.ndarray],
                                  images_1024: Dict[str, np.ndarray],
                                  osm_counts: Dict[str, int]) -> Dict:
        """
        Compare predictions at different resolutions against OSM counts
        """
        results = {
            '3000x3000': [],
            '1024x1024': [],
            'multi_scale': []
        }
        
        for substation_id in tqdm(images_3000.keys(), desc="Analyzing resolutions"):
            if substation_id not in images_1024 or substation_id not in osm_counts:
                continue
                
            # Process 3000x3000 image
            pred_3000 = self._process_high_resolution(images_3000[substation_id])
            count_3000 = self._count_transformers(pred_3000)
            
            # Process 1024x1024 image  
            pred_1024 = self._process_standard_resolution(images_1024[substation_id])
            count_1024 = self._count_transformers(pred_1024)
            
            # Multi-scale fusion
            pred_multi = self._multi_scale_fusion(images_3000[substation_id])
            count_multi = self._count_transformers(pred_multi)
            
            # OSM reference
            osm_count = osm_counts[substation_id]
            
            # Store results
            results['3000x3000'].append({
                'id': substation_id,
                'predicted': count_3000,
                'osm': osm_count,
                'error': abs(count_3000 - osm_count)
            })
            
            results['1024x1024'].append({
                'id': substation_id,
                'predicted': count_1024,
                'osm': osm_count,
                'error': abs(count_1024 - osm_count)
            })
            
            results['multi_scale'].append({
                'id': substation_id,
                'predicted': count_multi,
                'osm': osm_count,
                'error': abs(count_multi - osm_count)
            })
        
        return self._summarize_results(results)
    
    def _process_high_resolution(self, image_3000: np.ndarray) -> np.ndarray:
        """Process 3000x3000 image using sliding window"""
        # Sliding window approach for high resolution
        window_size = 1200
        stride = 800  # Overlap for better coverage
        
        h, w = image_3000.shape[:2]
        predictions = np.zeros((h, w), dtype=np.float32)
        counts = np.zeros((h, w), dtype=np.float32)
        
        for y in range(0, h - window_size + 1, stride):
            for x in range(0, w - window_size + 1, stride):
                # Extract patch
                patch = image_3000[y:y+window_size, x:x+window_size]
                
                # Predict
                pred = self._predict_single(patch)
                
                # Accumulate predictions
                predictions[y:y+window_size, x:x+window_size] += pred
                counts[y:y+window_size, x:x+window_size] += 1
        
        # Average overlapping predictions
        predictions = predictions / (counts + 1e-8)
        return (predictions > 0.5).astype(np.uint8)
    
    def _process_standard_resolution(self, image_1024: np.ndarray) -> np.ndarray:
        """Process 1024x1024 image with resizing"""
        # Resize to training resolution
        resized = cv2.resize(image_1024, (1200, 1200), interpolation=cv2.INTER_LINEAR)
        pred = self._predict_single(resized)
        
        # Resize back to original
        pred_resized = cv2.resize(pred, (1024, 1024), interpolation=cv2.INTER_LINEAR)
        return (pred_resized > 0.5).astype(np.uint8)
    
    def _multi_scale_fusion(self, image_3000: np.ndarray) -> np.ndarray:
        """Combine predictions at multiple scales"""
        scales = [1.0, 0.8, 1.2]  # Different zoom levels
        predictions = []
        
        for scale in scales:
            # Resize image
            new_size = int(1200 * scale)
            resized = cv2.resize(image_3000, (new_size, new_size))
            
            # Extract center crop of 1200x1200
            if new_size > 1200:
                start = (new_size - 1200) // 2
                crop = resized[start:start+1200, start:start+1200]
            else:
                # Pad if smaller
                pad = (1200 - new_size) // 2
                crop = cv2.copyMakeBorder(resized, pad, pad, pad, pad, 
                                         cv2.BORDER_REFLECT)
                crop = crop[:1200, :1200]
            
            pred = self._predict_single(crop)
            predictions.append(pred)
        
        # Average predictions
        avg_pred = np.mean(predictions, axis=0)
        return (avg_pred > 0.5).astype(np.uint8)
    
    def _predict_single(self, image: np.ndarray) -> np.ndarray:
        """Run inference on single image"""
        # Preprocess
        preprocessed = self.preprocessing_fn(image=image)['image']
        img_tensor = torch.from_numpy(preprocessed).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(img_tensor)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
        
        return probs
    
    def _count_transformers(self, mask: np.ndarray) -> int:
        """Count transformer instances with morphological cleanup"""
        from scipy import ndimage
        from skimage.morphology import remove_small_objects
        
        # Clean up small noise
        cleaned = remove_small_objects(mask.astype(bool), min_size=50)
        
        # Label connected components
        labeled, num_features = ndimage.label(cleaned)
        
        return num_features
    
    def _summarize_results(self, results: Dict) -> Dict:
        """Calculate summary statistics"""
        summary = {}
        
        for resolution, data in results.items():
            df = pd.DataFrame(data)
            
            summary[resolution] = {
                'mean_error': df['error'].mean(),
                'median_error': df['error'].median(),
                'correlation': df[['predicted', 'osm']].corr().iloc[0, 1],
                'within_2_count': (df['error'] <= 2).sum(),
                'total_samples': len(df)
            }
        
        return summary
    
    def recommend_qgis_parameters(self, analysis_results: Dict) -> Dict:
        """Generate QGIS parameter recommendations"""
        
        # Find best performing approach
        best_approach = min(analysis_results.items(), 
                           key=lambda x: x[1]['mean_error'])[0]
        
        if best_approach == '3000x3000':
            # High resolution performs best - use larger buffer
            recommendations = {
                'image_size': 1200,
                'buffer_meters': 300,  # Larger buffer
                'resolution_mpp': 0.5,
                'reasoning': 'High resolution analysis shows better results with context'
            }
        elif best_approach == '1024x1024':
            # Lower resolution sufficient - use tighter crop
            recommendations = {
                'image_size': 1200,
                'buffer_meters': 200,  # Smaller buffer
                'resolution_mpp': 0.5,
                'reasoning': 'Compact images perform well, minimize unnecessary context'
            }
        else:  # multi_scale
            # Multi-scale works best - use standard buffer
            recommendations = {
                'image_size': 1200,
                'buffer_meters': 250,  # Medium buffer
                'resolution_mpp': 0.5,
                'reasoning': 'Multi-scale benefits from balanced context'
            }
        
        # Add OSM footprint usage strategy
        recommendations['osm_footprint_strategy'] = {
            'use_footprint': True,
            'footprint_expansion': 1.5,  # Expand footprint by 50%
            'fallback_radius': recommendations['buffer_meters'],
            'reasoning': 'Use OSM footprint when available, expand slightly for safety'
        }
        
        return recommendations

def quick_experiment_notebook():
    """
    Quick experiment setup for Jupyter notebook
    """
    return experiment_code

# Main execution suggestion
if __name__ == "__main__":
    print("STREAMLINED APPROACH:")
    print("1. Run quick validation on 20-50 existing images")
    print("2. Compare 3000x3000 vs 1024x1024 performance")
    print("3. Use OSM counts as rough accuracy indicator")
    print("4. Generate QGIS parameters based on results")
    print("5. Apply to 5000 image generation")