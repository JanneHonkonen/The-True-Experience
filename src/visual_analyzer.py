"""
Visual content analysis module.

Uses vision-language models (CLIP, BLIP) to analyze visual content,
detect objects, characters, environments, and cinematographic elements.
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional
import cv2
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration
import json


class VisualAnalyzer:
    """Handles visual content analysis using vision-language models."""
    
    def __init__(self, use_gpu: bool = False):
        """
        Initialize visual analyzer.
        
        Args:
            use_gpu: Whether to use GPU acceleration
        """
        self.use_gpu = use_gpu
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Initializing visual analyzer on device: {self.device}")
        
        # Initialize CLIP model
        self._init_clip()
        
        # Initialize BLIP model
        self._init_blip()
        
        # Initialize image preprocessing
        self._init_preprocessing()
    
    def _init_clip(self):
        """Initialize CLIP model for visual analysis."""
        try:
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model.to(self.device)
            self.logger.info("CLIP model loaded successfully")
        except Exception as e:
            self.logger.warning(f"Failed to load CLIP model: {e}")
            self.clip_model = None
            self.clip_processor = None
    
    def _init_blip(self):
        """Initialize BLIP model for image captioning."""
        try:
            self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            self.blip_model.to(self.device)
            self.logger.info("BLIP model loaded successfully")
        except Exception as e:
            self.logger.warning(f"Failed to load BLIP model: {e}")
            self.blip_model = None
            self.blip_processor = None
    
    def _init_preprocessing(self):
        """Initialize image preprocessing transforms."""
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def analyze_frames(self, frames: List[np.ndarray], 
                      scenes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze visual content of frames and scenes.
        
        Args:
            frames: List of video frames
            scenes: List of detected scenes
            
        Returns:
            Dictionary with visual analysis results
        """
        self.logger.info(f"Analyzing visual content of {len(frames)} frames")
        
        analysis = {
            'frame_analysis': [],
            'scene_analysis': [],
            'character_tracking': {},
            'object_detection': [],
            'environment_analysis': [],
            'cinematography': []
        }
        
        # Analyze individual frames
        for i, frame in enumerate(frames):
            frame_analysis = self._analyze_single_frame(frame, i)
            analysis['frame_analysis'].append(frame_analysis)
        
        # Analyze scenes
        for scene in scenes:
            scene_analysis = self._analyze_scene(scene, frames)
            analysis['scene_analysis'].append(scene_analysis)
        
        # Detect characters across frames
        analysis['character_tracking'] = self._track_characters(analysis['frame_analysis'])
        
        # Detect objects across all frames
        analysis['object_detection'] = self._detect_objects(analysis['frame_analysis'])
        
        # Analyze environments
        analysis['environment_analysis'] = self._analyze_environments(analysis['frame_analysis'])
        
        # Analyze cinematography
        analysis['cinematography'] = self._analyze_cinematography(frames, scenes)
        
        return analysis
    
    def _analyze_single_frame(self, frame: np.ndarray, frame_index: int) -> Dict[str, Any]:
        """Analyze a single frame for visual content."""
        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(frame)
        
        analysis = {
            'frame_index': frame_index,
            'objects': [],
            'environment': '',
            'characters': [],
            'emotion': 'neutral',
            'lighting': 'normal',
            'color_palette': [],
            'camera_movement': 'static',
            'confidence': 0.0
        }
        
        # Object detection using CLIP
        if self.clip_model is not None:
            objects = self._detect_objects_clip(pil_image)
            analysis['objects'] = objects
        
        # Environment description using BLIP
        if self.blip_model is not None:
            environment = self._describe_environment_blip(pil_image)
            analysis['environment'] = environment
        
        # Character detection (simplified)
        characters = self._detect_characters_simple(frame)
        analysis['characters'] = characters
        
        # Analyze lighting and color
        lighting, color_palette = self._analyze_lighting_color(frame)
        analysis['lighting'] = lighting
        analysis['color_palette'] = color_palette
        
        # Detect camera movement (simplified)
        camera_movement = self._detect_camera_movement(frame)
        analysis['camera_movement'] = camera_movement
        
        return analysis
    
    def _detect_objects_clip(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Detect objects in image using CLIP."""
        if self.clip_model is None:
            return []
        
        # Common object categories
        object_categories = [
            "person", "car", "building", "tree", "road", "sky", "water", "grass",
            "furniture", "food", "animal", "vehicle", "clothing", "book", "phone",
            "computer", "television", "chair", "table", "bed", "door", "window"
        ]
        
        try:
            # Process image and text
            inputs = self.clip_processor(
                text=object_categories, 
                images=image, 
                return_tensors="pt", 
                padding=True
            ).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)
            
            # Get top predictions
            top_indices = torch.topk(probs, k=min(5, len(object_categories))).indices[0]
            
            objects = []
            for idx in top_indices:
                category = object_categories[idx.item()]
                confidence = probs[0][idx].item()
                
                if confidence > 0.1:  # Threshold for object detection
                    objects.append({
                        'category': category,
                        'confidence': confidence,
                        'bounding_box': None  # CLIP doesn't provide bounding boxes
                    })
            
            return objects
            
        except Exception as e:
            self.logger.warning(f"CLIP object detection failed: {e}")
            return []
    
    def _describe_environment_blip(self, image: Image.Image) -> str:
        """Describe environment using BLIP."""
        if self.blip_model is None:
            return "Environment not analyzed"
        
        try:
            inputs = self.blip_processor(image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                out = self.blip_model.generate(**inputs, max_length=50)
            
            description = self.blip_processor.decode(out[0], skip_special_tokens=True)
            return description
            
        except Exception as e:
            self.logger.warning(f"BLIP environment description failed: {e}")
            return "Environment analysis failed"
    
    def _detect_characters_simple(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Simple character detection using OpenCV."""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Use Haar cascade for face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        characters = []
        for i, (x, y, w, h) in enumerate(faces):
            characters.append({
                'character_id': f"person_{i}",
                'bounding_box': [int(x), int(y), int(w), int(h)],
                'confidence': 0.8,  # Placeholder confidence
                'type': 'person'
            })
        
        return characters
    
    def _analyze_lighting_color(self, frame: np.ndarray) -> Tuple[str, List[str]]:
        """Analyze lighting conditions and color palette."""
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        
        # Analyze brightness
        brightness = np.mean(hsv[:, :, 2])
        
        if brightness < 50:
            lighting = "dark"
        elif brightness < 150:
            lighting = "dim"
        elif brightness < 200:
            lighting = "normal"
        else:
            lighting = "bright"
        
        # Analyze dominant colors
        pixels = frame.reshape(-1, 3)
        
        # Simple color clustering
        from sklearn.cluster import KMeans
        
        try:
            kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
            kmeans.fit(pixels)
            
            colors = kmeans.cluster_centers_.astype(int)
            color_palette = [f"rgb({r},{g},{b})" for r, g, b in colors]
            
        except Exception:
            color_palette = ["rgb(128,128,128)"]  # Fallback
        
        return lighting, color_palette
    
    def _detect_camera_movement(self, frame: np.ndarray) -> str:
        """Detect camera movement type (simplified)."""
        # This is a simplified implementation
        # In practice, you would compare with previous frames
        
        # For now, return static as default
        return "static"
    
    def _analyze_scene(self, scene: Dict[str, Any], frames: List[np.ndarray]) -> Dict[str, Any]:
        """Analyze visual content of a scene."""
        scene_analysis = {
            'scene_id': scene['scene_id'],
            'visual_summary': scene.get('visual_summary', ''),
            'dominant_objects': [],
            'environment_type': '',
            'lighting_consistency': 'consistent',
            'color_scheme': [],
            'camera_work': [],
            'visual_rhythm': 'moderate'
        }
        
        # Analyze shots in the scene
        shot_analyses = []
        for shot in scene['shots']:
            start_frame = shot['start_frame']
            end_frame = shot['end_frame']
            
            # Get representative frame (middle of shot)
            mid_frame_idx = (start_frame + end_frame) // 2
            if mid_frame_idx < len(frames):
                shot_analysis = self._analyze_single_frame(frames[mid_frame_idx], mid_frame_idx)
                shot_analyses.append(shot_analysis)
        
        if shot_analyses:
            # Aggregate shot analyses
            all_objects = []
            all_environments = []
            all_lighting = []
            
            for analysis in shot_analyses:
                all_objects.extend(analysis['objects'])
                all_environments.append(analysis['environment'])
                all_lighting.append(analysis['lighting'])
            
            # Find dominant objects
            object_counts = {}
            for obj in all_objects:
                category = obj['category']
                object_counts[category] = object_counts.get(category, 0) + 1
            
            scene_analysis['dominant_objects'] = sorted(
                object_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5]
            
            # Determine environment type
            if all_environments:
                scene_analysis['environment_type'] = max(set(all_environments), key=all_environments.count)
            
            # Check lighting consistency
            unique_lighting = set(all_lighting)
            if len(unique_lighting) == 1:
                scene_analysis['lighting_consistency'] = 'consistent'
            elif len(unique_lighting) <= 2:
                scene_analysis['lighting_consistency'] = 'mostly_consistent'
            else:
                scene_analysis['lighting_consistency'] = 'variable'
        
        return scene_analysis
    
    def _track_characters(self, frame_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Track characters across frames."""
        character_tracking = {
            'characters': {},
            'appearances': [],
            'interactions': []
        }
        
        character_id_counter = 0
        
        for frame_idx, analysis in enumerate(frame_analyses):
            for character in analysis['characters']:
                char_id = character['character_id']
                
                if char_id not in character_tracking['characters']:
                    character_tracking['characters'][char_id] = {
                        'character_id': char_id,
                        'first_appearance': frame_idx,
                        'last_appearance': frame_idx,
                        'total_frames': 1,
                        'bounding_boxes': []
                    }
                    character_id_counter += 1
                else:
                    char_data = character_tracking['characters'][char_id]
                    char_data['last_appearance'] = frame_idx
                    char_data['total_frames'] += 1
                
                # Store bounding box
                if character['bounding_box']:
                    character_tracking['characters'][char_id]['bounding_boxes'].append({
                        'frame': frame_idx,
                        'bounding_box': character['bounding_box']
                    })
        
        return character_tracking
    
    def _detect_objects(self, frame_analyses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect and track objects across frames."""
        object_tracking = {}
        
        for frame_idx, analysis in enumerate(frame_analyses):
            for obj in analysis['objects']:
                category = obj['category']
                
                if category not in object_tracking:
                    object_tracking[category] = {
                        'category': category,
                        'first_appearance': frame_idx,
                        'last_appearance': frame_idx,
                        'total_frames': 1,
                        'max_confidence': obj['confidence']
                    }
                else:
                    obj_data = object_tracking[category]
                    obj_data['last_appearance'] = frame_idx
                    obj_data['total_frames'] += 1
                    obj_data['max_confidence'] = max(obj_data['max_confidence'], obj['confidence'])
        
        return list(object_tracking.values())
    
    def _analyze_environments(self, frame_analyses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze environments across frames."""
        environments = []
        
        for frame_idx, analysis in enumerate(frame_analyses):
            if analysis['environment'] and analysis['environment'] != "Environment not analyzed":
                environments.append({
                    'frame': frame_idx,
                    'environment': analysis['environment'],
                    'lighting': analysis['lighting'],
                    'confidence': analysis['confidence']
                })
        
        return environments
    
    def _analyze_cinematography(self, frames: List[np.ndarray], 
                              scenes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze cinematographic elements."""
        cinematography = []
        
        for scene in scenes:
            scene_cinematography = {
                'scene_id': scene['scene_id'],
                'shot_count': len(scene['shots']),
                'average_shot_length': 0,
                'visual_rhythm': 'moderate',
                'camera_work': [],
                'lighting_style': 'normal'
            }
            
            if scene['shots']:
                total_duration = sum(shot['duration'] for shot in scene['shots'])
                scene_cinematography['average_shot_length'] = total_duration / len(scene['shots'])
                
                # Determine visual rhythm
                avg_shot_length = scene_cinematography['average_shot_length']
                if avg_shot_length < 2.0:
                    scene_cinematography['visual_rhythm'] = 'fast'
                elif avg_shot_length > 8.0:
                    scene_cinematography['visual_rhythm'] = 'slow'
            
            cinematography.append(scene_cinematography)
        
        return cinematography