"""
JSON schema validation module.

Defines and validates the output schema for the video analysis system.
"""

import json
import jsonschema
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime


class AnalysisSchema:
    """Handles JSON schema validation for analysis output."""
    
    def __init__(self):
        """Initialize schema validator."""
        self.logger = logging.getLogger(__name__)
        self.schema = self._create_schema()
    
    def _create_schema(self) -> Dict[str, Any]:
        """Create JSON schema for analysis output."""
        return {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "required": ["metadata", "scenes", "timeline"],
            "properties": {
                "metadata": {
                    "type": "object",
                    "required": ["video_duration", "frame_rate", "audio_rate"],
                    "properties": {
                        "video_duration": {"type": "number", "minimum": 0},
                        "frame_rate": {"type": "number", "minimum": 0},
                        "audio_rate": {"type": "number", "minimum": 0},
                        "timeline_interval": {"type": "number", "minimum": 0},
                        "video_path": {"type": "string"},
                        "analysis_timestamp": {"type": "string"}
                    }
                },
                "scenes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["scene_id", "start_time", "end_time"],
                        "properties": {
                            "scene_id": {"type": "integer", "minimum": 0},
                            "start_time": {"type": "number", "minimum": 0},
                            "end_time": {"type": "number", "minimum": 0},
                            "duration": {"type": "number", "minimum": 0},
                            "shots": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "required": ["shot_id", "start_frame", "end_frame"],
                                    "properties": {
                                        "shot_id": {"type": "integer", "minimum": 0},
                                        "start_frame": {"type": "integer", "minimum": 0},
                                        "end_frame": {"type": "integer", "minimum": 0},
                                        "start_time": {"type": "number", "minimum": 0},
                                        "end_time": {"type": "number", "minimum": 0},
                                        "duration": {"type": "number", "minimum": 0}
                                    }
                                }
                            },
                            "visual_summary": {"type": "string"},
                            "audio_summary": {"type": "string"},
                            "emotion": {"type": "string"},
                            "narrative_function": {"type": "string"},
                            "characters_present": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "objects_detected": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "category": {"type": "string"},
                                        "confidence": {"type": "number", "minimum": 0, "maximum": 1}
                                    }
                                }
                            },
                            "environment": {"type": "string"},
                            "lighting": {"type": "string"},
                            "camera_work": {"type": "string"},
                            "dialogue_count": {"type": "integer", "minimum": 0},
                            "music_present": {"type": "boolean"},
                            "sound_effects": {
                                "type": "array",
                                "items": {"type": "string"}
                            }
                        }
                    }
                },
                "timeline": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["start_time", "end_time", "emotion"],
                        "properties": {
                            "start_time": {"type": "string"},
                            "end_time": {"type": "string"},
                            "duration": {"type": "number", "minimum": 0},
                            "visual_mood": {"type": "string"},
                            "audio_mood": {"type": "string"},
                            "emotion": {"type": "string"},
                            "emotion_confidence": {"type": "number", "minimum": 0, "maximum": 1},
                            "characters": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "dialogue": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "speaker": {"type": "string"},
                                        "text": {"type": "string"},
                                        "emotion": {"type": "string"},
                                        "confidence": {"type": "number", "minimum": 0, "maximum": 1}
                                    }
                                }
                            },
                            "sound_effects": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "music": {
                                "type": "object",
                                "properties": {
                                    "present": {"type": "boolean"},
                                    "type": {"type": "string"}
                                }
                            },
                            "key_events": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "narrative_function": {"type": "string"},
                            "scene_ids": {
                                "type": "array",
                                "items": {"type": "integer"}
                            },
                            "contrasts_detected": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "type": {"type": "string"},
                                        "description": {"type": "string"},
                                        "confidence": {"type": "number", "minimum": 0, "maximum": 1}
                                    }
                                }
                            }
                        }
                    }
                },
                "fused_analysis": {
                    "type": "object",
                    "properties": {
                        "metadata": {
                            "type": "object",
                            "properties": {
                                "fusion_timestamp": {"type": "string"},
                                "data_sources": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                },
                                "total_scenes": {"type": "integer", "minimum": 0}
                            }
                        },
                        "scenes": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "scene_id": {"type": "integer"},
                                    "start_time": {"type": "number"},
                                    "end_time": {"type": "number"},
                                    "duration": {"type": "number"},
                                    "visual_summary": {"type": "string"},
                                    "audio_summary": {"type": "string"},
                                    "dialogue_summary": {"type": "string"},
                                    "emotion": {"type": "string"},
                                    "emotion_confidence": {"type": "number"},
                                    "narrative_function": {"type": "string"},
                                    "contrasts_detected": {
                                        "type": "array",
                                        "items": {"type": "object"}
                                    },
                                    "characters_present": {
                                        "type": "array",
                                        "items": {"type": "string"}
                                    },
                                    "objects_detected": {
                                        "type": "array",
                                        "items": {"type": "object"}
                                    },
                                    "environment": {"type": "string"},
                                    "lighting": {"type": "string"},
                                    "camera_work": {"type": "string"},
                                    "dialogue_count": {"type": "integer"},
                                    "music_present": {"type": "boolean"},
                                    "sound_effects": {
                                        "type": "array",
                                        "items": {"type": "string"}
                                    }
                                }
                            }
                        },
                        "timeline_fusion": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "start_time": {"type": "number"},
                                    "end_time": {"type": "number"},
                                    "scene_id": {"type": "integer"},
                                    "primary_emotion": {"type": "string"},
                                    "narrative_function": {"type": "string"},
                                    "characters_present": {
                                        "type": "array",
                                        "items": {"type": "string"}
                                    },
                                    "key_events": {
                                        "type": "array",
                                        "items": {"type": "string"}
                                    },
                                    "contrasts": {
                                        "type": "array",
                                        "items": {"type": "object"}
                                    }
                                }
                            }
                        },
                        "character_analysis": {
                            "type": "object",
                            "properties": {
                                "visual_characters": {"type": "object"},
                                "dialogue_speakers": {"type": "object"},
                                "character_mapping": {"type": "object"},
                                "character_development": {"type": "object"}
                            }
                        },
                        "narrative_structure": {
                            "type": "object",
                            "properties": {
                                "total_scenes": {"type": "integer"},
                                "scene_types": {"type": "object"},
                                "emotional_progression": {
                                    "type": "array",
                                    "items": {"type": "object"}
                                },
                                "narrative_arc": {"type": "string"}
                            }
                        },
                        "emotional_arc": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "scene_id": {"type": "integer"},
                                    "position": {"type": "number"},
                                    "emotion": {"type": "string"},
                                    "intensity": {"type": "number"},
                                    "narrative_function": {"type": "string"}
                                }
                            }
                        },
                        "conflicts_detected": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "type": {"type": "string"},
                                    "scene_id": {"type": "integer"},
                                    "description": {"type": "string"},
                                    "confidence": {"type": "number"}
                                }
                            }
                        }
                    }
                }
            }
        }
    
    def validate(self, data: Dict[str, Any]) -> bool:
        """
        Validate data against the schema.
        
        Args:
            data: Data to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            jsonschema.validate(data, self.schema)
            self.logger.info("Data validation successful")
            return True
        except jsonschema.ValidationError as e:
            self.logger.error(f"Validation error: {e.message}")
            self.logger.error(f"Error path: {'.'.join(str(p) for p in e.absolute_path)}")
            return False
        except Exception as e:
            self.logger.error(f"Validation failed with error: {e}")
            return False
    
    def validate_file(self, file_path: str) -> bool:
        """
        Validate JSON file against the schema.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            True if valid, False otherwise
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return self.validate(data)
        except FileNotFoundError:
            self.logger.error(f"File not found: {file_path}")
            return False
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error: {e}")
            return False
        except Exception as e:
            self.logger.error(f"File validation failed: {e}")
            return False
    
    def get_schema_errors(self, data: Dict[str, Any]) -> List[str]:
        """
        Get detailed validation errors.
        
        Args:
            data: Data to validate
            
        Returns:
            List of error messages
        """
        errors = []
        
        try:
            jsonschema.validate(data, self.schema)
        except jsonschema.ValidationError as e:
            errors.append(f"Validation error at {'.'.join(str(p) for p in e.absolute_path)}: {e.message}")
        except Exception as e:
            errors.append(f"Validation failed: {e}")
        
        return errors
    
    def create_sample_output(self) -> Dict[str, Any]:
        """Create a sample output structure for reference."""
        return {
            "metadata": {
                "video_duration": 120.5,
                "frame_rate": 6.0,
                "audio_rate": 44100,
                "timeline_interval": 5.0,
                "video_path": "/path/to/video.mp4",
                "analysis_timestamp": datetime.now().isoformat()
            },
            "scenes": [
                {
                    "scene_id": 0,
                    "start_time": 0.0,
                    "end_time": 30.5,
                    "duration": 30.5,
                    "shots": [
                        {
                            "shot_id": 0,
                            "start_frame": 0,
                            "end_frame": 180,
                            "start_time": 0.0,
                            "end_time": 30.0,
                            "duration": 30.0
                        }
                    ],
                    "visual_summary": "Opening scene with establishing shot",
                    "audio_summary": "Background music and ambient sound",
                    "emotion": "neutral",
                    "narrative_function": "setup",
                    "characters_present": ["character_1"],
                    "objects_detected": [
                        {"category": "person", "confidence": 0.9},
                        {"category": "building", "confidence": 0.8}
                    ],
                    "environment": "urban street",
                    "lighting": "normal",
                    "camera_work": "static",
                    "dialogue_count": 3,
                    "music_present": True,
                    "sound_effects": ["footsteps", "traffic"]
                }
            ],
            "timeline": [
                {
                    "start_time": "00:00:00",
                    "end_time": "00:00:05",
                    "duration": 5.0,
                    "visual_mood": "bright and cheerful",
                    "audio_mood": "musical",
                    "emotion": "neutral",
                    "emotion_confidence": 0.7,
                    "characters": ["character_1"],
                    "dialogue": [
                        {
                            "speaker": "character_1",
                            "text": "Hello, how are you?",
                            "emotion": "neutral",
                            "confidence": 0.8
                        }
                    ],
                    "sound_effects": ["footsteps"],
                    "music": {"present": True, "type": "background"},
                    "key_events": ["Character introduction", "Dialogue exchange"],
                    "narrative_function": "setup",
                    "scene_ids": [0],
                    "contrasts_detected": []
                }
            ]
        }