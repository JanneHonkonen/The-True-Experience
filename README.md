# The True Experience – AI Film Analysis System

A comprehensive video analysis system that processes video files and generates detailed, timestamped JSON summaries of all visual events, dialogue, characters, sound, music, emotional content, and narrative structure.

## Features

- **Video Processing**: Frame extraction at configurable FPS with audio preprocessing
- **Scene Detection**: Advanced shot boundary detection using histogram delta, SSIM, and PySceneDetect
- **Visual Analysis**: Object detection, character tracking, environment analysis using CLIP and BLIP models
- **Audio Analysis**: Speech-to-text with Whisper, speaker identification, emotion analysis
- **Multimodal Fusion**: Intelligent combination of visual, audio, and dialogue data
- **Timeline Generation**: Detailed emotional and structural timeline with fixed intervals
- **JSON Output**: Comprehensive, validated JSON schema for LLM processing

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ai-film-analysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install additional dependencies for source separation (optional):
```bash
pip install demucs
```

## Usage

### Basic Usage

```bash
python main.py input_video.mp4 -o analysis_output.json
```

### Advanced Usage

```bash
python main.py input_video.mp4 \
    --output detailed_analysis.json \
    --fps 8.0 \
    --timeline-interval 3.0 \
    --gpu \
    --verbose
```

### Command Line Options

- `video_path`: Path to input video file (required)
- `-o, --output`: Output JSON file path (default: analysis_output.json)
- `-v, --verbose`: Enable verbose logging
- `--fps`: Frame rate for analysis (default: 6.0 fps)
- `--timeline-interval`: Timeline interval in seconds (default: 5.0s)
- `--gpu`: Use GPU acceleration when available

## Output Format

The system generates a comprehensive JSON file with the following structure:

```json
{
  "metadata": {
    "video_duration": 5400,
    "frame_rate": 6.0,
    "audio_rate": 44100,
    "timeline_interval": 5.0
  },
  "scenes": [
    {
      "scene_id": 1,
      "start_time": 0.0,
      "end_time": 120.5,
      "visual_summary": "Opening scene with establishing shot",
      "audio_summary": "Background music and ambient sound",
      "emotion": "tension",
      "narrative_function": "introduction",
      "characters_present": ["John", "Sarah"],
      "objects_detected": [...],
      "environment": "urban street",
      "lighting": "dim",
      "camera_work": "static"
    }
  ],
  "timeline": [
    {
      "start_time": "00:03:10",
      "end_time": "00:03:20",
      "visual_mood": "cold and dim",
      "audio_mood": "minor strings, wind",
      "emotion": "loneliness",
      "characters": ["John", "Unknown Figure"],
      "dialogue": [
        {
          "speaker": "John",
          "text": "Who's there?",
          "emotion": "nervous"
        }
      ],
      "sound_effects": [
        {"type": "door_creak", "intensity": "medium"}
      ],
      "music": {"mood": "eerie", "genre": "orchestral"},
      "narrative_function": "tension buildup"
    }
  ]
}
```

## Architecture

### Core Components

1. **VideoProcessor**: Handles video loading and frame extraction
2. **AudioProcessor**: Manages audio processing and feature extraction
3. **SceneDetector**: Detects scenes and shots using multiple algorithms
4. **VisualAnalyzer**: Analyzes visual content using CLIP and BLIP models
5. **DialogueProcessor**: Processes speech-to-text and speaker identification
6. **MultimodalFusion**: Combines data from all modalities
7. **TimelineGenerator**: Creates detailed timeline analysis
8. **AnalysisSchema**: Validates output JSON structure

### Processing Pipeline

1. **Input Processing**: Extract video frames and audio at specified rates
2. **Scene Detection**: Identify shot boundaries and group into scenes
3. **Visual Analysis**: Analyze each frame for objects, characters, environments
4. **Audio Analysis**: Transcribe speech, identify speakers, analyze emotions
5. **Multimodal Fusion**: Combine all data sources with conflict resolution
6. **Timeline Generation**: Create detailed timeline with fixed intervals
7. **Output Generation**: Generate validated JSON output

## Performance Guidelines

- **GPU Recommended**: For visual analysis with CLIP/BLIP models
- **Memory**: 8GB+ RAM recommended for large videos
- **Storage**: Ensure sufficient space for temporary audio files
- **Processing Time**: ~1-2x real-time for 6 FPS analysis

## Error Handling

The system includes comprehensive error handling:

- **Silence Detection**: Marks audio mood as "silence" explicitly
- **Low Confidence**: Includes confidence fields for all outputs
- **Corrupted Segments**: Skips with error flags
- **Validation**: JSON schema validation before output

## Dependencies

### Core Dependencies
- OpenCV (video processing)
- PyTorch (ML models)
- Transformers (CLIP, BLIP, Whisper)
- Librosa (audio processing)
- NumPy, SciPy (numerical computing)

### Optional Dependencies
- Demucs (audio source separation)
- PySceneDetect (advanced scene detection)
- CUDA (GPU acceleration)

## Examples

### Example 1: Basic Analysis
```bash
python main.py movie_clip.mp4
```

### Example 2: High-Resolution Analysis
```bash
python main.py movie_clip.mp4 --fps 12.0 --timeline-interval 2.0 --gpu
```

### Example 3: Verbose Debugging
```bash
python main.py movie_clip.mp4 --verbose --output debug_analysis.json
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce FPS or use CPU mode
2. **Audio Processing Errors**: Check audio codec compatibility
3. **Scene Detection Issues**: Adjust thresholds in SceneDetector
4. **Validation Errors**: Check JSON schema compliance

### Debug Mode

Enable verbose logging to see detailed processing information:
```bash
python main.py input_video.mp4 --verbose
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

See project main branch

## Citation

If you use this system in your research, please cite:

```bibtex
@software{ai_film_analysis,
  title={The True Experience – AI Film Analysis System},
  author={AI Film Analysis Team},
  year={2024},
  url={https://github.com/your-repo/ai-film-analysis}
}
```

## Support

For issues and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the verbose logs for debugging information
