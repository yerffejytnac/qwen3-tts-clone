# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-02-03

### Added
- Voice cloning system using Qwen3-TTS-12Hz-1.7B-Base model
- Multi-reference audio sample support (ref_1, ref_2) with 1s silence gaps
- Reusable voice model generation and serialization (jeffrey_voice.pkl)
- Gradio web UI with parameter controls and audio examples
- Realtime TTS using pre-generated voice models
- Multi-language support (English, Chinese, Japanese, Korean, French, German, Spanish)
- Advanced sampling parameters (temperature, top_p, top_k, repetition_penalty)
- Subtalker-specific sampling controls for improved quality
- X-vector only mode option for faster generation
- Command-line scripts: clone_voice.py, create_voice_prompt.py, realtime_tts.py
- Web interface: demo_app.py with interactive parameter controls
- Documentation: configuration guide, realtime usage, references
- Pre-generated voice model and sample outputs

### Changed
- None

### Deprecated
- None

### Removed
- None

### Fixed
- None

### Security
- None

---

## Conventional Commits

This project follows the [Conventional Commits](https://www.conventionalcommits.org/) specification.

### Format

```
<type>[optional scope]: <description>

[optional body]
```

### Types
- **feat**: New feature
- **fix**: Bug fix
- **docs**: Documentation changes
- **refactor**: Code refactoring
- **perf**: Performance improvement
- **test**: Tests
- **build**: Build system changes
- **chore**: Maintenance

### Examples
```
feat(clone): add multi-reference voice cloning
fix(types): resolve type annotation issues
docs(config): update generation parameters
refactor: simplify reference file handling
```
