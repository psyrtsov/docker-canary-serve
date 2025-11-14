# Word Boosting Guide

Word boosting allows you to bias the ASR model to better recognize specific words or phrases without retraining. This is perfect for:

- Voice control commands
- Industry-specific terminology
- Product/brand names
- Uncommon words

## How It Works

Word boosting uses GPU-PB (GPU-accelerated Phrase Boosting) to increase the likelihood of specific words during decoding.

## API Usage

### Format

Pass `word_boosting` as a JSON array of objects with `word` and `score`:

```json
[
  {"word": "nvidia", "score": 50},
  {"word": "vscode", "score": 50},
  {"word": "delete", "score": 100}
]
```

### Boost Scores

- **Positive scores (20-100)**: Increase word likelihood
- **Negative scores (-20 to -100)**: Decrease word likelihood
- **Higher score**: Stronger bias

### Examples

#### Voice Control Commands

```bash
curl http://localhost:9000/v1/audio/transcriptions \
  -F file=@command.wav \
  -F language=en \
  -F 'word_boosting=[{"word":"open","score":80},{"word":"close","score":80},{"word":"minimize","score":80},{"word":"click","score":80}]'
```

#### Application Names

```bash
curl http://localhost:9000/inference \
  -F file=@audio.wav \
  -F language=en \
  -F response_format=text \
  -F 'word_boosting=[{"word":"chrome","score":60},{"word":"firefox","score":60},{"word":"vscode","score":60},{"word":"terminal","score":60}]'
```

#### Technical Terms

```bash
curl http://localhost:9000/v1/audio/transcriptions \
  -F file=@technical.wav \
  -F language=en \
  -F 'word_boosting=[{"word":"kubernetes","score":70},{"word":"docker","score":70},{"word":"gpu","score":70}]'
```

#### Negative Boosting (Discourage Words)

```bash
curl http://localhost:9000/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F language=en \
  -F 'word_boosting=[{"word":"profanity","score":-100}]'
```

## Voice Control Example List

```json
[
  {"word": "open", "score": 80},
  {"word": "close", "score": 80},
  {"word": "minimize", "score": 80},
  {"word": "maximize", "score": 80},
  {"word": "click", "score": 90},
  {"word": "double-click", "score": 90},
  {"word": "right-click", "score": 90},
  {"word": "type", "score": 80},
  {"word": "delete", "score": 100},
  {"word": "backspace", "score": 80},
  {"word": "enter", "score": 80},
  {"word": "escape", "score": 80},
  {"word": "tab", "score": 80},
  {"word": "file", "score": 70},
  {"word": "folder", "score": 70},
  {"word": "window", "score": 70},
  {"word": "chrome", "score": 70},
  {"word": "firefox", "score": 70},
  {"word": "vscode", "score": 70},
  {"word": "terminal", "score": 70},
  {"word": "save", "score": 80},
  {"word": "copy", "score": 80},
  {"word": "paste", "score": 80},
  {"word": "undo", "score": 80},
  {"word": "redo", "score": 80}
]
```

## Performance

- **Speed**: GPU-PB adds only 2-5% overhead
- **Scalability**: Supports up to 20,000 boosted words
- **No training required**

## Limitations

- Works best with beam search decoding (beam_size > 1)
- Phrase boosting (multi-word) has limited support
- Case-sensitive matching
