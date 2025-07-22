# Edge AI: Recyclable Item Classifier (Minimal)

## Quick Start

1. **Install dependencies**
   ```
   pip install -r requirements.txt
   ```

2. **Prepare dataset**
   - Place images in `dataset/class_name/` folders.

3. **Train model**
   ```
   python train.py
   ```

4. **Convert to TFLite**
   ```
   python convert.py
   ```

5. **Evaluate**
   ```
   python evaluate.py
   ```

## Files

- `train.py` — Train and save Keras model
- `convert.py` — Convert to TFLite
- `evaluate.py` — Evaluate TFLite model
