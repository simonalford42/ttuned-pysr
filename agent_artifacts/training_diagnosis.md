# Training Diagnosis: Neural SR Model

## Summary
The model is **severely overfitting** and has barely seen the training data (only 10% of one epoch). The eval loss is increasing while train loss decreases, and the model produces mostly degenerate outputs (simple variables like `x0` instead of complex expressions).

## Key Findings

### 1. **Severe Overfitting**
- **Training progress**: Only 0.10 epochs (10% of training data seen once)
- **Train loss**: 0.42 (decreasing)
- **Eval loss trajectory**: 1.25 (step 20k) → 1.34 (step 40k) → 1.42 (step 60k) ❌
- **Best checkpoint**: Step 20,000 (eval loss 1.25)
- **Current checkpoint**: Step 60,000 (eval loss 1.42) - **40k steps past best!**

### 2. **Poor Generation Quality**
Model predictions on validation data are mostly degenerate:
- Predictions: `x0`, `x1`, `(x0 - x0)`, `x0`, etc. (mostly simple variables)
- Expected: Complex expressions like `((x2 * x2) + ((x0 * x0) + (x1 * x1)))`

From neural_comparison.py output:
```
PREDICTED TARGETS: x0 x0 x0 x0 x0 x0 x0 x0 x0 x0 ...
```
The model is stuck predicting simple patterns.

### 3. **Training Data Statistics**
- **Total training examples**: 1,000,000
- **Target distribution**:
  - 77.8% start with `(` (complex expressions)
  - 22.2% start with `x` or numbers (simple expressions)
- **First token accuracy on training set**: 61.8% ❌
  - Should be much higher if model is learning properly

### 4. **Model Predictions vs Data Distribution**
On validation data, model predicts:
- `x` token: 41.4% probability (rank 1)
- `(` token: 31.9% probability (rank 2)
- `((` token: 23.5% probability (rank 3)

But training data shows:
- 77.8% of targets start with `(`
- Only 22.2% start with `x`

**The model is inverting the distribution!** This is a strong sign it hasn't learned the task.

### 5. **Tokenization Details**
Tokenization appears correct:
- `<TARGET>` token ID: 50260
- `((` token ID: 19510
- `(` token ID: 7
- `x` token ID: 87
- Vocab size: 50,261

### 6. **Training Configuration Issues**

From `onestep-tiny.json`:
```json
"num_train_epochs": 1
"per_device_train_batch_size": 4
"gradient_accumulation_steps": 4
```

Effective batch size: 4 * 4 = 16

With 1M training examples and batch size 16:
- Steps per epoch: 1,000,000 / 16 = 62,500
- At step 60,000: only 0.96 epochs
- **But epoch counter shows 0.10 epochs!**

This suggests the effective training may be even slower than expected, or there's a mismatch in how steps/epochs are counted.

## Root Causes

1. **Training stopped too early** (or hasn't progressed far enough)
   - Only saw 10% of one epoch
   - Model hasn't had chance to learn the full distribution

2. **Overfitting is severe**
   - Eval loss increasing from step 20k onward
   - Should have stopped at step 20k (best checkpoint)
   - `load_best_model_at_end: true` should help, but inference is using step 60k

3. **Possible data issues**
   - Very short targets (avg 13-15 tokens to learn)
   - Only 1.5-3% of sequence length is actual training signal
   - Model may need more epochs to learn from such sparse supervision

## Recommendations

### Immediate Actions

1. **Use the best checkpoint (step 20k) instead of step 60k**
   ```bash
   python neural_comparison.py --checkpoint training/checkpoints/tiny_183867/checkpoint-20000
   ```

2. **Train for more epochs**
   - Current: 1 epoch (incomplete - only 10% done)
   - Recommended: At least 3-5 epochs
   - Monitor eval loss carefully to detect overfitting

3. **Adjust batch size or learning rate**
   - Current effective batch size: 16 (very small)
   - Consider increasing to 32-64 for more stable gradients
   - Or reduce learning rate (currently 1e-4)

4. **Add regularization**
   - Current weight_decay: 0.01 (reasonable)
   - Consider adding dropout to model
   - Try label smoothing

### Training Configuration Changes

```json
{
  "num_train_epochs": 5,  // Increase from 1
  "per_device_train_batch_size": 8,  // Increase from 4
  "gradient_accumulation_steps": 4,
  "learning_rate": 5e-5,  // Reduce from 1e-4
  "warmup_steps": 4000,
  "eval_steps": 5000,  // More frequent evals (was 20000)
  "save_steps": 5000,  // More frequent saves
  "early_stopping_patience": 3  // Stop after 3 evals without improvement
}
```

### Data Improvements

1. **Increase target complexity in training data**
   - Current: lots of simple expressions (x0, x1)
   - Need more complex targets to match test distribution

2. **Balance the dataset**
   - 77.8% complex vs 22.2% simple is good
   - But ensure model sees this distribution enough times

3. **Consider curriculum learning**
   - Start with simple expressions
   - Gradually introduce more complex ones

## Files Created

Analysis scripts in `agent_artifacts/`:
1. `inspect_training.py` - Inspect training data and tokenization
2. `test_generation.py` - Test model generation behavior
3. `analyze_target_distribution.py` - Analyze target distribution
4. `check_loss_computation.py` - Verify loss computation
5. `training_diagnosis.md` - This file

## Next Steps

1. Test with checkpoint-20000 (best model)
2. If still poor, retrain with recommended settings
3. Monitor eval loss closely - stop when it starts increasing
4. Consider using a learning rate scheduler that reduces LR when eval loss plateaus
