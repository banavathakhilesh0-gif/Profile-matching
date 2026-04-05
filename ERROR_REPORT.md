# Error Report - Profile Matching ML

## Errors Found:

### 1. **Scikit-Learn Version Mismatch Warning** ⚠️
**File:** `pipeline.py` (cached pickle file)  
**Issue:** The pickled pipeline was created with scikit-learn 1.6.1 but the system is using version 1.8.0
```
InconsistentVersionWarning: Trying to unpickle estimator TfidfTransformer from 
version 1.6.1 when using version 1.8.0. This might lead to breaking code or invalid results.
```
**Severity:** Medium - May cause unexpected behavior  
**Solution:** Delete the `pipeline_cache.pkl` file to force rebuild with current scikit-learn version

---

### 2. **Hardcoded File Paths in pipeline.py** ❌
**File:** [pipeline.py](pipeline.py#L78-L79)  
**Issue:** The `__main__` block has hardcoded paths that won't exist:
```python
if __name__ == "__main__":
    pipeline = DataPipeline('e:/Profile matching/users.csv', 'e:/Profile matching/feedback.csv')
    pipeline.run()
```
These paths point to `e:/Profile matching/` which likely doesn't exist on the system.

**Severity:** High - Will crash if script is run directly  
**Solution:** Update hardcoded paths or use relative paths/environment variables

---

### 3. **NumPy Type Issues in Scoring Results** ⚠️
**File:** `scoring_engine.py`  
**Issue:** The `get_score()` method returns NumPy float64 types instead of Python floats:
```python
'text_sim': np.float64(0.023...), 'total_score': np.float64(0.257...)
```
While this works in most cases, it may cause issues with JSON serialization in the FastAPI endpoint in `server.py`.

**Severity:** Medium - May break API responses or data persistence  
**Solution:** Convert numpy types to Python types before returning

---

## Summary:
- **Critical Issues:** 2 (hardcoded paths)
- **Medium Issues:** 2 (version mismatch, numpy type conversion)
- **Status:** Core functionality works, but edge cases and file handling need fixes
