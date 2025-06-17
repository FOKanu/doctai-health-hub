# Environment Variables Configuration

This document explains how to configure environment variables for the DocTAI Health Hub application.

## Feature Flags

### `VITE_USE_NEW_PREDICTION_API`
- **Default**: `false`
- **Description**: Controls whether to use the new modern prediction API or the legacy API
- **Values**: `'true'` | `'false'`

```bash
# To enable the new prediction API
export VITE_USE_NEW_PREDICTION_API=true
npm run dev
```

### `VITE_DEBUG_PREDICTIONS`
- **Default**: `false`
- **Description**: Enables detailed logging for prediction operations
- **Values**: `'true'` | `'false'`

```bash
# To enable debug logging
export VITE_DEBUG_PREDICTIONS=true
npm run dev
```

### `VITE_ML_API_ENDPOINT`
- **Default**: `'http://localhost:8000/api/predict'`
- **Description**: The endpoint for the ML prediction API
- **Values**: Any valid URL

```bash
# To set a custom ML API endpoint
export VITE_ML_API_ENDPOINT=https://your-ml-api.com/predict
npm run dev
```

## Usage Examples

### Safe Mode (Default)
```bash
# Uses legacy prediction API (safe, tested)
npm run dev
```

### Testing New Features
```bash
# Enable new API with debug logging
export VITE_USE_NEW_PREDICTION_API=true
export VITE_DEBUG_PREDICTIONS=true
npm run dev
```

### Production with New API
```bash
# Enable new API without debug logs
export VITE_USE_NEW_PREDICTION_API=true
npm run build
```

## Important Notes

1. **Vite Environment Variables**: This project uses Vite, so environment variables must be prefixed with `VITE_` to be accessible in the browser.

2. **Safe Defaults**: All feature flags default to `false` to ensure backward compatibility.

3. **Error Handling**: The application includes fallback mechanisms, so if the new API fails, it automatically falls back to the legacy implementation.

4. **Development vs Production**: Debug logging is automatically disabled in production builds regardless of the environment variable setting.
