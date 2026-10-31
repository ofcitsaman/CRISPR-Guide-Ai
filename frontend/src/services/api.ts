import {
  PredictionResult,
  BatchPredictResponse,
  MetricsResponse,
  ModelInfoResponse,
  DatasetInfoResponse,
} from '../types';

const API_BASE = import.meta.env.VITE_API_URL || '';

export async function checkHealth(): Promise<any> {
  const res = await fetch(`${API_BASE}/api/health`);
  if (!res.ok) throw new Error('Backend health check failed');
  return res.json();
}

export async function predictSingleGuide(
  sequence: string,
  modelName: string = 'XGBoost',
  datasetName: string = 'HeLa'
): Promise<PredictionResult> {
  const res = await fetch(`${API_BASE}/api/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      sequence,
      model_name: modelName,
      dataset_name: datasetName,
    }),
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({ detail: 'Network error' }));
    throw new Error(errorData.detail || 'Prediction failed');
  }

  const payload = await res.json();
  return payload.data;
}

export async function predictBatchGuides(
  guides: Array<{ id?: string; sequence: string }>,
  modelName: string = 'XGBoost',
  datasetName: string = 'HeLa'
): Promise<BatchPredictResponse> {
  const res = await fetch(`${API_BASE}/api/batch-predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      guides,
      model_name: modelName,
      dataset_name: datasetName,
    }),
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({ detail: 'Network error' }));
    throw new Error(errorData.detail || 'Batch prediction failed');
  }

  return res.json();
}

export async function uploadCsvBatch(
  file: File,
  modelName: string = 'XGBoost',
  datasetName: string = 'HeLa'
): Promise<BatchPredictResponse> {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('model_name', modelName);
  formData.append('dataset_name', datasetName);

  const res = await fetch(`${API_BASE}/api/upload-csv`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({ detail: 'Network error' }));
    throw new Error(errorData.detail || 'CSV upload failed');
  }

  return res.json();
}

export const uploadCsvGuides = uploadCsvBatch;

export async function fetchMetrics(datasetName?: string): Promise<MetricsResponse> {
  const url = datasetName
    ? `${API_BASE}/api/metrics?dataset=${encodeURIComponent(datasetName)}`
    : `${API_BASE}/api/metrics`;
  const res = await fetch(url);
  if (!res.ok) throw new Error('Failed to load model metrics');
  return res.json();
}

export async function fetchFeatureImportance(datasetName: string = 'HeLa'): Promise<any> {
  const res = await fetch(`${API_BASE}/api/feature-importance?dataset=${encodeURIComponent(datasetName)}`);
  if (!res.ok) throw new Error('Failed to load feature importance');
  return res.json();
}

export async function fetchModelInfo(): Promise<ModelInfoResponse> {
  const res = await fetch(`${API_BASE}/api/model-info`);
  if (!res.ok) throw new Error('Failed to load model descriptions');
  return res.json();
}

export async function fetchDatasetInfo(): Promise<DatasetInfoResponse> {
  const res = await fetch(`${API_BASE}/api/dataset-info`);
  if (!res.ok) throw new Error('Failed to load dataset details');
  return res.json();
}
