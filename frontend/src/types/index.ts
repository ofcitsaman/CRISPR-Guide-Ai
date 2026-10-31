export interface BiophysicalProperties {
  gc_content_pct: number;
  seed_gc_pct: number;
  pam_motif: string;
  has_canonical_pam: boolean;
  poly_t_max_run: number;
  has_terminator: boolean;
  distal_gc_pct: number;
  cleavage_flank_gc_pct: number;
  warnings: string[];
}

export interface FeatureAttribution {
  feature: string;
  global_importance: number;
  sample_value_normalized: number;
  effect_direction: 'positive' | 'negative';
  interpretation: string;
  biological_context?: string;
}

export interface RocCurveData {
  fpr: number[];
  tpr: number[];
  auc: number;
}

export interface ModelMetricsSummary {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  specificity: number;
  roc_auc: number;
  pr_auc: number;
  rmse: number;
  mae: number;
  mse: number;
  r2: number;
  pearson: number;
  spearman: number;
}

export interface CvMetricsItem {
  mean: ModelMetricsSummary;
  std: ModelMetricsSummary;
  threshold: number;
}

export interface FinalHoldoutMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  specificity: number;
  roc_auc: number;
  pr_auc: number;
  rmse: number;
  mae: number;
  mse: number;
  r2: number;
  pearson: number;
  spearman: number;
  threshold: number;
}

export interface ModelVerificationData {
  dataset: string;
  model: string;
  protocol: string;
  cv_100_fold?: CvMetricsItem;
  final_holdout_validation?: FinalHoldoutMetrics;
  roc_curve?: RocCurveData;
  note?: string;
}

export interface PredictionResult {
  sequence: string;
  was_extended_with_pam: boolean;
  sequence_length: number;
  dataset_used: string;
  dataset_key?: string;
  model_used: string;
  predicted_efficiency_score: number;
  relative_efficacy_score: number;
  scaled_efficiency: number;
  relative_efficacy_score_display: string;
  score_explanation: string;
  activity_classification: string;
  binary_prediction?: number;
  binary_prediction_label?: string;
  classification_probability?: number;
  classification_probability_pct?: number;
  classification_model_used?: string;
  decision_threshold?: number;
  classification_summary?: string;
  ensemble_dispersion: number | null;
  dispersion_bounds: [number, number] | null;
  uncertainty: number | null;
  confidence_interval: [number, number] | null;
  uncertainty_note: string;
  biophysical_properties: BiophysicalProperties;
  positional_saliency: number[];
  top_contributing_features: FeatureAttribution[];
  verification_metrics?: ModelVerificationData;
  scientific_disclaimer: string;
}

export interface BatchResultItem {
  guide_id: string;
  sequence: string;
  rank?: number;
  status: 'success' | 'error';
  error?: string | null;
  predicted_efficiency_score?: number | null;
  relative_efficacy_score?: number | null;
  scaled_efficiency?: number | null;
  activity_classification?: string;
  batch_category?: string;
  binary_prediction?: number;
  binary_prediction_label?: string;
  classification_probability?: number;
  classification_model_used?: string;
  decision_threshold?: number;
  ensemble_dispersion?: number | null;
  dataset_used?: string;
  model_used?: string;
  biophysical_properties?: BiophysicalProperties | null;
}

export interface BatchPredictResponse {
  status: string;
  dataset_used: string;
  model_used: string;
  total_submitted: number;
  total_successful: number;
  results: BatchResultItem[];
}

export interface DatasetSummary {
  cell_line: string;
  display_name: string;
  total_samples: number;
  unique_sequences: number;
  positive_labels: number;
  negative_labels: number;
  development_pool_count: number;
  final_holdout_count: number;
  cv_folds: number;
  sequence_overlap_count: number;
  random_seed: number;
}

export interface DatasetMetrics {
  cell_line: string;
  display_name: string;
  summary: DatasetSummary;
  models: Record<
    string,
    {
      cv_100_fold: CvMetricsItem;
      final_validation_20: FinalHoldoutMetrics;
      threshold: number;
      roc_curve: RocCurveData;
    }
  >;
  regression_models?: Record<
    string,
    {
      cv_100_fold: CvMetricsItem;
      final_validation_20: FinalHoldoutMetrics;
      threshold: number;
      roc_curve: RocCurveData;
    }
  >;
  classification_models?: Record<
    string,
    {
      cv_100_fold: CvMetricsItem;
      final_validation_20: FinalHoldoutMetrics;
      threshold: number;
      roc_curve: RocCurveData;
    }
  >;
  baselines?: Record<string, any>;
  feature_importance: Array<{
    feature: string;
    importance: number;
    rank: number;
  }>;
}

export interface MetricsResponse {
  protocol?: {
    name: string;
    description: string;
    cv_folds: number;
    random_seed: number;
    sequence_overlap_count: number;
    models_evaluated: string[];
  };
  datasets?: Record<string, DatasetMetrics>;
  // Single-dataset direct response
  cell_line?: string;
  display_name?: string;
  summary?: DatasetSummary;
  models?: Record<
    string,
    {
      cv_100_fold: CvMetricsItem;
      final_validation_20: FinalHoldoutMetrics;
      threshold: number;
      roc_curve: RocCurveData;
    }
  >;
  regression_models?: Record<
    string,
    {
      cv_100_fold: CvMetricsItem;
      final_validation_20: FinalHoldoutMetrics;
      threshold: number;
      roc_curve: RocCurveData;
    }
  >;
  classification_models?: Record<
    string,
    {
      cv_100_fold: CvMetricsItem;
      final_validation_20: FinalHoldoutMetrics;
      threshold: number;
      roc_curve: RocCurveData;
    }
  >;
  baselines?: Record<string, any>;
  feature_importance?: Array<{
    feature: string;
    importance: number;
    rank: number;
  }>;
}

export interface ModelInfoItem {
  name: string;
  type: string;
  features_used: string;
  description: string;
  strengths: string;
  limitations: string;
}

export interface ModelInfoResponse {
  models: ModelInfoItem[];
  biophysical_context: {
    target_system: string;
    pam_requirement: string;
    seed_region: string;
    cleavage_plane: string;
    transcription_terminator: string;
  };
}

export interface DatasetCellItem {
  id: string;
  name: string;
  samples: number;
  unique_sgrnas: number;
  positive_count: number;
  negative_count: number;
  organism: string;
  assay_type: string;
  original_study: string;
  doi: string;
}

export interface CitationItem {
  title: string;
  authors: string;
  journal: string;
  year: number;
  doi: string;
  url: string;
}

export interface DatasetInfoResponse {
  dataset_name: string;
  description: string;
  cell_lines: DatasetCellItem[];
  target_variables: {
    continuous_target: string;
    binary_target: string;
  };
  citations: CitationItem[];
}
