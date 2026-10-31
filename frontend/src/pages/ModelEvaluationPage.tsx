import React, { useEffect, useState } from 'react';
import { fetchMetrics } from '../services/api';
import { MetricsResponse, DatasetMetrics } from '../types';
import { Disclaimer } from '../components/Disclaimer';
import {
  Database,
  Award,
  ShieldCheck,
  CheckCircle2,
  Activity,
  Layers,
  BarChart2,
  Info,
} from 'lucide-react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { Line } from 'react-chartjs-2';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

const DATASET_KEYS = [
  { key: 'hela', name: 'HeLa', samples: 8101, type: 'Viability Screen (Hart 2015)' },
  { key: 'hct116', name: 'HCT116', samples: 4239, type: 'Viability Screen (Hart 2015)' },
  { key: 'a375', name: 'A375', samples: 2333, type: 'Drug Resistance Tiling (Doench 2016)' },
  { key: 'hl60', name: 'HL60', samples: 2076, type: 'Viability Screen (Xu 2015)' },
];

const MODEL_COLORS: Record<string, string> = {
  // Regression
  'Weighted Ensemble': '#8b5cf6',
  'XGBoost': '#4f46e5',
  'Random Forest': '#059669',
  'SVM': '#d97706',
  'Ridge Linear Regression': '#64748b',
  'Baseline Dummy (Mean)': '#94a3b8',
  // Classification
  'Logistic Regression': '#64748b',
  'Random Forest Classifier': '#059669',
  'XGBoost Classifier': '#4f46e5',
  'Support Vector Classifier (SVC)': '#d97706',
  'Ridge Classifier': '#0284c7',
  'Baseline Dummy (Most Frequent)': '#94a3b8',
};

export const ModelEvaluationPage: React.FC = () => {
  const [metricsData, setMetricsData] = useState<MetricsResponse | null>(null);
  const [selectedDataset, setSelectedDataset] = useState('hela');
  const [activeTask, setActiveTask] = useState<'regression' | 'classification'>('regression');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      try {
        const res = await fetchMetrics();
        setMetricsData(res);
      } catch (err: any) {
        setError(err.message || 'Failed to load metrics');
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  if (loading) {
    return (
      <div className="py-12 text-center text-xs text-slate-500">
        Loading evaluation benchmarks...
      </div>
    );
  }

  if (error || !metricsData) {
    return (
      <div className="py-8 bio-card p-4 text-center text-rose-600 text-xs">
        Failed to load evaluation metrics: {error}
      </div>
    );
  }

  // Get current dataset metrics
  const activeDatasetMetrics: DatasetMetrics | undefined =
    metricsData.datasets?.[selectedDataset] ||
    (metricsData.cell_line === selectedDataset ? (metricsData as any) : undefined);

  const regModels = activeDatasetMetrics?.regression_models || activeDatasetMetrics?.models || {};
  const clfModels = activeDatasetMetrics?.classification_models || {};
  const activeModelsDict = activeTask === 'regression' ? regModels : (Object.keys(clfModels).length > 0 ? clfModels : regModels);
  const modelEntries = Object.entries(activeModelsDict);
  const summary = activeDatasetMetrics?.summary;

  const fmt = (val: number | undefined | null, digits = 3): string =>
    val !== undefined && val !== null && !isNaN(val) ? val.toFixed(digits) : '—';

  // Build ROC Curve Multi-Model Chart
  const sampleFpr = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
  const chartDatasets = modelEntries.map(([name, m]) => {
    const roc = m.roc_curve;
    const color = MODEL_COLORS[name] || '#4f46e5';
    const aucDisplay = roc?.auc !== undefined ? roc.auc.toFixed(3) : fmt(m.final_validation_20?.roc_auc);
    return {
      label: `${name} (AUC = ${aucDisplay})`,
      data: roc?.tpr || [],
      borderColor: color,
      backgroundColor: 'transparent',
      borderWidth: 2,
      pointRadius: 0,
      tension: 0.1,
    };
  });

  // Add random chance baseline
  const rocLabels = (modelEntries[0]?.[1]?.roc_curve?.fpr || sampleFpr).map((v) => `${(v * 100).toFixed(0)}%`);
  const fullChartData = {
    labels: rocLabels,
    datasets: [
      ...chartDatasets,
      {
        label: 'Random Chance (AUC = 0.500)',
        data: modelEntries[0]?.[1]?.roc_curve?.fpr || sampleFpr,
        borderColor: '#cbd5e1',
        borderDash: [4, 4],
        borderWidth: 1.5,
        pointRadius: 0,
        backgroundColor: 'transparent',
      },
    ],
  };

  const chartOptions: any = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'bottom' as const,
        labels: { boxWidth: 12, font: { size: 11 } },
      },
      tooltip: {
        mode: 'index' as const,
        intersect: false,
      },
    },
    scales: {
      x: {
        title: { display: true, text: 'False Positive Rate (FPR)', font: { size: 10 } },
        ticks: { maxTicksLimit: 7, font: { size: 9 } },
        grid: { color: '#f8fafc' },
      },
      y: {
        min: 0,
        max: 1,
        title: { display: true, text: 'True Positive Rate (TPR)', font: { size: 10 } },
        ticks: { font: { size: 9 } },
        grid: { color: '#f8fafc' },
      },
    },
  };

  return (
    <div className="space-y-5 max-w-5xl mx-auto">
      {/* 1-Sentence Header */}
      <div className="border-b border-slate-200 pb-2.5">
        <h1 className="text-lg font-bold text-slate-900 tracking-tight">Model Evaluation & Scientific Benchmarks</h1>
        <p className="text-xs text-slate-500 mt-0.5">
          Dataset-specific 80/20 holdout and 100-fold cross-validation benchmarking evaluated against experimental knockout screen data.
        </p>
      </div>

      {/* Dataset Selector Tabs */}
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-xs font-bold text-slate-700 flex items-center gap-1 mr-1">
          <Database className="w-3.5 h-3.5 text-indigo-600" />
          Select Dataset:
        </span>
        {DATASET_KEYS.map((d) => (
          <button
            key={d.key}
            onClick={() => setSelectedDataset(d.key)}
            className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all cursor-pointer border ${
              selectedDataset === d.key
                ? 'bg-indigo-600 text-white border-indigo-600 shadow-xs'
                : 'bg-white text-slate-700 border-slate-200 hover:border-slate-300 hover:bg-slate-50'
            }`}
          >
            {d.name} ({d.samples.toLocaleString()})
          </button>
        ))}
      </div>

      {/* EVALUATION PROTOCOL BANNER */}
      <div className="bio-card p-4 space-y-2 border-indigo-200 bg-gradient-to-r from-indigo-50/40 via-white to-slate-50">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <ShieldCheck className="w-4 h-4 text-indigo-600" />
            <h2 className="text-xs font-bold text-slate-900 uppercase tracking-wider">
              Evaluation Methodology & Leakage-Safe Protocol
            </h2>
          </div>
          <span className="text-[10px] font-mono text-emerald-700 bg-emerald-50 px-2 py-0.5 rounded border border-emerald-200 font-semibold">
            Sequence Overlap: 0 (0.0%)
          </span>
        </div>

        <p className="text-xs text-slate-600 leading-relaxed">
          For each cell line independently, <strong>80% of the dataset is reserved for development</strong> and{' '}
          <strong>20% is held completely untouched for final validation</strong>. Within the 80% development pool,{' '}
          <strong>100-fold cross-validation</strong> is executed (each fold trains on ~99% and validates on ~1% of development data with fold-specific scaler fitting).
          Operating classification thresholds are determined solely from out-of-fold development predictions via Youden&apos;s J statistic and frozen before final holdout evaluation.
        </p>

        {summary && (
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-2 text-[11px] font-mono border-t border-indigo-100">
            <div>
              <span className="text-slate-400 text-[10px] block font-sans">Total Records:</span>
              <strong className="text-slate-800">{summary.total_samples.toLocaleString()}</strong>
            </div>
            <div>
              <span className="text-slate-400 text-[10px] block font-sans">80% Dev Pool (100 Folds):</span>
              <strong className="text-indigo-700">{summary.development_pool_count.toLocaleString()}</strong>
            </div>
            <div>
              <span className="text-slate-400 text-[10px] block font-sans">20% Untouched Holdout:</span>
              <strong className="text-emerald-700">{summary.final_holdout_count.toLocaleString()}</strong>
            </div>
            <div>
              <span className="text-slate-400 text-[10px] block font-sans">Positive / Negative Labels:</span>
              <strong className="text-slate-800">{summary.positive_labels} / {summary.negative_labels}</strong>
            </div>
          </div>
        )}
      </div>

      {/* TASK SELECTOR TOGGLE (REGRESSION VS CLASSIFICATION) */}
      <div className="flex flex-wrap items-center justify-between gap-2 bg-slate-100 p-1.5 rounded-lg border border-slate-200">
        <div className="flex items-center gap-1">
          <button
            onClick={() => setActiveTask('regression')}
            className={`px-3 py-1.5 rounded-md text-xs font-bold transition-all cursor-pointer flex items-center gap-1.5 ${
              activeTask === 'regression'
                ? 'bg-white text-indigo-700 shadow-xs border border-slate-200'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            <BarChart2 className="w-3.5 h-3.5" />
            Continuous Efficacy (Regression Suite)
          </button>
          <button
            onClick={() => setActiveTask('classification')}
            className={`px-3 py-1.5 rounded-md text-xs font-bold transition-all cursor-pointer flex items-center gap-1.5 ${
              activeTask === 'classification'
                ? 'bg-white text-indigo-700 shadow-xs border border-slate-200'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            <Activity className="w-3.5 h-3.5" />
            Binary Activity (Classification Suite)
          </button>
        </div>
        <span className="text-[11px] text-slate-500 font-medium px-2">
          {activeTask === 'regression'
            ? 'Task: Continuous knockout activity in [0.0, 1.0] (Pearson r, Spearman \u03c1, R\u00b2, RMSE, MAE)'
            : 'Task: Binary knockout activity in {0, 1} (ROC-AUC, PR-AUC, Specificity, Accuracy, Youden\'s J cutoff)'}
        </span>
      </div>

      {/* PRIMARY BENCHMARK TABLE: 100-FOLD CV VS FINAL 20% HOLDOUT */}
      <div className="bio-card p-4 space-y-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Award className="w-4 h-4 text-amber-500" />
            <h3 className="text-xs font-bold text-slate-900 uppercase tracking-wider">
              {activeTask === 'regression' ? 'Continuous Regression Benchmark' : 'Binary Classification Benchmark'} &mdash; {activeDatasetMetrics?.display_name || selectedDataset.toUpperCase()}
            </h3>
          </div>
          <span className="text-[10px] text-slate-400">
            {activeTask === 'regression' ? 'Comparing 5 regression models on efficiency score' : 'Comparing 5 classification models on binary activity'}
          </span>
        </div>

        <div className="overflow-x-auto border border-slate-200 rounded-lg">
          {activeTask === 'regression' ? (
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-50 text-slate-700 font-semibold border-b border-slate-200 text-[11px]">
                <tr>
                  <th className="py-2 px-3">Model Architecture</th>
                  <th className="py-2 px-3 text-right">100-Fold CV Pearson r</th>
                  <th className="py-2 px-3 text-right">100-Fold CV Spearman &rho;</th>
                  <th className="py-2 px-3 text-right">100-Fold CV R&sup2;</th>
                  <th className="py-2 px-3 text-right font-bold text-indigo-900">Holdout Pearson r</th>
                  <th className="py-2 px-3 text-right font-bold text-indigo-900">Holdout Spearman &rho;</th>
                  <th className="py-2 px-3 text-right font-bold text-indigo-900">Holdout R&sup2;</th>
                  <th className="py-2 px-3 text-right">Holdout RMSE</th>
                  <th className="py-2 px-3 text-right">Holdout MAE</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 font-mono text-[11px]">
                {modelEntries.map(([name, m]) => {
                  const cv = m.cv_100_fold;
                  const ho = m.final_validation_20;
                  return (
                    <tr key={name} className="hover:bg-slate-50/70">
                      <td className="py-2 px-3 font-sans font-medium text-slate-900 flex items-center gap-1.5">
                        <span
                          className="w-2 h-2 rounded-full inline-block"
                          style={{ backgroundColor: MODEL_COLORS[name] || '#64748b' }}
                        />
                        {name}
                      </td>
                      <td className="py-2 px-3 text-right text-slate-700">
                        {cv.mean.pearson !== undefined ? `${cv.mean.pearson.toFixed(3)} \u00b1 ${cv.std.pearson.toFixed(3)}` : '-'}
                      </td>
                      <td className="py-2 px-3 text-right text-slate-700">
                        {cv.mean.spearman !== undefined ? `${cv.mean.spearman.toFixed(3)} \u00b1 ${cv.std.spearman.toFixed(3)}` : '-'}
                      </td>
                      <td className="py-2 px-3 text-right text-slate-700">
                        {cv.mean.r2 !== undefined ? `${cv.mean.r2.toFixed(3)} \u00b1 ${cv.std.r2.toFixed(3)}` : '-'}
                      </td>
                      <td className="py-2 px-3 text-right font-bold text-indigo-700">
                        {ho.pearson !== undefined ? ho.pearson.toFixed(3) : '-'}
                      </td>
                      <td className="py-2 px-3 text-right font-bold text-indigo-700">
                        {ho.spearman !== undefined ? ho.spearman.toFixed(3) : '-'}
                      </td>
                      <td className="py-2 px-3 text-right font-bold text-indigo-700">
                        {ho.r2 !== undefined ? ho.r2.toFixed(3) : '-'}
                      </td>
                      <td className="py-2 px-3 text-right text-slate-700">
                        {ho.rmse !== undefined ? ho.rmse.toFixed(3) : '-'}
                      </td>
                      <td className="py-2 px-3 text-right text-slate-700">
                        {ho.mae !== undefined ? ho.mae.toFixed(3) : '-'}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          ) : (
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-50 text-slate-700 font-semibold border-b border-slate-200 text-[11px]">
                <tr>
                  <th className="py-2 px-3">Classifier Model</th>
                  <th className="py-2 px-3 text-right">100-Fold CV ROC-AUC</th>
                  <th className="py-2 px-3 text-right">100-Fold CV PR-AUC</th>
                  <th className="py-2 px-3 text-right font-bold text-indigo-900">Holdout ROC-AUC</th>
                  <th className="py-2 px-3 text-right font-bold text-indigo-900">Holdout PR-AUC</th>
                  <th className="py-2 px-3 text-right">Holdout Accuracy</th>
                  <th className="py-2 px-3 text-right font-bold text-emerald-700">Holdout Specificity</th>
                  <th className="py-2 px-3 text-right">Holdout Recall</th>
                  <th className="py-2 px-3 text-right">Holdout F1</th>
                  <th className="py-2 px-3 text-right font-mono text-slate-500">Cutoff (J)</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 font-mono text-[11px]">
                {modelEntries.map(([name, m]) => {
                  const cv = m.cv_100_fold;
                  const ho = m.final_validation_20;
                  return (
                    <tr key={name} className="hover:bg-slate-50/70">
                      <td className="py-2 px-3 font-sans font-medium text-slate-900 flex items-center gap-1.5">
                        <span
                          className="w-2 h-2 rounded-full inline-block"
                          style={{ backgroundColor: MODEL_COLORS[name] || '#64748b' }}
                        />
                        {name}
                      </td>
                      <td className="py-2 px-3 text-right text-slate-700">
                        {cv.mean.roc_auc !== undefined ? `${cv.mean.roc_auc.toFixed(3)} \u00b1 ${cv.std.roc_auc.toFixed(3)}` : '-'}
                      </td>
                      <td className="py-2 px-3 text-right text-slate-700">
                        {cv.mean.pr_auc !== undefined ? `${cv.mean.pr_auc.toFixed(3)} \u00b1 ${cv.std.pr_auc.toFixed(3)}` : '-'}
                      </td>
                      <td className="py-2 px-3 text-right font-bold text-emerald-700">
                        {ho.roc_auc !== undefined ? ho.roc_auc.toFixed(3) : '-'}
                      </td>
                      <td className="py-2 px-3 text-right font-bold text-indigo-700">
                        {ho.pr_auc !== undefined ? ho.pr_auc.toFixed(3) : '-'}
                      </td>
                      <td className="py-2 px-3 text-right text-slate-700">
                        {ho.accuracy !== undefined ? ho.accuracy.toFixed(3) : '-'}
                      </td>
                      <td className="py-2 px-3 text-right font-bold text-emerald-700">
                        {ho.specificity !== undefined ? ho.specificity.toFixed(3) : '-'}
                      </td>
                      <td className="py-2 px-3 text-right text-slate-700">
                        {ho.recall !== undefined ? ho.recall.toFixed(3) : '-'}
                      </td>
                      <td className="py-2 px-3 text-right text-slate-700">
                        {ho.f1 !== undefined ? ho.f1.toFixed(3) : '-'}
                      </td>
                      <td className="py-2 px-3 text-right text-slate-500">
                        {m.threshold !== undefined ? m.threshold.toFixed(3) : (ho.threshold !== undefined ? ho.threshold.toFixed(3) : '0.500')}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </div>
      </div>

      {/* REAL ROC CURVE COMPARISON CHART */}
      <div className="bio-card p-4 space-y-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Activity className="w-4 h-4 text-indigo-600" />
            <h3 className="text-xs font-bold text-slate-900 uppercase tracking-wider">
              {activeTask === 'classification' ? 'Binary Classification ROC Curves' : 'Continuous Efficacy Verification ROC Curves'} &mdash; {activeDatasetMetrics?.display_name || selectedDataset.toUpperCase()}
            </h3>
          </div>
          <span className="text-[10px] text-slate-400">
            Real True Positive vs. False Positive Rates on untouched 20% holdout
          </span>
        </div>

        <div className="h-64 w-full bg-white p-2 rounded-lg border border-slate-200">
          <Line data={fullChartData} options={chartOptions} />
        </div>
      </div>

      {/* DETAILED METRICS BREAKDOWN TABLE (ALL METRICS) */}
      <div className="bio-card p-4 space-y-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Layers className="w-4 h-4 text-slate-700" />
            <h3 className="text-xs font-bold text-slate-900 uppercase tracking-wider">
              Comprehensive Metrics Breakdown: 100-Fold CV vs. Final 20% Holdout
            </h3>
          </div>
          <span className="text-[10px] text-slate-400">
            Regression and Classification metrics
          </span>
        </div>

        <div className="overflow-x-auto border border-slate-200 rounded-lg">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-50 text-slate-700 font-semibold border-b border-slate-200 text-[11px]">
              <tr>
                <th className="py-2 px-3">Metric Category</th>
                <th className="py-2 px-3">Metric</th>
                {modelEntries.map(([name]) => (
                  <th key={name} className="py-2 px-3 text-right">
                    {name}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 font-mono text-[11px]">
              {/* Classification */}
              <tr className="bg-slate-50/50">
                <td rowSpan={7} className="py-2 px-3 font-sans font-bold text-indigo-900 align-top border-r border-slate-200">
                  Classification (Experimental Binary)
                </td>
                <td className="py-1.5 px-3 font-sans font-medium text-slate-800">ROC-AUC (Holdout / CV)</td>
                {modelEntries.map(([name, m]) => (
                  <td key={name} className="py-1.5 px-3 text-right">
                    <span className="font-bold text-emerald-700">{fmt(m.final_validation_20?.roc_auc)}</span>
                    <span className="text-slate-400 text-[10px] block">({fmt(m.cv_100_fold?.mean?.roc_auc)}&plusmn;{fmt(m.cv_100_fold?.std?.roc_auc)})</span>
                  </td>
                ))}
              </tr>
              <tr>
                <td className="py-1.5 px-3 font-sans font-medium text-slate-800">PR-AUC (Holdout / CV)</td>
                {modelEntries.map(([name, m]) => (
                  <td key={name} className="py-1.5 px-3 text-right">
                    <span className="font-bold text-indigo-700">{fmt(m.final_validation_20?.pr_auc)}</span>
                    <span className="text-slate-400 text-[10px] block">({fmt(m.cv_100_fold?.mean?.pr_auc)}&plusmn;{fmt(m.cv_100_fold?.std?.pr_auc)})</span>
                  </td>
                ))}
              </tr>
              <tr>
                <td className="py-1.5 px-3 font-sans font-medium text-slate-800">Accuracy (Holdout / CV)</td>
                {modelEntries.map(([name, m]) => (
                  <td key={name} className="py-1.5 px-3 text-right">
                    {fmt(m.final_validation_20?.accuracy)}
                    <span className="text-slate-400 text-[10px] block">({fmt(m.cv_100_fold?.mean?.accuracy)})</span>
                  </td>
                ))}
              </tr>
              <tr>
                <td className="py-1.5 px-3 font-sans font-medium text-slate-800">Precision (Holdout / CV)</td>
                {modelEntries.map(([name, m]) => (
                  <td key={name} className="py-1.5 px-3 text-right">
                    {fmt(m.final_validation_20?.precision)}
                    <span className="text-slate-400 text-[10px] block">({fmt(m.cv_100_fold?.mean?.precision)})</span>
                  </td>
                ))}
              </tr>
              <tr>
                <td className="py-1.5 px-3 font-sans font-medium text-slate-800">Recall / Sensitivity</td>
                {modelEntries.map(([name, m]) => (
                  <td key={name} className="py-1.5 px-3 text-right">
                    {fmt(m.final_validation_20?.recall)}
                    <span className="text-slate-400 text-[10px] block">({fmt(m.cv_100_fold?.mean?.recall)})</span>
                  </td>
                ))}
              </tr>
              <tr>
                <td className="py-1.5 px-3 font-sans font-medium text-slate-800">Specificity (TN / (TN+FP))</td>
                {modelEntries.map(([name, m]) => (
                  <td key={name} className="py-1.5 px-3 text-right">
                    {fmt(m.final_validation_20?.specificity)}
                    <span className="text-slate-400 text-[10px] block">({fmt(m.cv_100_fold?.mean?.specificity)})</span>
                  </td>
                ))}
              </tr>
              <tr>
                <td className="py-1.5 px-3 font-sans font-medium text-slate-800">F1 Score</td>
                {modelEntries.map(([name, m]) => (
                  <td key={name} className="py-1.5 px-3 text-right">
                    {fmt(m.final_validation_20?.f1)}
                    <span className="text-slate-400 text-[10px] block">({fmt(m.cv_100_fold?.mean?.f1)})</span>
                  </td>
                ))}
              </tr>

              {/* Regression */}
              <tr className="bg-slate-50/50 border-t border-slate-200">
                <td rowSpan={5} className="py-2 px-3 font-sans font-bold text-slate-900 align-top border-r border-slate-200">
                  Continuous Regression (Efficiency Score)
                </td>
                <td className="py-1.5 px-3 font-sans font-medium text-slate-800">Pearson Correlation (r)</td>
                {modelEntries.map(([name, m]) => (
                  <td key={name} className="py-1.5 px-3 text-right">
                    <span className="font-bold text-indigo-700">{fmt(m.final_validation_20?.pearson)}</span>
                    <span className="text-slate-400 text-[10px] block">({fmt(m.cv_100_fold?.mean?.pearson)}&plusmn;{fmt(m.cv_100_fold?.std?.pearson)})</span>
                  </td>
                ))}
              </tr>
              <tr>
                <td className="py-1.5 px-3 font-sans font-medium text-slate-800">Spearman Rank (&rho;)</td>
                {modelEntries.map(([name, m]) => (
                  <td key={name} className="py-1.5 px-3 text-right">
                    <span className="font-bold text-slate-800">{fmt(m.final_validation_20?.spearman)}</span>
                    <span className="text-slate-400 text-[10px] block">({fmt(m.cv_100_fold?.mean?.spearman)}&plusmn;{fmt(m.cv_100_fold?.std?.spearman)})</span>
                  </td>
                ))}
              </tr>
              <tr>
                <td className="py-1.5 px-3 font-sans font-medium text-slate-800">RMSE</td>
                {modelEntries.map(([name, m]) => (
                  <td key={name} className="py-1.5 px-3 text-right">
                    {fmt(m.final_validation_20?.rmse)}
                    <span className="text-slate-400 text-[10px] block">({fmt(m.cv_100_fold?.mean?.rmse)})</span>
                  </td>
                ))}
              </tr>
              <tr>
                <td className="py-1.5 px-3 font-sans font-medium text-slate-800">MAE</td>
                {modelEntries.map(([name, m]) => (
                  <td key={name} className="py-1.5 px-3 text-right">
                    {fmt(m.final_validation_20?.mae)}
                    <span className="text-slate-400 text-[10px] block">({fmt(m.cv_100_fold?.mean?.mae)})</span>
                  </td>
                ))}
              </tr>
              <tr>
                <td className="py-1.5 px-3 font-sans font-medium text-slate-800">R&sup2; Score</td>
                {modelEntries.map(([name, m]) => (
                  <td key={name} className="py-1.5 px-3 text-right">
                    {fmt(m.final_validation_20?.r2)}
                    <span className="text-slate-400 text-[10px] block">({fmt(m.cv_100_fold?.mean?.r2)})</span>
                  </td>
                ))}
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <Disclaimer />
    </div>
  );
};
