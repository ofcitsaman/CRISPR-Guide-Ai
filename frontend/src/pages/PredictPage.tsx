import React, { useState } from 'react';
import { predictSingleGuide } from '../services/api';
import { PredictionResult } from '../types';
import { SequenceVisualizer } from '../components/SequenceVisualizer';
import { Disclaimer } from '../components/Disclaimer';
import {
  Play,
  RotateCcw,
  CheckCircle2,
  AlertTriangle,
  ChevronDown,
  ChevronUp,
  HelpCircle,
  Database,
  Activity,
  BarChart3,
  ShieldCheck,
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
  Filler,
} from 'chart.js';
import { Line } from 'react-chartjs-2';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

const DATASET_OPTIONS = [
  { id: 'HeLa', name: 'HeLa (8,101 guides, Cervical)', desc: 'Viability dropout screen (Hart 2015)' },
  { id: 'HCT116', name: 'HCT116 (4,239 guides, Colorectal)', desc: 'Viability dropout screen (Hart 2015)' },
  { id: 'A375', name: 'A375 (2,333 guides, Melanoma)', desc: 'Drug resistance tiling screen (Doench 2016)' },
  { id: 'HL60', name: 'HL60 (2,076 guides, Leukemia)', desc: 'Viability dropout screen (Xu 2015)' },
];

const MODEL_OPTIONS = [
  { id: 'Weighted Ensemble', name: 'Weighted Ensemble (Recommended)', desc: 'Optimal OOF-weighted ensemble of Ridge, RF, XGBoost & SVR' },
  { id: 'XGBoost', name: 'XGBoost', desc: 'Gradient boosted trees (top rank correlation)' },
  { id: 'Random Forest', name: 'Random Forest', desc: '100 bagged trees with empirical dispersion' },
  { id: 'SVM', name: 'SVM (Support Vector Regression)', desc: 'Tuned RBF kernel SVR' },
  { id: 'Ridge Linear Regression', name: 'Ridge Linear Regression', desc: 'Analytical regularized linear baseline' },
];

const PRESET_GUIDES = [
  { name: 'HeLa Screen Top Guide', seq: 'CTTGCTCGCGCAGGACGAGGCGG', dataset: 'HeLa' },
  { name: 'HCT116 EMX1 Exon 3', seq: 'GAGTCCGAGCAGAAGAAGAAGGG', dataset: 'HCT116' },
  { name: 'A375 Benchmark (Doench 2016)', seq: 'GGTGAGTGAGTGTGTGCGTGTGG', dataset: 'A375' },
  { name: 'HL60 Poly-T Terminator', seq: 'GACCTGATCTTTTGATCGTATGG', dataset: 'HL60' },
];

export const PredictPage: React.FC = () => {
  const [sequence, setSequence] = useState('CTTGCTCGCGCAGGACGAGGCGG');
  const [pamInput, setPamInput] = useState('NGG');
  const [datasetName, setDatasetName] = useState('HeLa');
  const [modelName, setModelName] = useState('Weighted Ensemble');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showTechnicalDetails, setShowTechnicalDetails] = useState(false);

  // Live sequence validation
  const cleanSeq = sequence.trim().toUpperCase();
  const invalidChars = Array.from(new Set(cleanSeq.replace(/[ACGT]/g, '')));
  const isLengthValid = cleanSeq.length === 20 || cleanSeq.length === 23;
  const isCharactersValid = invalidChars.length === 0 && cleanSeq.length > 0;
  const canSubmit = isLengthValid && isCharactersValid && !loading;

  const handlePredict = async (seqToRun?: string, modelToRun?: string, datasetToRun?: string) => {
    const s = seqToRun || sequence;
    const m = modelToRun || modelName;
    const d = datasetToRun || datasetName;
    setError(null);
    setLoading(true);

    try {
      const data = await predictSingleGuide(s, m, d);
      setResult(data);
    } catch (err: any) {
      setError(err.message || 'Failed to generate prediction');
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setSequence('');
    setResult(null);
    setError(null);
  };

  // Prepare real ROC Curve chart data from verification metrics
  const rocData = result?.verification_metrics?.roc_curve;
  const chartData = rocData && rocData.fpr && rocData.tpr ? {
    labels: rocData.fpr.map((v) => (v * 100).toFixed(0) + '%'),
    datasets: [
      {
        label: `${result.model_used} on ${result.dataset_used} (Holdout ROC-AUC = ${rocData.auc.toFixed(3)})`,
        data: rocData.tpr,
        borderColor: '#4f46e5',
        backgroundColor: 'rgba(79, 70, 229, 0.08)',
        fill: true,
        tension: 0.1,
        pointRadius: 0,
        pointHitRadius: 6,
        borderWidth: 2,
      },
      {
        label: 'Random Chance Baseline (AUC = 0.500)',
        data: rocData.fpr,
        borderColor: '#94a3b8',
        borderDash: [4, 4],
        pointRadius: 0,
        fill: false,
        borderWidth: 1.5,
      },
    ],
  } : null;

  const chartOptions: any = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'bottom' as const,
        labels: { boxWidth: 12, font: { size: 10 } },
      },
      tooltip: {
        callbacks: {
          title: (items: any) => `FPR: ${items[0].label}`,
          label: (item: any) => `${item.dataset.label.split('(')[0]}: TPR = ${(item.raw * 100).toFixed(1)}%`,
        },
      },
    },
    scales: {
      x: {
        title: { display: true, text: 'False Positive Rate (FPR)', font: { size: 10 } },
        ticks: { maxTicksLimit: 6, font: { size: 9 } },
        grid: { color: '#f1f5f9' },
      },
      y: {
        min: 0,
        max: 1,
        title: { display: true, text: 'True Positive Rate (TPR / Sensitivity)', font: { size: 10 } },
        ticks: { font: { size: 9 } },
        grid: { color: '#f1f5f9' },
      },
    },
  };

  const vm = result?.verification_metrics;
  const cvM = vm?.cv_100_fold?.mean;
  const cvStd = vm?.cv_100_fold?.std;
  const hoM = vm?.final_holdout_validation;

  return (
    <div className="space-y-5 max-w-5xl mx-auto">
      {/* 1-Sentence Header */}
      <div className="border-b border-slate-200 pb-2.5">
        <h1 className="text-lg font-bold text-slate-900 tracking-tight">CRISPR-Cas9 sgRNA Efficiency Prediction</h1>
        <p className="text-xs text-slate-500 mt-0.5">
          Predict on-target knockout efficacy using dataset-specific models evaluated under a strict 80/20 holdout + 100-fold cross-validation framework.
        </p>
      </div>

      {/* Main Container */}
      <div className="grid grid-cols-1 md:grid-cols-12 gap-5 items-start">
        {/* Left Column: Inputs */}
        <div className="md:col-span-5 space-y-3">
          <div className="bio-card p-4 space-y-3">
            {/* Dataset Selector */}
            <div>
              <label className="text-[11px] font-bold text-slate-900 uppercase tracking-wider flex items-center gap-1 mb-1">
                <Database className="w-3.5 h-3.5 text-indigo-600" />
                Target Cell-Line Dataset
              </label>
              <select
                value={datasetName}
                onChange={(e) => {
                  setDatasetName(e.target.value);
                  if (result) handlePredict(sequence, modelName, e.target.value);
                }}
                className="w-full p-2 bg-slate-50 border border-slate-300 rounded-lg text-xs font-semibold text-slate-800 focus:ring-2 focus:ring-indigo-100 focus:border-indigo-500"
              >
                {DATASET_OPTIONS.map((d) => (
                  <option key={d.id} value={d.id}>
                    {d.name}
                  </option>
                ))}
              </select>
              <span className="text-[10px] text-slate-400 block mt-0.5">
                Models are independently trained on each cell line to prevent cross-assay confounding.
              </span>
            </div>

            {/* Sequence Input */}
            <div className="space-y-1.5 pt-1">
              <div className="flex items-center justify-between">
                <label className="text-xs font-bold text-slate-900 uppercase tracking-wider">
                  sgRNA Sequence
                </label>
                <span className="text-[11px] font-mono text-slate-400">{cleanSeq.length} nt</span>
              </div>

              <textarea
                value={sequence}
                onChange={(e) => setSequence(e.target.value.toUpperCase())}
                placeholder="Enter 20-nt spacer or 23-nt sequence (e.g. CTTGCTCGCGCAGGACGAGGCGG)..."
                rows={3}
                className={`w-full p-2.5 font-mono text-xs tracking-wider uppercase rounded-lg border focus:ring-2 focus:outline-hidden transition-colors ${
                  !isCharactersValid && cleanSeq.length > 0
                    ? 'border-rose-300 bg-rose-50/20'
                    : 'border-slate-300 focus:border-indigo-500 focus:ring-indigo-100 bg-white'
                }`}
              />

              {/* Validation Feedback */}
              <div className="flex flex-wrap items-center gap-1 text-xs">
                {cleanSeq.length === 20 && (
                  <span className="text-sky-700 bg-sky-50 px-2 py-0.5 rounded text-[11px] font-mono border border-sky-200">
                    20-nt spacer (Canonical TGG PAM will be appended)
                  </span>
                )}
                {cleanSeq.length === 23 && (
                  <span className="text-emerald-700 bg-emerald-50 px-2 py-0.5 rounded text-[11px] font-mono border border-emerald-200 flex items-center gap-1">
                    <CheckCircle2 className="w-3 h-3" /> 23-nt valid sequence
                  </span>
                )}
                {cleanSeq.length > 0 && !isLengthValid && (
                  <span className="text-amber-700 bg-amber-50 px-2 py-0.5 rounded text-[11px] border border-amber-200 flex items-center gap-1">
                    <AlertTriangle className="w-3 h-3" /> {cleanSeq.length} nt (Expected 20 or 23 nt)
                  </span>
                )}
                {invalidChars.length > 0 && (
                  <span className="text-rose-700 bg-rose-50 px-2 py-0.5 rounded text-[11px] border border-rose-200">
                    Invalid: {invalidChars.join(', ')} (Must be A, C, G, T)
                  </span>
                )}
              </div>
            </div>

            {/* Model & PAM Row */}
            <div className="grid grid-cols-2 gap-2 pt-0.5">
              <div>
                <label className="text-[11px] font-semibold text-slate-700 block mb-1">Model Architecture</label>
                <select
                  value={modelName}
                  onChange={(e) => {
                    setModelName(e.target.value);
                    if (result) handlePredict(sequence, e.target.value, datasetName);
                  }}
                  className="w-full p-1.5 bg-slate-50 border border-slate-300 rounded text-xs font-medium text-slate-800"
                >
                  {MODEL_OPTIONS.map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.name}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="text-[11px] font-semibold text-slate-700 block mb-1">Target PAM</label>
                <input
                  type="text"
                  value={pamInput}
                  onChange={(e) => setPamInput(e.target.value.toUpperCase())}
                  disabled={cleanSeq.length === 23}
                  className="w-full p-1.5 bg-slate-50 border border-slate-300 rounded text-xs font-mono font-medium text-slate-700 disabled:opacity-60"
                  placeholder="NGG"
                />
              </div>
            </div>

            {/* Action Buttons */}
            <div className="flex items-center gap-2 pt-1">
              <button
                onClick={() => handlePredict()}
                disabled={!canSubmit}
                className={`flex-1 inline-flex items-center justify-center gap-1.5 px-4 py-2 rounded-lg text-xs font-bold text-white shadow-xs transition-all cursor-pointer ${
                  canSubmit ? 'bg-indigo-600 hover:bg-indigo-700' : 'bg-slate-300 cursor-not-allowed text-slate-500'
                }`}
              >
                <Play className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
                {loading ? 'Evaluating...' : 'Predict Relative Efficacy'}
              </button>

              <button
                onClick={handleClear}
                className="px-2.5 py-2 rounded-lg border border-slate-300 bg-white hover:bg-slate-50 text-slate-600 text-xs transition-colors cursor-pointer"
                title="Clear sequence"
              >
                <RotateCcw className="w-3.5 h-3.5" />
              </button>
            </div>

            {error && (
              <div className="p-2 rounded bg-rose-50 border border-rose-200 text-rose-700 text-xs">
                {error}
              </div>
            )}

            {/* Benchmark Presets */}
            <div className="pt-2 border-t border-slate-100">
              <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider block mb-1">
                Benchmark Presets:
              </span>
              <div className="grid grid-cols-2 gap-1 text-[11px]">
                {PRESET_GUIDES.map((preset, idx) => (
                  <button
                    key={idx}
                    onClick={() => {
                      setSequence(preset.seq);
                      setDatasetName(preset.dataset);
                      handlePredict(preset.seq, modelName, preset.dataset);
                    }}
                    className="p-1 text-left rounded border border-slate-200 hover:border-indigo-300 hover:bg-indigo-50/30 text-slate-700 truncate text-[11px] cursor-pointer"
                    title={`${preset.name}: ${preset.seq} (${preset.dataset})`}
                  >
                    <span className="font-semibold">{preset.dataset}</span>: {preset.name.split(' ')[1]}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Right Column: Prediction Results & Model Verification */}
        <div className="md:col-span-7 space-y-3">
          {result ? (
            <div className="space-y-3">
              {/* Main Score Box */}
              <div className="bio-card p-4 space-y-3 border-slate-300">
                <div className="flex flex-wrap items-baseline justify-between gap-2">
                  <div>
                    <div className="flex items-center gap-1">
                      <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
                        Relative Efficacy Score
                      </span>
                      <span
                        className="text-slate-400 cursor-help"
                        title="Normalized score learned from experimentally measured sgRNA activity in this cell line. It is not a probability, percentile, or guaranteed editing percentage."
                      >
                        <HelpCircle className="w-3.5 h-3.5" />
                      </span>
                    </div>
                    <div className="flex items-baseline gap-1.5 mt-0.5">
                      <span className="text-3xl font-extrabold text-slate-900 font-mono tracking-tight">
                        {result.relative_efficacy_score.toFixed(1)}
                      </span>
                      <span className="text-xs text-slate-500 font-medium">/ 100</span>
                    </div>
                  </div>

                  <div className="flex flex-col items-end gap-1">
                    <span
                      className={`px-2.5 py-0.5 rounded-full text-xs font-bold border ${
                        result.activity_classification.includes('High')
                          ? 'bg-emerald-50 text-emerald-700 border-emerald-200'
                          : result.activity_classification.includes('Moderate')
                          ? 'bg-indigo-50 text-indigo-700 border-indigo-200'
                          : 'bg-slate-100 text-slate-700 border-slate-200'
                      }`}
                    >
                      {result.activity_classification}
                    </span>
                    <span className="text-[11px] font-mono text-slate-500">
                      {result.dataset_used} &bull; {result.model_used}
                    </span>
                  </div>
                </div>

                {/* Score Progress Bar */}
                <div className="w-full bg-slate-100 rounded-full h-2 overflow-hidden">
                  <div
                    className={`h-full transition-all duration-300 ${
                      result.relative_efficacy_score >= 50
                        ? 'bg-emerald-500'
                        : result.relative_efficacy_score >= 25
                        ? 'bg-indigo-500'
                        : 'bg-amber-500'
                    }`}
                    style={{ width: `${Math.min(100, Math.max(3, result.relative_efficacy_score))}%` }}
                  />
                </div>

                {/* Interpretation */}
                <div className="text-xs text-slate-700 bg-slate-50 rounded-lg p-2.5 border border-slate-200 leading-relaxed">
                  <p className="font-medium text-slate-900">
                    {result.relative_efficacy_score >= 50
                      ? `The model predicts high knockout activity on the ${result.dataset_used} screening scale.`
                      : result.relative_efficacy_score >= 25
                      ? `The model predicts moderate knockout activity relative to ${result.dataset_used} benchmarks.`
                      : `The model predicts lower relative knockout efficacy in ${result.dataset_used} screens.`}
                  </p>
                  <p className="text-[11px] text-slate-500 mt-0.5">
                    {result.score_explanation}
                  </p>
                </div>

                {/* Reliability & Ensemble Dispersion */}
                <div className="pt-1 flex flex-wrap items-center justify-between text-xs text-slate-600 gap-2 border-t border-slate-100">
                  <div>
                    <span className="text-slate-500">Dataset / Model: </span>
                    <span className="font-semibold text-slate-800">{result.dataset_used} &bull; {result.model_used}</span>
                  </div>

                  {result.ensemble_dispersion !== null ? (
                    <div className="flex items-center gap-1">
                      <span className="text-slate-500">Empirical Ensemble Dispersion: </span>
                      <span className="font-mono font-bold text-slate-900">
                        &plusmn;{(result.ensemble_dispersion * 100).toFixed(1)}
                      </span>
                      <span
                        className="text-slate-400 cursor-help"
                        title="Variation among Random Forest decision trees (±1.96σ). Not a formally calibrated Bayesian confidence interval."
                      >
                        <HelpCircle className="w-3.5 h-3.5" />
                      </span>
                    </div>
                  ) : (
                    <div className="text-[11px] text-slate-400 italic">
                      Empirical tree dispersion is available when selecting Random Forest.
                    </div>
                  )}
                </div>
              </div>

              {/* MODEL VERIFICATION SECTION */}
              <div className="bio-card p-4 space-y-3 border-indigo-200 bg-gradient-to-b from-indigo-50/20 to-white">
                <div className="flex items-center justify-between border-b border-indigo-100 pb-2">
                  <div className="flex items-center gap-1.5">
                    <ShieldCheck className="w-4 h-4 text-indigo-600" />
                    <h3 className="text-xs font-bold text-slate-900 uppercase tracking-wider">
                      Model Verification &mdash; {result.dataset_used} / {result.model_used}
                    </h3>
                  </div>
                  <span className="text-[10px] font-mono text-indigo-700 bg-indigo-50 px-2 py-0.5 rounded border border-indigo-100">
                    80% Dev (100-Fold CV) &bull; 20% Holdout
                  </span>
                </div>

                {/* Verification Metrics Table */}
                {cvM && hoM ? (
                  <div className="overflow-x-auto border border-slate-200 rounded-lg">
                    <table className="w-full text-left text-xs">
                      <thead className="bg-slate-50 text-slate-700 font-semibold border-b border-slate-200 text-[11px]">
                        <tr>
                          <th className="py-1.5 px-2.5">Evaluation Metric</th>
                          <th className="py-1.5 px-2.5 text-right">100-Fold CV (Mean &plusmn; SD)</th>
                          <th className="py-1.5 px-2.5 text-right font-bold text-indigo-900">Final 20% Holdout</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-100 font-mono text-[11px]">
                        <tr className="hover:bg-slate-50/60">
                          <td className="py-1.5 px-2.5 font-sans font-medium text-slate-800">ROC-AUC (Continuous)</td>
                          <td className="py-1.5 px-2.5 text-right">{cvM?.roc_auc !== undefined ? cvM.roc_auc.toFixed(3) : '—'} &plusmn; {cvStd?.roc_auc !== undefined ? cvStd.roc_auc.toFixed(3) : '0.000'}</td>
                          <td className="py-1.5 px-2.5 text-right font-bold text-indigo-700">{hoM?.roc_auc !== undefined ? hoM.roc_auc.toFixed(3) : '—'}</td>
                        </tr>
                        <tr className="hover:bg-slate-50/60">
                          <td className="py-1.5 px-2.5 font-sans font-medium text-slate-800">PR-AUC (Continuous)</td>
                          <td className="py-1.5 px-2.5 text-right">{cvM?.pr_auc !== undefined ? cvM.pr_auc.toFixed(3) : '—'} &plusmn; {cvStd?.pr_auc !== undefined ? cvStd.pr_auc.toFixed(3) : '0.000'}</td>
                          <td className="py-1.5 px-2.5 text-right font-bold text-indigo-700">{hoM?.pr_auc !== undefined ? hoM.pr_auc.toFixed(3) : '—'}</td>
                        </tr>
                        <tr className="hover:bg-slate-50/60">
                          <td className="py-1.5 px-2.5 font-sans font-medium text-slate-800">Accuracy (Binary)</td>
                          <td className="py-1.5 px-2.5 text-right">{cvM?.accuracy !== undefined ? cvM.accuracy.toFixed(3) : '—'} &plusmn; {cvStd?.accuracy !== undefined ? cvStd.accuracy.toFixed(3) : '0.000'}</td>
                          <td className="py-1.5 px-2.5 text-right font-bold text-slate-800">{hoM?.accuracy !== undefined ? hoM.accuracy.toFixed(3) : '—'}</td>
                        </tr>
                        <tr className="hover:bg-slate-50/60">
                          <td className="py-1.5 px-2.5 font-sans font-medium text-slate-800">Precision</td>
                          <td className="py-1.5 px-2.5 text-right">{cvM?.precision !== undefined ? cvM.precision.toFixed(3) : '—'} &plusmn; {cvStd?.precision !== undefined ? cvStd.precision.toFixed(3) : '0.000'}</td>
                          <td className="py-1.5 px-2.5 text-right font-bold text-slate-800">{hoM?.precision !== undefined ? hoM.precision.toFixed(3) : '—'}</td>
                        </tr>
                        <tr className="hover:bg-slate-50/60">
                          <td className="py-1.5 px-2.5 font-sans font-medium text-slate-800">Recall (Sensitivity)</td>
                          <td className="py-1.5 px-2.5 text-right">{cvM?.recall !== undefined ? cvM.recall.toFixed(3) : '—'} &plusmn; {cvStd?.recall !== undefined ? cvStd.recall.toFixed(3) : '0.000'}</td>
                          <td className="py-1.5 px-2.5 text-right font-bold text-slate-800">{hoM?.recall !== undefined ? hoM.recall.toFixed(3) : '—'}</td>
                        </tr>
                        <tr className="hover:bg-slate-50/60">
                          <td className="py-1.5 px-2.5 font-sans font-medium text-slate-800">Specificity (TN / (TN+FP))</td>
                          <td className="py-1.5 px-2.5 text-right">{cvM?.specificity !== undefined ? cvM.specificity.toFixed(3) : '—'} &plusmn; {cvStd?.specificity !== undefined ? cvStd.specificity.toFixed(3) : '0.000'}</td>
                          <td className="py-1.5 px-2.5 text-right font-bold text-slate-800">{hoM?.specificity !== undefined ? hoM.specificity.toFixed(3) : '—'}</td>
                        </tr>
                        <tr className="hover:bg-slate-50/60">
                          <td className="py-1.5 px-2.5 font-sans font-medium text-slate-800">Pearson Correlation (r)</td>
                          <td className="py-1.5 px-2.5 text-right">{cvM?.pearson !== undefined ? cvM.pearson.toFixed(3) : '—'} &plusmn; {cvStd?.pearson !== undefined ? cvStd.pearson.toFixed(3) : '0.000'}</td>
                          <td className="py-1.5 px-2.5 text-right font-bold text-slate-800">{hoM?.pearson !== undefined ? hoM.pearson.toFixed(3) : '—'}</td>
                        </tr>
                        <tr className="hover:bg-slate-50/60">
                          <td className="py-1.5 px-2.5 font-sans font-medium text-slate-800">RMSE</td>
                          <td className="py-1.5 px-2.5 text-right">{cvM?.rmse !== undefined ? cvM.rmse.toFixed(3) : '—'} &plusmn; {cvStd?.rmse !== undefined ? cvStd.rmse.toFixed(3) : '0.000'}</td>
                          <td className="py-1.5 px-2.5 text-right font-bold text-slate-800">{hoM?.rmse !== undefined ? hoM.rmse.toFixed(3) : '—'}</td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <div className="text-xs text-slate-500 py-2">Verification metrics loading...</div>
                )}

                {/* Important Distinction Note */}
                <p className="text-[10px] text-slate-500 italic">
                  Note: The verification table above reflects model performance across the independent 80% development pool (100-fold CV) and fixed historical 20% validation holdout, not metrics computed on the single query sequence.
                </p>

                {/* Real ROC Curve Graph */}
                {chartData && (
                  <div className="pt-2 border-t border-slate-100">
                    <div className="flex items-center justify-between mb-1.5">
                      <span className="text-[11px] font-bold text-slate-900 flex items-center gap-1">
                        <Activity className="w-3.5 h-3.5 text-indigo-600" />
                        Final 20% Holdout ROC Curve ({result.dataset_used})
                      </span>
                      <span className="text-[10px] font-mono text-emerald-700 bg-emerald-50 px-1.5 py-0.5 rounded border border-emerald-200">
                        ROC-AUC: {rocData?.auc.toFixed(3)}
                      </span>
                    </div>
                    <div className="h-44 w-full bg-white p-2 rounded-lg border border-slate-200">
                      <Line data={chartData} options={chartOptions} />
                    </div>
                  </div>
                )}
              </div>

              {/* Key Influential Sequence Features */}
              {result.top_contributing_features.length > 0 && (
                <div className="bio-card p-4 space-y-2.5">
                  <div className="flex items-center justify-between">
                    <h3 className="text-xs font-bold text-slate-900 uppercase tracking-wider">
                      Key Influential Sequence Features
                    </h3>
                    <span className="text-[10px] text-slate-400">Top model-associated features</span>
                  </div>

                  <div className="space-y-1">
                    {result.top_contributing_features.slice(0, 4).map((feat, idx) => (
                      <div
                        key={idx}
                        className="p-2 rounded border border-slate-100 bg-slate-50/70 flex items-center justify-between text-xs"
                      >
                        <div className="flex items-center gap-2">
                          <span className="font-mono font-semibold text-slate-800">{feat.feature}</span>
                          <span className="text-[11px] text-slate-500">
                            {feat.feature === 'pos_18_C' ? '• Cleavage site Cytosine' :
                             feat.feature === 'pos_20_G' ? '• PAM-adjacent Guanine' :
                             feat.feature === 'pos_20_T' || feat.feature === 'pos_18_T' ? '• PAM-proximal Thymine' :
                             feat.interpretation}
                          </span>
                        </div>
                        <span
                          className={`px-1.5 py-0.2 rounded text-[10px] font-semibold uppercase ${
                            feat.effect_direction === 'positive'
                              ? 'bg-emerald-50 text-emerald-700'
                              : 'bg-slate-100 text-slate-600'
                          }`}
                        >
                          {feat.effect_direction} influence
                        </span>
                      </div>
                    ))}
                  </div>

                  <p className="text-[11px] text-slate-500 italic pt-0.5">
                    This feature had strong influence on the model prediction. This model association is consistent with previously reported experimental sequence preferences.
                  </p>
                </div>
              )}

              {/* Expandable Technical Details */}
              <div className="bio-card border-slate-200">
                <button
                  onClick={() => setShowTechnicalDetails(!showTechnicalDetails)}
                  className="w-full p-3.5 flex items-center justify-between text-xs font-bold text-slate-800 hover:bg-slate-50 transition-colors cursor-pointer select-none"
                >
                  <span className="flex items-center gap-1.5">
                    <span>Technical Details & Sequence Architecture Map</span>
                    <span className="text-[10px] font-normal text-slate-400">
                      (PAM, Seed, Cleavage plane, GC composition)
                    </span>
                  </span>
                  {showTechnicalDetails ? <ChevronUp className="w-4 h-4 text-slate-500" /> : <ChevronDown className="w-4 h-4 text-slate-500" />}
                </button>

                {showTechnicalDetails && (
                  <div className="p-4 pt-0 border-t border-slate-100 space-y-3">
                    {/* Interactive 23-mer Visualizer */}
                    <div className="pt-2">
                      <SequenceVisualizer
                        sequence={result.sequence}
                        positionalWeights={result.positional_saliency}
                      />
                    </div>

                    {/* Biophysical Parameters Table */}
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs pt-1">
                      <div className="p-2 rounded bg-slate-50 border border-slate-200">
                        <div className="text-slate-400 text-[10px]">Overall GC</div>
                        <div className="font-mono font-bold text-slate-800 text-xs mt-0.5">
                          {result.biophysical_properties.gc_content_pct}%
                        </div>
                      </div>

                      <div className="p-2 rounded bg-slate-50 border border-slate-200">
                        <div className="text-slate-400 text-[10px]">Seed Region GC</div>
                        <div className="font-mono font-bold text-indigo-700 text-xs mt-0.5">
                          {result.biophysical_properties.seed_gc_pct}%
                        </div>
                      </div>

                      <div className="p-2 rounded bg-slate-50 border border-slate-200">
                        <div className="text-slate-400 text-[10px]">PAM Motif</div>
                        <div className="font-mono font-bold text-amber-700 text-xs mt-0.5">
                          {result.biophysical_properties.pam_motif}
                        </div>
                      </div>

                      <div className="p-2 rounded bg-slate-50 border border-slate-200">
                        <div className="text-slate-400 text-[10px]">Poly-T Run</div>
                        <div className="font-mono font-bold text-slate-800 text-xs mt-0.5">
                          {result.biophysical_properties.poly_t_max_run} Ts
                        </div>
                      </div>
                    </div>

                    {result.biophysical_properties.warnings.length > 0 && (
                      <div className="p-2 rounded bg-amber-50 border border-amber-200 text-amber-800 text-xs space-y-0.5">
                        {result.biophysical_properties.warnings.map((w, idx) => (
                          <div key={idx} className="flex items-center gap-1.5">
                            <AlertTriangle className="w-3.5 h-3.5 shrink-0 text-amber-600" />
                            <span>{w}</span>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          ) : (
            <div className="bio-card p-8 text-center space-y-2 text-slate-500">
              <Play className="w-6 h-6 text-slate-300 mx-auto" />
              <div className="text-xs font-semibold text-slate-700">Ready for Sequence Input</div>
              <p className="text-[11px] text-slate-400 max-w-sm mx-auto">
                Select a target cell line, choose a predictive model, enter an sgRNA sequence, and click "Predict Relative Efficacy" to view efficacy and verification metrics.
              </p>
            </div>
          )}

          <Disclaimer />
        </div>
      </div>
    </div>
  );
};
