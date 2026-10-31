import React, { useState } from 'react';
import { predictBatchGuides, uploadCsvGuides } from '../services/api';
import { BatchResultItem } from '../types';
import { Disclaimer } from '../components/Disclaimer';
import {
  Upload,
  Play,
  Download,
  ArrowUpDown,
  Database,
  Cpu,
} from 'lucide-react';

const DATASET_OPTIONS = [
  { id: 'HeLa', name: 'HeLa (Cervical)' },
  { id: 'HCT116', name: 'HCT116 (Colorectal)' },
  { id: 'A375', name: 'A375 (Melanoma - Doench 2016)' },
  { id: 'HL60', name: 'HL60 (Leukemia)' },
];

const MODEL_OPTIONS = [
  { id: 'Weighted Ensemble', name: 'Weighted Ensemble (Recommended)' },
  { id: 'XGBoost', name: 'XGBoost' },
  { id: 'Random Forest', name: 'Random Forest' },
  { id: 'SVM', name: 'SVM (Support Vector Regression)' },
  { id: 'Ridge Linear Regression', name: 'Ridge Linear Regression' },
];

export const BatchPredictPage: React.FC = () => {
  const [inputText, setInputText] = useState(
    'CTTGCTCGCGCAGGACGAGGCGG\nGAGTCCGAGCAGAAGAAGAAGGG\nGGTGAGTGAGTGTGTGCGTGTGG\nGACCTGATCTTTTGATCGTATGG\nACATCAGGTTACCTCTACCAAGG'
  );
  const [datasetName, setDatasetName] = useState('HeLa');
  const [modelName, setModelName] = useState('Weighted Ensemble');
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<BatchResultItem[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [filterActiveOnly, setFilterActiveOnly] = useState(false);
  const [sortAscending, setSortAscending] = useState(false);

  const handlePredictText = async () => {
    setError(null);
    const lines = inputText
      .split('\n')
      .map((l) => l.trim().toUpperCase())
      .filter((l) => l.length > 0);

    if (lines.length === 0) {
      setError('Please provide at least one sequence to predict.');
      return;
    }

    if (lines.length > 500) {
      setError('Batch size limited to 500 sequences per submission in the web UI.');
      return;
    }

    const guides = lines.map((seq, i) => ({ id: `gRNA_${i + 1}`, sequence: seq }));
    setLoading(true);

    try {
      const resp = await predictBatchGuides(guides, modelName, datasetName);
      setResults(resp.results);
    } catch (err: any) {
      setError(err.message || 'Batch prediction failed');
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setError(null);
    setLoading(true);
    try {
      const resp = await uploadCsvGuides(file, modelName, datasetName);
      setResults(resp.results);
    } catch (err: any) {
      setError(err.message || 'CSV upload failed');
    } finally {
      setLoading(false);
    }
  };

  // Filter & Sort
  const filteredResults = results
    .filter((r) => {
      if (filterActiveOnly && r.activity_classification?.includes('Low')) return false;
      return true;
    })
    .sort((a, b) => {
      const scoreA = a.relative_efficacy_score ?? -1;
      const scoreB = b.relative_efficacy_score ?? -1;
      return sortAscending ? scoreA - scoreB : scoreB - scoreA;
    });

  const exportCsv = (technical = false) => {
    if (results.length === 0) return;

    const headers = technical
      ? ['Rank', 'Guide_ID', 'Sequence', 'Relative_Efficacy_Score', 'Category', 'Ensemble_Dispersion', 'Dataset', 'Model', 'GC_Content', 'Seed_GC', 'PAM', 'Poly_T_Run', 'Status', 'Error']
      : ['Rank', 'Guide_ID', 'Sequence', 'Relative_Efficacy_Score', 'Category', 'Ensemble_Dispersion', 'Dataset', 'Model'];

    const rows = filteredResults.map((r) => {
      if (technical) {
        return [
          r.rank ?? '',
          r.guide_id,
          r.sequence,
          r.relative_efficacy_score ?? '',
          r.activity_classification ?? '',
          r.ensemble_dispersion ? `±${(r.ensemble_dispersion * 100).toFixed(1)}` : '',
          r.dataset_used ?? datasetName,
          r.model_used ?? modelName,
          r.biophysical_properties?.gc_content_pct ?? '',
          r.biophysical_properties?.seed_gc_pct ?? '',
          r.biophysical_properties?.pam_motif ?? '',
          r.biophysical_properties?.poly_t_max_run ?? '',
          r.status,
          r.error ?? '',
        ];
      }
      return [
        r.rank ?? '',
        r.guide_id,
        r.sequence,
        r.relative_efficacy_score ?? '',
        r.activity_classification ?? '',
        r.ensemble_dispersion ? `±${(r.ensemble_dispersion * 100).toFixed(1)}` : '',
        r.dataset_used ?? datasetName,
        r.model_used ?? modelName,
      ];
    });

    const csvContent =
      'data:text/csv;charset=utf-8,' +
      [headers.join(','), ...rows.map((e) => e.map((val) => `"${val}"`).join(','))].join('\n');

    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', `crispr_batch_${datasetName.toLowerCase()}_${modelName.toLowerCase().replace(/ /g, '_')}_results.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="space-y-6 max-w-5xl mx-auto">
      {/* Header */}
      <div className="border-b border-slate-200 pb-3">
        <h1 className="text-xl font-bold text-slate-900 tracking-tight">Batch sgRNA Prediction & Candidate Ranking</h1>
        <p className="text-xs text-slate-500 mt-0.5">
          Evaluate and rank up to 500 candidate guide sequences simultaneously with dataset-specific models.
        </p>
      </div>

      {/* Input Section */}
      <div className="bio-card p-5 space-y-4">
        <div className="grid grid-cols-1 sm:grid-cols-12 gap-4 items-end">
          <div className="sm:col-span-7 space-y-1.5">
            <label className="text-xs font-bold text-slate-900 uppercase tracking-wider">
              Input Candidate Sequences (One per line)
            </label>
            <textarea
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              placeholder="Paste 20-nt or 23-nt guide sequences here..."
              rows={4}
              className="w-full p-3 font-mono text-xs tracking-wider uppercase rounded-lg border border-slate-300 focus:border-indigo-500 focus:ring-2 focus:ring-indigo-100 bg-white"
            />
          </div>

          <div className="sm:col-span-5 space-y-2.5">
            {/* Dataset Selector */}
            <div>
              <label className="text-[11px] font-semibold text-slate-700 block mb-1 flex items-center gap-1">
                <Database className="w-3.5 h-3.5 text-indigo-600" />
                Target Cell Line
              </label>
              <select
                value={datasetName}
                onChange={(e) => setDatasetName(e.target.value)}
                className="w-full p-1.5 bg-slate-50 border border-slate-300 rounded text-xs font-medium text-slate-800"
              >
                {DATASET_OPTIONS.map((d) => (
                  <option key={d.id} value={d.id}>
                    {d.name}
                  </option>
                ))}
              </select>
            </div>

            {/* Model Selector */}
            <div>
              <label className="text-[11px] font-semibold text-slate-700 block mb-1 flex items-center gap-1">
                <Cpu className="w-3.5 h-3.5 text-indigo-600" />
                Model Architecture
              </label>
              <select
                value={modelName}
                onChange={(e) => setModelName(e.target.value)}
                className="w-full p-1.5 bg-slate-50 border border-slate-300 rounded text-xs font-medium text-slate-800"
              >
                {MODEL_OPTIONS.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.name}
                  </option>
                ))}
              </select>
            </div>

            <div className="flex gap-2 pt-1">
              <button
                onClick={handlePredictText}
                disabled={loading}
                className="flex-1 inline-flex items-center justify-center gap-1.5 px-4 py-2 rounded-lg text-xs font-bold text-white bg-indigo-600 hover:bg-indigo-700 cursor-pointer shadow-xs"
              >
                <Play className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
                {loading ? 'Evaluating...' : 'Evaluate All'}
              </button>

              <label className="inline-flex items-center justify-center px-3 py-2 rounded-lg text-xs font-semibold text-slate-700 bg-white border border-slate-300 hover:bg-slate-50 cursor-pointer" title="Upload CSV file with 'sequence' column">
                <Upload className="w-3.5 h-3.5" />
                <input type="file" accept=".csv" onChange={handleFileUpload} className="hidden" />
              </label>
            </div>
          </div>
        </div>

        {error && (
          <div className="p-2.5 rounded bg-rose-50 border border-rose-200 text-rose-700 text-xs">
            {error}
          </div>
        )}
      </div>

      {/* Results Section */}
      {results.length > 0 && (
        <div className="bio-card p-5 space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div>
              <h2 className="text-xs font-bold text-slate-900 uppercase tracking-wider">
                Ranked Guide Results ({filteredResults.length} of {results.length} guides) &mdash; {datasetName} / {modelName}
              </h2>
              <p className="text-[11px] text-slate-500">
                Sorted by Relative Efficacy Score (Normalized experimental screen activity scale).
              </p>
            </div>

            <div className="flex items-center gap-2 text-xs">
              {/* Filter */}
              <button
                onClick={() => setFilterActiveOnly(!filterActiveOnly)}
                className={`px-2.5 py-1 rounded border text-xs font-medium cursor-pointer transition-colors ${
                  filterActiveOnly
                    ? 'bg-emerald-50 text-emerald-700 border-emerald-300'
                    : 'bg-white text-slate-600 border-slate-200'
                }`}
              >
                Exclude Low Efficacy
              </button>

              {/* Sort Order */}
              <button
                onClick={() => setSortAscending(!sortAscending)}
                className="inline-flex items-center gap-1 px-2.5 py-1 rounded border border-slate-200 bg-white text-slate-600 hover:bg-slate-50 cursor-pointer"
              >
                <ArrowUpDown className="w-3 h-3" />
                {sortAscending ? 'Ascending' : 'Descending'}
              </button>

              {/* Export */}
              <button
                onClick={() => exportCsv(false)}
                className="inline-flex items-center gap-1 px-3 py-1 bg-slate-800 hover:bg-slate-900 text-white rounded text-xs font-semibold cursor-pointer shadow-xs"
              >
                <Download className="w-3 h-3" />
                Export CSV
              </button>
            </div>
          </div>

          <div className="overflow-x-auto border border-slate-200 rounded-lg">
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-50 text-slate-700 font-semibold border-b border-slate-200 text-[11px]">
                <tr>
                  <th className="py-2 px-3 w-16">Rank</th>
                  <th className="py-2 px-3">Sequence (23-nt)</th>
                  <th className="py-2 px-3 text-right">Relative Efficacy Score</th>
                  <th className="py-2 px-3">Classification</th>
                  <th className="py-2 px-3 text-right">Ensemble Dispersion</th>
                  <th className="py-2 px-3">Dataset / Model</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 font-mono text-[11px]">
                {filteredResults.map((r, i) => {
                  const isSuccess = r.status === 'success';
                  return (
                    <tr key={i} className="hover:bg-slate-50/70">
                      <td className="py-2 px-3 font-bold text-slate-800">
                        {r.rank ? `#${r.rank}` : '-'}
                      </td>
                      <td className="py-2 px-3 font-mono font-medium text-slate-900">
                        {r.sequence}
                      </td>
                      <td className="py-2 px-3 text-right">
                        {isSuccess && r.relative_efficacy_score !== null ? (
                          <span className="font-bold text-slate-900">
                            {r.relative_efficacy_score?.toFixed(1)}{' '}
                            <span className="text-[10px] text-slate-400 font-normal">/ 100</span>
                          </span>
                        ) : (
                          <span className="text-rose-500 font-sans">Error</span>
                        )}
                      </td>
                      <td className="py-2 px-3">
                        {isSuccess ? (
                          <span
                            className={`px-2 py-0.5 rounded-full text-[10px] font-bold border ${
                              r.activity_classification?.includes('High')
                                ? 'bg-emerald-50 text-emerald-700 border-emerald-200'
                                : r.activity_classification?.includes('Moderate')
                                ? 'bg-indigo-50 text-indigo-700 border-indigo-200'
                                : 'bg-slate-100 text-slate-700 border-slate-200'
                            }`}
                          >
                            {r.activity_classification}
                          </span>
                        ) : (
                          <span className="text-[10px] text-rose-600 font-sans">{r.error}</span>
                        )}
                      </td>
                      <td className="py-2 px-3 text-right text-slate-600">
                        {r.ensemble_dispersion ? (
                          <span>&plusmn;{(r.ensemble_dispersion * 100).toFixed(1)}</span>
                        ) : (
                          <span className="text-slate-400 text-[10px] font-sans">N/A</span>
                        )}
                      </td>
                      <td className="py-2 px-3 font-sans text-[11px] text-slate-600">
                        {r.dataset_used || datasetName} &bull; {r.model_used || modelName}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      <Disclaimer />
    </div>
  );
};
