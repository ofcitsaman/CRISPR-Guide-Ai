import React, { useEffect, useState } from 'react';
import { fetchDatasetInfo } from '../services/api';
import { DatasetInfoResponse } from '../types';
import { Disclaimer } from '../components/Disclaimer';
import { Database, ExternalLink, ShieldAlert, BookOpen } from 'lucide-react';

export const DatasetInfoPage: React.FC = () => {
  const [data, setData] = useState<DatasetInfoResponse | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      try {
        const res = await fetchDatasetInfo();
        setData(res);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  if (loading || !data) {
    return (
      <div className="py-12 text-center text-xs text-slate-500">
        Loading dataset metadata...
      </div>
    );
  }

  return (
    <div className="space-y-6 max-w-5xl mx-auto">
      {/* 1-Sentence Header */}
      <div className="border-b border-slate-200 pb-3">
        <h1 className="text-xl font-bold text-slate-900 tracking-tight">Experimental Training Dataset</h1>
        <p className="text-xs text-slate-500 mt-0.5">
          Provenance, sample composition, target variables, and citations for the 16,749 experimentally validated human sgRNAs.
        </p>
      </div>

      {/* Dataset Summary Cards */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
        <div className="bio-card p-4">
          <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">Total Samples</span>
          <div className="text-xl font-mono font-bold text-slate-900 mt-1">16,749</div>
          <span className="text-[11px] text-slate-500">Screening data points</span>
        </div>

        <div className="bio-card p-4">
          <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">Unique sgRNAs</span>
          <div className="text-xl font-mono font-bold text-indigo-700 mt-1">12,465</div>
          <span className="text-[11px] text-slate-500">Distinct 23-mer sequences</span>
        </div>

        <div className="bio-card p-4">
          <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">Human Cell Lines</span>
          <div className="text-xl font-mono font-bold text-slate-900 mt-1">4 Lines</div>
          <span className="text-[11px] text-slate-500">HeLa, HCT116, A375, HL60</span>
        </div>

        <div className="bio-card p-4">
          <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">Primary Target</span>
          <div className="text-xl font-mono font-bold text-emerald-700 mt-1">[0.0, 1.0]</div>
          <span className="text-[11px] text-slate-500">Normalized activity scale</span>
        </div>
      </div>

      {/* Cell Line Breakdown Table */}
      <div className="bio-card p-5 space-y-3">
        <h2 className="text-xs font-bold text-slate-900 uppercase tracking-wider">
          Cell Line Composition & Assays
        </h2>

        <div className="overflow-x-auto border border-slate-200 rounded-lg">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-50 text-slate-700 font-semibold border-b border-slate-200">
              <tr>
                <th className="py-2.5 px-3">Cell Line</th>
                <th className="py-2.5 px-3">Organism / Lineage</th>
                <th className="py-2.5 px-3 text-right">Samples</th>
                <th className="py-2.5 px-3">Original Study & Assay</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 font-mono text-xs">
              {data.cell_lines.map((c) => (
                <tr key={c.name} className="hover:bg-slate-50/70">
                  <td className="py-2.5 px-3 font-sans font-bold uppercase text-slate-900">{c.name}</td>
                  <td className="py-2.5 px-3 font-sans text-slate-600">{c.organism}</td>
                  <td className="py-2.5 px-3 text-right font-bold text-indigo-700">{c.samples.toLocaleString()}</td>
                  <td className="py-2.5 px-3 font-sans text-slate-700">{c.original_study}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Target Variable & Binary Label Definition */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
        <div className="bio-card p-5 space-y-2">
          <h3 className="font-bold text-slate-900 uppercase tracking-wider text-xs">
            Target Variable: Relative Efficacy Score
          </h3>
          <p className="text-slate-600 leading-relaxed">
            The target variable is a continuous float bounded in [0.0, 1.0], derived from collaborative filtering and rank normalization of raw experimental screening log-fold changes.
          </p>
          <p className="text-[11px] text-slate-500 bg-slate-50 p-2.5 rounded border border-slate-200">
            <strong>Display Convention:</strong> Scaled by 100 for user visualization (e.g. 78.4 / 100). This is a relative activity score, NOT a percent probability of cleavage.
          </p>
        </div>

        <div className="bio-card p-5 space-y-2">
          <h3 className="font-bold text-slate-900 uppercase tracking-wider text-xs">
            Binary Classification Definition
          </h3>
          <p className="text-slate-600 leading-relaxed">
            Ground-truth classification uses the experimental study's original binary criterion:
          </p>
          <div className="font-mono text-slate-900 bg-slate-50 p-2.5 rounded border border-slate-200">
            Active Guide: log2 fold-change &ge; 1.0 (binary_label = 1)
          </div>
          <p className="text-[11px] text-slate-500">
            Used directly for calculating ROC-AUC and PR-AUC. No artificial thresholding is applied.
          </p>
        </div>
      </div>

      {/* Dataset Limitations & Outlier Biology */}
      <div className="bio-card p-5 space-y-2 border-amber-200 bg-amber-50/15">
        <div className="flex items-center gap-2 text-amber-900 font-bold text-xs">
          <ShieldAlert className="w-4 h-4 text-amber-600" />
          Dataset Scope & Limitations
        </div>
        <ul className="text-xs text-slate-600 space-y-1.5 list-disc pl-4 leading-relaxed">
          <li>
            <strong>SpCas9 Canonical PAM (NGG):</strong> The dataset evaluates <em>Streptococcus pyogenes</em> Cas9 with 5'-NGG PAM sequences. Non-NGG Cas enzymes (Cas12a, CasX) are not modeled.
          </li>
          <li>
            <strong>Assay Differences:</strong> Hart et al. (HeLa/HCT116) measured cell viability dropout of core essential genes genome-wide, whereas Doench et al. (A375) measured drug resistance and cell viability across 8 densely tiled human genes. Cross-assay transfer without calibration degrades performance.
          </li>
          <li>
            <strong>Sequence-Only Modeling:</strong> Models evaluate intrinsic guide sequence and thermodynamic properties. Chromatin accessibility, nucleosome positioning, and local DNA methylation at endogenous loci are not captured.
          </li>
        </ul>
      </div>

      {/* Primary Citations */}
      <div className="bio-card p-5 space-y-3">
        <h3 className="text-xs font-bold text-slate-900 uppercase tracking-wider">
          Primary Benchmark Literature
        </h3>
        <div className="space-y-2 text-xs">
          {data.citations.map((c, idx) => (
            <div key={idx} className="p-3 rounded border border-slate-100 bg-slate-50/60 flex items-start justify-between gap-3">
              <div>
                <div className="font-semibold text-slate-900">{c.title}</div>
                <div className="text-[11px] text-slate-500 mt-0.5">
                  {c.authors} &bull; <em>{c.journal}</em> ({c.year})
                </div>
              </div>
              <a
                href={c.url}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-1 text-[11px] font-mono text-indigo-600 hover:text-indigo-800 shrink-0"
              >
                DOI <ExternalLink className="w-3 h-3" />
              </a>
            </div>
          ))}
        </div>
      </div>

      <Disclaimer />
    </div>
  );
};
