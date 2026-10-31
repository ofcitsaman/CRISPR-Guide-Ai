import React, { useState } from 'react';
import { Disclaimer } from '../components/Disclaimer';
import { Cpu, ChevronDown, ChevronUp, CheckCircle2, ShieldCheck, Database } from 'lucide-react';

export const ModelInfoPage: React.FC = () => {
  const [expandedModel, setExpandedModel] = useState<string | null>('Weighted Ensemble Regressor');

  const models = [
    {
      name: 'Weighted Ensemble Regressor',
      type: 'Non-Negative Constrained Stacking Ensemble',
      purpose: 'Optimal multi-paradigm meta-model achieving top continuous correlation across benchmark datasets.',
      whyIncluded: 'Combines the regularized linear additive properties of Ridge, non-linear axis-aligned partitioning of Random Forest & XGBoost, and margin-maximizing support vectors of SVR without overfitting.',
      featuresUsed: '600 engineered sequence & biophysical features (via base models)',
      technicalDetails:
        'Solves a non-negative constrained quadratic optimization min ||y_dev - sum(w_m * y_oof,m)||^2 subject to w_m >= 0 and sum(w_m) = 1.0 using out-of-fold cross-validation predictions exclusively from the 80% development pool. Evaluates empirical ensemble dispersion using underlying Random Forest trees.',
    },
    {
      name: 'XGBoost Regressor',
      type: 'Gradient-Boosted Decision Trees (200-250 Estimators)',
      purpose: 'High-performance non-linear model capturing epistatic nucleotide interactions.',
      whyIncluded: 'Consistently provides top rank correlation across biological sequence feature spaces with leaf regularization and column subsampling.',
      featuresUsed: '600 engineered sequence & biophysical features',
      technicalDetails:
        'Constructs 200-250 boosted trees with max_depth=4-6, learning_rate=0.03, subsample=0.8, and colsample_bytree=0.6-0.7. Optimizes second-order gradient approximations with exact tree splitting and L1/L2 leaf weight penalties.',
    },
    {
      name: 'Random Forest Regressor',
      type: 'Ensemble Bagged Decision Trees (250-300 Trees)',
      purpose: 'Non-linear tree ensemble providing empirical ensemble dispersion.',
      whyIncluded: 'Evaluates variance among decorrelated bootstrap trees to provide empirical dispersion (±1.96σ) reflecting tree disagreement.',
      featuresUsed: '600 engineered sequence & biophysical features',
      technicalDetails:
        'Constructs 250-300 decorrelated decision trees with max_depth=12-16, min_samples_leaf=1-3, and max_features=0.25-0.35. Evaluates variation among tree outputs {y_b} without making unverified Bayesian or normality assumptions.',
    },
    {
      name: 'Support Vector Regression (SVM/SVR)',
      type: 'Kernel Support Vector Machine (RBF Kernel)',
      purpose: 'Non-linear maximum-margin regression in high-dimensional feature space.',
      whyIncluded: 'Standard kernel baseline effective in high-dimensional space with epsilon-insensitive loss robust to experimental assay noise. Tuned hyperparameters achieved up to +17% holdout correlation gains.',
      featuresUsed: '600 standardized sequence & biophysical features',
      technicalDetails:
        'Fits a Support Vector Regressor with Radial Basis Function (RBF) kernel, C=1.0-2.0, and epsilon=0.05-0.08 on strictly standardized feature matrices. Solves the dual quadratic optimization problem to identify support vectors.',
    },
    {
      name: 'Ridge Linear Regression',
      type: 'Parametric Linear Model with L2 Regularization',
      purpose: 'Transparent additive baseline evaluating linear positional frequencies.',
      whyIncluded: 'Standard scientific reference baseline testing whether complex non-linear ensembles outperform additive positional contributions.',
      featuresUsed: '600 standardized sequence features',
      technicalDetails:
        'Analytical Ridge regression with alpha=200-1000 fitted on standardized feature matrices with top-k feature selection. Signed coefficients provide interpretable positional preferences and penalties.',
    },
  ];

  return (
    <div className="space-y-6 max-w-5xl mx-auto">
      {/* 1-Sentence Header */}
      <div className="border-b border-slate-200 pb-3">
        <h1 className="text-xl font-bold text-slate-900 tracking-tight">Predictive Model Architectures</h1>
        <p className="text-xs text-slate-500 mt-0.5">
          Overview of the 4 production machine learning models trained independently on each cell-line dataset.
        </p>
      </div>

      {/* Architecture Context Banner */}
      <div className="bio-card p-4 bg-slate-50 border-slate-200 text-xs text-slate-700 leading-relaxed space-y-1">
        <div className="font-bold text-slate-900 flex items-center gap-1.5">
          <ShieldCheck className="w-4 h-4 text-indigo-600" />
          <span>Independent Dataset-Specific Training</span>
        </div>
        <p>
          Each of the four models below is trained independently for each cell line (HeLa, HCT116, A375, HL60) using an 80% development pool with 100-fold cross-validation and evaluated on a 20% fixed historical holdout partition (seed 42).
        </p>
      </div>

      {/* Model Cards */}
      <div className="space-y-3">
        {models.map((m) => {
          const isExpanded = expandedModel === m.name;
          return (
            <div key={m.name} className="bio-card p-5 space-y-3">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div>
                  <div className="flex items-center gap-2">
                    <h2 className="text-sm font-bold text-slate-900">{m.name}</h2>
                    <span className="text-[10px] font-medium bg-slate-100 text-slate-700 px-2 py-0.5 rounded border border-slate-200">
                      {m.type}
                    </span>
                  </div>
                  <p className="text-xs text-slate-600 mt-1">{m.purpose}</p>
                </div>
              </div>

              <div className="text-xs text-slate-600 bg-slate-50 rounded-md p-2.5 border border-slate-100">
                <span className="font-semibold text-slate-800">Why included: </span>
                {m.whyIncluded}
              </div>

              {/* Expandable Technical Details */}
              <div>
                <button
                  onClick={() => setExpandedModel(isExpanded ? null : m.name)}
                  className="flex items-center justify-between w-full text-xs font-semibold text-indigo-700 hover:text-indigo-900 pt-1 cursor-pointer"
                >
                  <span>Technical & Hyperparameter Specification</span>
                  {isExpanded ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
                </button>

                {isExpanded && (
                  <div className="mt-2.5 p-3 rounded bg-slate-50 border border-slate-200 space-y-2 text-xs">
                    <div>
                      <span className="font-semibold text-slate-700 block text-[11px]">Feature Representation:</span>
                      <p className="text-slate-600 font-mono text-[11px] mt-0.5">{m.featuresUsed}</p>
                    </div>

                    <div>
                      <span className="font-semibold text-slate-700 block text-[11px]">Implementation Details:</span>
                      <p className="text-slate-600 leading-relaxed mt-0.5">{m.technicalDetails}</p>
                    </div>
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>

      <Disclaimer />
    </div>
  );
};
