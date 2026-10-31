import React from 'react';
import { Disclaimer } from '../components/Disclaimer';
import { BookOpen, Dna, Layers, ShieldCheck, Award, Database } from 'lucide-react';

export const MethodologyPage: React.FC = () => {
  return (
    <div className="space-y-6 max-w-5xl mx-auto">
      {/* 1-Sentence Header */}
      <div className="border-b border-slate-200 pb-3">
        <h1 className="text-xl font-bold text-slate-900 tracking-tight">Computational & Biophysical Methodology</h1>
        <p className="text-xs text-slate-500 mt-0.5">
          Detailed breakdown of feature extraction, dataset-specific 80/20 holdout + 100-fold cross-validation, and SpCas9 cleavage mechanics.
        </p>
      </div>

      {/* Dataset-Specific Evaluation Framework */}
      <div className="bio-card p-5 space-y-3 border-indigo-200 bg-gradient-to-r from-indigo-50/30 to-white">
        <h2 className="text-xs font-bold text-slate-900 uppercase tracking-wider flex items-center gap-1.5">
          <ShieldCheck className="w-4 h-4 text-indigo-600" />
          Dataset-Specific 80/20 Holdout + 100-Fold Cross-Validation Protocol
        </h2>
        <p className="text-xs text-slate-600 leading-relaxed">
          For each cell-line dataset independently (HeLa, HCT116, A375, HL60), <strong>80% of the data is reserved as a development set and 20% is evaluated as a fixed historical holdout partition (seed 42)</strong>. The 80% development set is evaluated using <strong>100-fold cross-validation</strong> (where each iteration uses ~99% of development data for training and 1% for validation). Final model performance is measured on the disjoint 20% holdout partition.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs pt-1">
          <div className="p-3 rounded border border-slate-200 bg-white space-y-1">
            <div className="font-bold text-slate-800">1. Zero Sequence Leakage</div>
            <p className="text-slate-600">
              Zero sequence overlap is strictly enforced between the 80% development set and the 20% final validation holdout, as well as across all 100 cross-validation folds.
            </p>
          </div>

          <div className="p-3 rounded border border-slate-200 bg-white space-y-1">
            <div className="font-bold text-slate-800">2. Leakage-Safe Preprocessing</div>
            <p className="text-slate-600">
              StandardScalers are fitted strictly on the training partitions and only applied as transformations to the validation and final holdout sets.
            </p>
          </div>

          <div className="p-3 rounded border border-slate-200 bg-white space-y-1">
            <div className="font-bold text-slate-800">3. Out-Of-Fold Thresholding</div>
            <p className="text-slate-600">
              Operating classification thresholds are determined on out-of-fold development continuous predictions using Youden&apos;s J statistic and frozen before final validation evaluation.
            </p>
          </div>
        </div>
      </div>

      {/* Feature Engineering Architecture */}
      <div className="bio-card p-5 space-y-3">
        <h2 className="text-xs font-bold text-slate-900 uppercase tracking-wider flex items-center gap-1.5">
          <Layers className="w-4 h-4 text-indigo-600" />
          593 Engineered Sequence & Biophysical Features
        </h2>
        <p className="text-xs text-slate-600 leading-relaxed">
          The feature engine generates 593 biophysical descriptors across the 23-mer guide sequence (5&apos; to 3&apos;, ending with the 3-nt PAM):
        </p>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs pt-1">
          <div className="p-3 rounded border border-slate-200 bg-slate-50/50 space-y-1">
            <div className="font-bold text-slate-800">1. Position-Specific 1-mers (92 Features)</div>
            <p className="text-slate-600">
              Binary indicator for each nucleotide (A, C, G, T) across all 23 positions (23 &times; 4 = 92). Captures base preferences at the cleavage site (17/18) and seed (11-20).
            </p>
          </div>

          <div className="p-3 rounded border border-slate-200 bg-slate-50/50 space-y-1">
            <div className="font-bold text-slate-800">2. Positional Adjacent 2-mers (352 Features)</div>
            <p className="text-slate-600">
              Binary indicators for adjacent dinucleotides at every step along the 22 transition junctions (22 &times; 16 = 352). Captures local DNA stacking energy and helical flexibility.
            </p>
          </div>

          <div className="p-3 rounded border border-slate-200 bg-slate-50/50 space-y-1">
            <div className="font-bold text-slate-800">3. Position-Independent k-mers (80 Features)</div>
            <p className="text-slate-600">
              Frequency counts of all 16 dinucleotides and 64 trinucleotides across the entire guide. Captures global compositional bias.
            </p>
          </div>

          <div className="p-3 rounded border border-slate-200 bg-slate-50/50 space-y-1">
            <div className="font-bold text-slate-800">4. n-Gapped Dinucleotides (48 Features)</div>
            <p className="text-slate-600">
              Counts of nucleotide pairs separated by gap distances of 1, 2, and 3 bases (3 &times; 16 = 48). Captures periodic helical periodicity.
            </p>
          </div>
        </div>
      </div>

      {/* SpCas9 Biophysical Mechanics */}
      <div className="bio-card p-5 space-y-3">
        <h2 className="text-xs font-bold text-slate-900 uppercase tracking-wider flex items-center gap-1.5">
          <Dna className="w-4 h-4 text-emerald-600" />
          SpCas9 Structural Biology & Kinetics
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs">
          <div className="p-3 rounded border border-slate-200 bg-white space-y-1">
            <div className="font-bold text-slate-900">Cleavage Site (Bases 17 & 18)</div>
            <p className="text-slate-600 leading-relaxed">
              Cas9 catalytic residues (HNH and RuvC domains) cut target and non-target strands 3 bp upstream of the PAM. Empirical models identify Cytosine at position 18 as having strong influence on predicted efficacy, which is consistent with previously reported experimental sequence preferences.
            </p>
          </div>

          <div className="p-3 rounded border border-slate-200 bg-white space-y-1">
            <div className="font-bold text-slate-900">Seed Region (Bases 11 to 20)</div>
            <p className="text-slate-600 leading-relaxed">
              The 8-10 bases immediately adjacent to the PAM form the seed. R-loop formation initiates here. Moderate GC content (40-60%) ensures rapid strand invasion without excessive RNA-DNA hybrid stability.
            </p>
          </div>

          <div className="p-3 rounded border border-slate-200 bg-white space-y-1">
            <div className="font-bold text-slate-900">Poly-T Terminator Motif</div>
            <p className="text-slate-600 leading-relaxed">
              Four or more consecutive thymines (TTTT) act as a strong transcription termination signal for RNA Polymerase III (U6/H1 promoters), preventing complete sgRNA expression in lentiviral and plasmid vectors.
            </p>
          </div>
        </div>
      </div>

      {/* Model Association vs Biological Causality */}
      <div className="bio-card p-5 space-y-2 border-indigo-200 bg-indigo-50/20">
        <h3 className="text-xs font-bold text-indigo-900 uppercase tracking-wider flex items-center gap-1.5">
          <ShieldCheck className="w-4 h-4 text-indigo-700" />
          Model Association vs. Biological Causal Mechanism
        </h3>
        <p className="text-xs text-slate-700 leading-relaxed">
          In computational biology, predictive models identify <strong>statistical associations</strong> between sequence patterns and measured knockout phenotypes. An influential feature (such as <code className="font-mono bg-white px-1 py-0.5 rounded border border-indigo-200 text-indigo-800">pos_18_C</code>) had strong influence on the model&apos;s prediction and is consistent with previously reported structural preferences in the literature, but computational attribution alone does not constitute independent proof of direct biochemical causality.
        </p>
      </div>

      <Disclaimer />
    </div>
  );
};
