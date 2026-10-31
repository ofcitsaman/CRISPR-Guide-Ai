import React from 'react';
import { Scissors, AlertCircle } from 'lucide-react';

interface SequenceVisualizerProps {
  sequence: string;
  positionalWeights?: number[];
  showCleavageMarker?: boolean;
}

const BASE_COLORS: Record<string, { bg: string; text: string; border: string }> = {
  A: { bg: 'bg-emerald-50', text: 'text-emerald-700', border: 'border-emerald-300' },
  C: { bg: 'bg-blue-50', text: 'text-blue-700', border: 'border-blue-300' },
  G: { bg: 'bg-amber-50', text: 'text-amber-700', border: 'border-amber-300' },
  T: { bg: 'bg-rose-50', text: 'text-rose-700', border: 'border-rose-300' },
};

export const SequenceVisualizer: React.FC<SequenceVisualizerProps> = ({
  sequence,
  positionalWeights,
  showCleavageMarker = true,
}) => {
  const bases = sequence.toUpperCase().split('');
  const is23 = bases.length === 23;

  return (
    <div className="w-full bg-white border border-slate-200 rounded-xl p-5 shadow-xs">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h4 className="text-sm font-semibold text-slate-900 tracking-tight">
            Interactive sgRNA Nucleotide Architecture
          </h4>
          <p className="text-xs text-slate-500">
            {is23 ? '20-nt Protospacer Target + 3-nt PAM (Positions 1 to 23)' : `${bases.length}-nt Sequence`}
          </p>
        </div>

        {/* Region Legend */}
        {is23 && (
          <div className="hidden sm:flex items-center gap-3 text-xs">
            <span className="inline-flex items-center gap-1 text-slate-600">
              <span className="w-2.5 h-2.5 rounded-full bg-slate-300 inline-block" />
              Distal (1-10)
            </span>
            <span className="inline-flex items-center gap-1 text-indigo-700 font-medium">
              <span className="w-2.5 h-2.5 rounded-full bg-indigo-500 inline-block" />
              Seed (11-20)
            </span>
            <span className="inline-flex items-center gap-1 text-amber-700 font-medium">
              <span className="w-2.5 h-2.5 rounded-full bg-amber-500 inline-block" />
              PAM (21-23)
            </span>
          </div>
        )}
      </div>

      {/* Nucleotide Track */}
      <div className="relative overflow-x-auto pb-4 pt-2">
        <div className="inline-flex items-end gap-1 min-w-[620px] px-1">
          {bases.map((base, idx) => {
            const pos = idx + 1;
            const style = BASE_COLORS[base] || { bg: 'bg-slate-50', text: 'text-slate-700', border: 'border-slate-300' };
            const isSeed = pos >= 11 && pos <= 20;
            const isPam = pos >= 21;
            const isCleavageFlank = pos === 17 || pos === 18;
            const weight = positionalWeights && positionalWeights[idx] !== undefined ? positionalWeights[idx] : null;

            return (
              <div key={idx} className="relative flex flex-col items-center group">
                {/* Cleavage Marker between 17 and 18 */}
                {showCleavageMarker && pos === 17 && (
                  <div className="absolute -right-2 top-0 bottom-0 z-20 flex flex-col items-center justify-center pointer-events-none">
                    <div className="h-full border-r-2 border-dashed border-rose-500/80 my-1" />
                    <div className="absolute -top-3 bg-rose-600 text-white rounded-full p-0.5 shadow-xs">
                      <Scissors className="w-2.5 h-2.5 rotate-90" />
                    </div>
                  </div>
                )}

                {/* Nucleotide Box */}
                <div
                  className={`w-7 h-9 rounded-md border flex items-center justify-center font-mono font-bold text-sm transition-all duration-150 group-hover:scale-105 group-hover:shadow-sm ${style.bg} ${style.text} ${style.border} ${
                    isSeed ? 'ring-1 ring-indigo-300' : ''
                  } ${isPam ? 'ring-1 ring-amber-400 font-extrabold' : ''}`}
                  title={`Position ${pos}: ${base} ${isPam ? '(PAM)' : isSeed ? '(Seed)' : '(Distal)'}`}
                >
                  {base}
                </div>

                {/* Position Label */}
                <span className="text-[10px] font-mono text-slate-400 mt-1">
                  {pos}
                </span>

                {/* Saliency Bar if provided */}
                {weight !== null && (
                  <div className="w-full h-8 flex flex-col justify-end items-center mt-1">
                    <div
                      className={`w-4 rounded-t transition-all ${
                        isPam ? 'bg-amber-500' : isSeed ? 'bg-indigo-500' : 'bg-slate-400'
                      }`}
                      style={{ height: `${Math.max(4, Math.min(32, weight * 32))}px` }}
                      title={`Model Saliency: ${(weight * 100).toFixed(1)}%`}
                    />
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Annotations Footer */}
      {is23 && (
        <div className="mt-3 pt-3 border-t border-slate-100 flex flex-wrap items-center justify-between text-xs text-slate-500 gap-2">
          <div className="flex items-center gap-2">
            <span className="font-mono bg-rose-50 text-rose-700 px-1.5 py-0.5 rounded border border-rose-200">
              Cut Site: Between Pos 17 & 18
            </span>
            <span className="font-mono bg-indigo-50 text-indigo-700 px-1.5 py-0.5 rounded border border-indigo-200">
              Seed: Pos 11-20
            </span>
            <span className="font-mono bg-amber-50 text-amber-700 px-1.5 py-0.5 rounded border border-amber-200">
              PAM: Pos 21-23 ({sequence.slice(20, 23)})
            </span>
          </div>
          {positionalWeights && (
            <span className="text-slate-400 italic">
              Vertical bars indicate model sensitivity / feature importance at each nucleotide.
            </span>
          )}
        </div>
      )}
    </div>
  );
};
