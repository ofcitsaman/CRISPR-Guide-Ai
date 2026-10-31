import React from 'react';
import { Info } from 'lucide-react';

export const Disclaimer: React.FC = () => {
  return (
    <div className="bg-slate-50 border border-slate-200 rounded-lg p-3.5 flex items-start gap-3 text-xs text-slate-600">
      <Info className="w-4 h-4 text-slate-400 shrink-0 mt-0.5" />
      <div>
        <span className="font-semibold text-slate-800">Scientific Research Notice: </span>
        This application provides computational predictions of CRISPR-Cas9 guide RNA efficiency based on previously measured experimental data. Predictions are not experimental validation and should not be interpreted as guaranteed biological outcomes.
      </div>
    </div>
  );
};
