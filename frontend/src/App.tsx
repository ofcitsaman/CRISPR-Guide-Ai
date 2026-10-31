import React, { useEffect, useState } from 'react';
import { Navbar, type ActiveTab } from './components/Navbar';
import { PredictPage } from './pages/PredictPage';
import { BatchPredictPage } from './pages/BatchPredictPage';
import { ModelEvaluationPage } from './pages/ModelEvaluationPage';
import { ModelInfoPage } from './pages/ModelInfoPage';
import { DatasetInfoPage } from './pages/DatasetInfoPage';
import { MethodologyPage } from './pages/MethodologyPage';
import { checkHealth } from './services/api';
import { Dna } from 'lucide-react';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<ActiveTab>('predict');
  const [isBackendHealthy, setIsBackendHealthy] = useState(false);

  useEffect(() => {
    async function verifyHealth() {
      try {
        const res = await checkHealth();
        setIsBackendHealthy(res.status === 'healthy');
      } catch {
        setIsBackendHealthy(false);
      }
    }
    verifyHealth();
    const interval = setInterval(verifyHealth, 15000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen flex flex-col bg-slate-50 text-slate-900 selection:bg-indigo-500 selection:text-white font-sans antialiased">
      {/* Top Navigation */}
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        isBackendHealthy={isBackendHealthy}
      />

      {/* Main Page Content */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {activeTab === 'predict' && <PredictPage />}
        {activeTab === 'batch' && <BatchPredictPage />}
        {activeTab === 'evaluation' && <ModelEvaluationPage />}
        {activeTab === 'dataset' && <DatasetInfoPage />}
        {activeTab === 'models' && <ModelInfoPage />}
        {activeTab === 'methodology' && <MethodologyPage />}
      </main>

      {/* Clean Scientific Footer */}
      <footer className="bg-white border-t border-slate-200 mt-auto py-5">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-3 text-xs text-slate-500">
          <div className="flex items-center gap-2">
            <div className="w-5 h-5 rounded-md bg-indigo-600 text-white flex items-center justify-center">
              <Dna className="w-3.5 h-3.5" />
            </div>
            <span className="font-semibold text-slate-800">CRISPR Guide AI</span>
            <span>&bull;</span>
            <span>Sequence-Disjoint Scientific Framework</span>
          </div>

          <div className="flex items-center gap-4 text-xs font-medium text-slate-600">
            <button
              onClick={() => setActiveTab('evaluation')}
              className="hover:text-indigo-600 transition-colors cursor-pointer"
            >
              Evaluation Benchmarks
            </button>
            <button
              onClick={() => setActiveTab('dataset')}
              className="hover:text-indigo-600 transition-colors cursor-pointer"
            >
              Dataset Provenance
            </button>
            <button
              onClick={() => setActiveTab('methodology')}
              className="hover:text-indigo-600 transition-colors cursor-pointer"
            >
              Methodology
            </button>
          </div>

          <div className="text-[11px] text-slate-400">
            Evaluated on <span className="font-mono text-slate-600">DeepCRISPR & CRISPRpred(SEQ)</span> human screen benchmarks.
          </div>
        </div>
      </footer>
    </div>
  );
};

export default App;
