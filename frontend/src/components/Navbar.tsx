import React from 'react';
import { Activity, FileSpreadsheet, Award, Database, Cpu, BookOpen, Dna } from 'lucide-react';

export type ActiveTab = 'predict' | 'batch' | 'evaluation' | 'dataset' | 'models' | 'methodology';

interface NavbarProps {
  activeTab: ActiveTab;
  setActiveTab: (tab: ActiveTab) => void;
  isBackendHealthy: boolean;
}

export const Navbar: React.FC<NavbarProps> = ({ activeTab, setActiveTab, isBackendHealthy }) => {
  const navItems: Array<{ id: ActiveTab; label: string; icon: React.ReactNode }> = [
    { id: 'predict', label: 'Predict', icon: <Activity className="w-4 h-4" /> },
    { id: 'batch', label: 'Batch', icon: <FileSpreadsheet className="w-4 h-4" /> },
    { id: 'evaluation', label: 'Evaluation', icon: <Award className="w-4 h-4" /> },
    { id: 'dataset', label: 'Dataset', icon: <Database className="w-4 h-4" /> },
    { id: 'models', label: 'Models', icon: <Cpu className="w-4 h-4" /> },
    { id: 'methodology', label: 'Methodology', icon: <BookOpen className="w-4 h-4" /> },
  ];

  return (
    <header className="sticky top-0 z-50 bg-white/95 backdrop-blur-md border-b border-slate-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-14">
          {/* Logo & Brand */}
          <div
            className="flex items-center gap-2.5 cursor-pointer select-none"
            onClick={() => setActiveTab('predict')}
          >
            <div className="w-8 h-8 rounded-lg bg-indigo-600 flex items-center justify-center text-white shadow-xs">
              <Dna className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="text-base font-bold text-slate-900 tracking-tight">CRISPR Guide AI</span>
                <span className="text-[10px] font-semibold tracking-wider uppercase px-1.5 py-0.5 rounded bg-indigo-50 text-indigo-700 border border-indigo-200">
                  SpCas9
                </span>
              </div>
            </div>
          </div>

          {/* Navigation Links */}
          <nav className="hidden md:flex items-center gap-1">
            {navItems.map((item) => {
              const isActive = activeTab === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => setActiveTab(item.id)}
                  className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-semibold transition-colors cursor-pointer ${
                    isActive
                      ? 'bg-slate-900 text-white shadow-2xs'
                      : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
                  }`}
                >
                  {item.icon}
                  {item.label}
                </button>
              );
            })}
          </nav>

          {/* Backend Status */}
          <div className="flex items-center gap-2">
            <div
              className={`inline-flex items-center gap-1.5 text-[11px] px-2.5 py-0.5 rounded-full border ${
                isBackendHealthy
                  ? 'bg-emerald-50 text-emerald-700 border-emerald-200'
                  : 'bg-amber-50 text-amber-700 border-amber-200'
              }`}
            >
              <span className={`w-1.5 h-1.5 rounded-full ${isBackendHealthy ? 'bg-emerald-500' : 'bg-amber-500'}`} />
              <span className="font-medium hidden sm:inline">
                {isBackendHealthy ? 'API Online' : 'Connecting...'}
              </span>
            </div>
          </div>
        </div>

        {/* Mobile Navigation */}
        <div className="md:hidden flex overflow-x-auto py-1.5 gap-1 border-t border-slate-100 no-scrollbar">
          {navItems.map((item) => (
            <button
              key={item.id}
              onClick={() => setActiveTab(item.id)}
              className={`shrink-0 inline-flex items-center gap-1 px-2.5 py-1 rounded text-xs font-medium whitespace-nowrap ${
                activeTab === item.id ? 'bg-slate-900 text-white font-semibold' : 'text-slate-600'
              }`}
            >
              {item.icon}
              {item.label}
            </button>
          ))}
        </div>
      </div>
    </header>
  );
};
