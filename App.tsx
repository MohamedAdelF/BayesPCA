
import * as React from 'react';
import { useState, useMemo, useEffect, useRef } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  ScatterChart, Scatter, LineChart, Line
} from 'recharts';
import {
  Brain, Database, ChevronRight, Activity, Layers,
  Grid3X3, Info, Download, Upload, Zap, Sparkles, RefreshCcw, ArrowRight, CheckCircle2, Box, Maximize2, Minimize2, Eye, Layout, Cpu,
  Table, Filter, BarChart3, TrendingUp
} from 'lucide-react';
import Plotly from 'plotly.js-dist';
import { DataPoint, Page, Metrics, PCAResult } from './types';
import { runPCA, GaussianNaiveBayes, MinimumDistanceClassifier, calculateMetrics, calculateMean, calculateCovariance, getMultivariateGaussianPDF } from './utils/math';

// --- CONSTANTS ---
const COLORS = ['#6366f1', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'];

// --- DATASET PRESETS (Adjusted for Realistic Complexity) ---
const getIrisData = () => Array.from({ length: 150 }, (_, i) => {
  const cls = i < 50 ? "Setosa" : i < 100 ? "Versicolor" : "Virginica";
  // Reduced offset to create overlap, increased noise amplitude
  const offset = i < 50 ? -0.8 : i < 100 ? 0 : 0.8;
  return {
    Sepal_L: 5.8 + offset + (Math.random() - 0.5) * 2.5,
    Sepal_W: 3.0 - offset * 0.2 + (Math.random() - 0.5) * 2.0,
    Petal_L: 3.7 + offset * 1.5 + (Math.random() - 0.5) * 2.5,
    Petal_W: 1.2 + offset * 0.8 + (Math.random() - 0.5) * 1.5,
    Species: cls
  };
});

const getWineData = () => Array.from({ length: 178 }, (_, i) => {
  const cls = i < 59 ? "Class_1" : i < 130 ? "Class_2" : "Class_3";
  // Reduced offsets significantly
  const offset = i < 59 ? 0.8 : i < 130 ? 0 : -0.8;
  return {
    Alcohol: 13 + offset + (Math.random() - 0.5) * 2.0,
    MalicAcid: 2 + (Math.random() - 0.5) * 3.0,
    Ash: 2.3 + (Math.random() - 0.5) * 1.0,
    Alkalinity: 19 - offset + (Math.random() - 0.5) * 8,
    Magnesium: 100 + offset * 5 + (Math.random() - 0.5) * 40,
    Label: cls
  };
});

const getCancerData = () => Array.from({ length: 200 }, (_, i) => {
  const isMalignant = i < 100;
  const cls = isMalignant ? "Malignant" : "Benign";
  // Reduced shift, added more noise to critical features
  const shift = isMalignant ? 1.0 : -1.0;
  return {
    Radius: 15 + shift + (Math.random() - 0.5) * 8,
    Texture: 20 + shift * 1.0 + (Math.random() - 0.5) * 10,
    Perimeter: 90 + shift * 3 + (Math.random() - 0.5) * 30,
    Area: 700 + shift * 50 + (Math.random() - 0.5) * 300,
    Smoothness: 0.1 + (isMalignant ? 0.01 : 0) + (Math.random() - 0.5) * 0.04,
    Diagnosis: cls
  };
});

// --- PERFORMANCE PATCH ---
// Fixes "Multiple readback operations..." warning by forcing willReadFrequently
const originalGetContext = HTMLCanvasElement.prototype.getContext;
// @ts-ignore
HTMLCanvasElement.prototype.getContext = function (type: string, attributes?: any) {
  if (type === '2d') {
    attributes = { ...attributes, willReadFrequently: true };
  }
  return originalGetContext.call(this, type, attributes);
};

// --- HELPER COMPONENTS ---
const PlotlyChart = ({ data, layout, style, config }: { data: any[], layout: any, style?: React.CSSProperties, config?: any }) => {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    // Use Plotly.react for updates (faster, preserves state)
    // It handles both new plots and updates
    Plotly.react(containerRef.current, data, layout, {
      responsive: true,
      displayModeBar: false,
      ...config
    });

    // Cleanup only on unmount
    return () => {
      // We don't necessarily need to purge on every update, only on unmount
      // But we can't easily detect unmount inside this effect if dependencies change.
      // However, for typical React + Plotly usage, we rely on Plotly.react to handle diffs.
    };
  }, [data, layout, config]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (containerRef.current) Plotly.purge(containerRef.current);
    };
  }, []);

  return <div ref={containerRef} style={{ width: '100%', height: '100%', ...style }} />;
};

const App: React.FC = () => {
  // Dataset Management State
  const [datasets, setDatasets] = useState<{ name: string, data: DataPoint[], isCustom: boolean }[]>([
    { name: 'Wine Quality', data: getWineData(), isCustom: false },
    { name: 'Iris Flowers', data: getIrisData(), isCustom: false },
    { name: 'Breast Cancer', data: getCancerData(), isCustom: false }
  ]);
  const [currentDatasetName, setCurrentDatasetName] = useState('Wine Quality');
  const [data, setData] = useState<DataPoint[]>(getWineData()); // Local copy for performance

  // Sync data when dataset changes
  useEffect(() => {
    const selected = datasets.find(d => d.name === currentDatasetName);
    if (selected) setData(selected.data);
  }, [currentDatasetName, datasets]);

  const [currentPage, setCurrentPage] = useState<Page>(Page.UPLOAD);
  const [targetColumn, setTargetColumn] = useState<string>('Label');
  const [selectedModel, setSelectedModel] = useState<'bayes' | 'mindist'>('bayes');
  const [allFeatures, setAllFeatures] = useState<string[]>([]);
  const [selectedFeatures, setSelectedFeatures] = useState<string[]>([]);
  const [baselineMetrics, setBaselineMetrics] = useState<Metrics | null>(null);
  const [is3DFocus, setIs3DFocus] = useState(false);
  const [pcaViewMode, setPcaViewMode] = useState<'2D' | '3D'>('2D');
  const [useNormalization, setUseNormalization] = useState(true);

  // --- STATE DERIVATIONS ---
  const uniqueClasses = useMemo(() => Array.from(new Set(data.map(d => String(d[targetColumn])))), [data, targetColumn]);

  useEffect(() => {
    const keys = Object.keys(data[0] || {});
    const numericKeys = keys.filter(k => !isNaN(Number(data[0][k])) && k !== targetColumn);
    setAllFeatures(numericKeys);
    setSelectedFeatures(numericKeys);
  }, [data, targetColumn]);

  // Helper to safely get numeric matrix, handling NaNs
  const getMatrix = (features: string[], normalize = false) => {
    const rawMatrix = data.map(d => features.map(f => {
      const val = Number(d[f]);
      return isNaN(val) ? 0 : val;
    }));

    if (normalize && rawMatrix.length > 0) {
      // Z-Score Normalization (Standardize)
      const n = rawMatrix.length;
      const p = rawMatrix[0].length;
      const means = new Array(p).fill(0);
      const stds = new Array(p).fill(0);

      // Calculate means
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < p; j++) {
          means[j] += rawMatrix[i][j];
        }
      }
      for (let j = 0; j < p; j++) means[j] /= n;

      // Calculate stddev
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < p; j++) {
          stds[j] += Math.pow(rawMatrix[i][j] - means[j], 2);
        }
      }
      for (let j = 0; j < p; j++) stds[j] = Math.sqrt(stds[j] / (n - 1)) || 1;

      // Apply formulation
      return rawMatrix.map(row => row.map((val, j) => (val - means[j]) / stds[j]));
    }

    return rawMatrix;
  };

  // Requirement 2: Baseline Metrics
  useEffect(() => {
    if (allFeatures.length > 0) {
      const matrix = getMatrix(allFeatures);
      const labels = data.map(d => String(d[targetColumn]));
      const clf = selectedModel === 'bayes' ? new GaussianNaiveBayes() : new MinimumDistanceClassifier();
      clf.fit(matrix, labels);
      const preds = clf.predict(matrix);
      setBaselineMetrics(calculateMetrics(labels, preds));
    }
  }, [allFeatures, data, targetColumn, selectedModel]);

  // Requirement 5 & 6: Re-classification with selected features
  const currentMetrics = useMemo(() => {
    if (selectedFeatures.length === 0) return null;
    const matrix = getMatrix(selectedFeatures);
    const labels = data.map(d => String(d[targetColumn]));
    const clf = selectedModel === 'bayes' ? new GaussianNaiveBayes() : new MinimumDistanceClassifier();
    clf.fit(matrix, labels);
    const preds = clf.predict(matrix);
    return calculateMetrics(labels, preds);
  }, [selectedFeatures, data, targetColumn, selectedModel]);

  // Requirement 3: Covariance Σ per class
  const classStats = useMemo(() => {
    const stats = new Map<string, { cov: number[][], mean: number[] }>();
    uniqueClasses.forEach(cls => {
      const classData = data.filter(d => String(d[targetColumn]) === cls);
      // Use clean numeric values
      const matrix = classData.map(d => selectedFeatures.map(f => {
        const v = Number(d[f]);
        return isNaN(v) ? 0 : v;
      }));
      if (matrix.length > 1) {
        const means = calculateMean(matrix);
        const cov = calculateCovariance(matrix, means);
        stats.set(cls, { cov, mean: means });
      }
    });
    return stats;
  }, [data, targetColumn, selectedFeatures, uniqueClasses]);

  // Requirement 3 & 4: PCA / Eigen Analysis
  const pcaResult = useMemo(() => {
    // PCA should ideally be on standardized data if features have different scales
    const matrix = getMatrix(selectedFeatures, useNormalization);
    if (!matrix || matrix.length === 0 || matrix[0].length === 0) return null;
    return runPCA(matrix);
  }, [selectedFeatures, data, useNormalization]);

  const projectedData = useMemo(() => {
    if (!pcaResult || selectedFeatures.length < 2) return [];
    // Must project the same data form used for PCA
    const matrix = getMatrix(selectedFeatures, useNormalization);

    // Safely retrieve eigenvectors, defaulting to zero-vector if missing
    const numFeatures = selectedFeatures.length;
    // Ensure we have vectors to dot product with
    const ev1 = pcaResult.eigenvectors[0] || new Array(numFeatures).fill(0);
    const ev2 = pcaResult.eigenvectors[1] || new Array(numFeatures).fill(0);
    const ev3 = pcaResult.eigenvectors[2] || new Array(numFeatures).fill(0);

    return matrix.map(point => ({
      pc1: point.reduce((sum, val, i) => sum + val * (ev1[i] || 0), 0),
      pc2: point.reduce((sum, val, i) => sum + val * (ev2[i] || 0), 0),
      pc3: point.reduce((sum, val, i) => sum + val * (ev3[i] || 0), 0),
    }));
  }, [selectedFeatures, pcaResult, useNormalization]);

  const pca3DTraces = useMemo(() => {
    if (!projectedData.length) return [];

    return uniqueClasses.map((cls, idx) => {
      // Safe filtering matching by index
      const indices = data.map((d, i) => String(d[targetColumn]) === cls ? i : -1).filter(i => i !== -1);
      const filtered = indices.map(i => projectedData[i]);

      return {
        x: filtered.map(d => d.pc1),
        y: filtered.map(d => d.pc2),
        z: filtered.map(d => d.pc3),
        mode: 'markers', type: 'scatter3d', name: cls,
        marker: {
          size: 5,
          color: COLORS[idx % COLORS.length],
          opacity: 0.9,
          line: { color: 'white', width: 0.5 }
        },
        hovertemplate: `<b>${cls}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<br>PC3: %{z:.2f}<extra></extra>`
      };
    });
  }, [projectedData, data, targetColumn, uniqueClasses]);

  // Requirement 7: Sketch probability distribution (MATCHING IMAGE A and B)
  const bayesianTraces = useMemo(() => {
    if (selectedFeatures.length < 2 || !classStats.size) return [];

    // Use only top 2 selected features for visualization
    const f1 = selectedFeatures[0];
    const f2 = selectedFeatures[1];

    // Determine range dynamically
    const f1Vals = data.map(d => Number(d[f1]));
    const f2Vals = data.map(d => Number(d[f2]));
    const minX = Math.min(...f1Vals), maxX = Math.max(...f1Vals);
    const minY = Math.min(...f2Vals), maxY = Math.max(...f2Vals);
    const paddingX = (maxX - minX) * 0.2;
    const paddingY = (maxY - minY) * 0.2;

    const steps = 50;
    const xValues = Array.from({ length: steps }, (_, i) => minX - paddingX + (i * (maxX - minX + 2 * paddingX)) / (steps - 1));
    const yValues = Array.from({ length: steps }, (_, i) => minY - paddingY + (i * (maxY - minY + 2 * paddingY)) / (steps - 1));

    const traces: any[] = [];
    const classes = Array.from(classStats.keys());

    classes.forEach((cls, idx) => {
      // Get actual stats for these 2 dimensions
      const fullStats = classStats.get(cls)!;
      // Extract 2x2 covariance and means for the first 2 selected features
      // Note: classStats.mean/cov correspond to `selectedFeatures` order
      const mean = [fullStats.mean[0], fullStats.mean[1]];
      const cov = [
        [fullStats.cov[0][0], fullStats.cov[0][1]],
        [fullStats.cov[1][0], fullStats.cov[1][1]]
      ];

      const z = yValues.map(yi => xValues.map(xi => {
        return getMultivariateGaussianPDF([xi, yi], mean, cov);
      }));

      // (a) 3D Surface - Wireframe Mesh style
      traces.push({
        z, x: xValues, y: yValues, type: 'surface', name: `${cls} (a)`,
        showscale: false, opacity: 0.7,
        colorscale: [[0, '#e2e8f0'], [1, COLORS[idx % COLORS.length]]],
        contours: {
          x: { show: true, color: '#fff', width: 0.5 },
          y: { show: true, color: '#fff', width: 0.5 },
          z: { show: true, color: '#fff', width: 0.5, project: { z: true } }
        },
        scene: 'scene1'
      });

      // (b) 2D Contours - Concentric Circles style
      traces.push({
        z, x: xValues, y: yValues, type: 'contour', name: `${cls} (b)`,
        showscale: false, ncontours: 12,
        line: { color: COLORS[idx % COLORS.length], width: 1.5, smoothing: 1.3 },
        xaxis: 'x2', yaxis: 'y2'
      });
    });
    return traces;
  }, [classStats, selectedFeatures, data]);

  // --- HANDLERS ---
  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();

    reader.onload = (event) => {
      const text = event.target?.result as string;
      const lines = text.split('\n').map(l => l.trim()).filter(l => l);
      if (lines.length < 2) return;

      const headers = lines[0].split(',').map(h => h.trim().replace(/^"|"$/g, '').replace(/^'|'$/g, ''));
      const parsedData = lines.slice(1).map(line => {
        // Simple CSV parser (not robust for commas inside quotes, but better than basic split)
        const values = line.split(',');
        const obj: any = {};
        headers.forEach((h, i) => {
          // Attempt to parse number, keep string if NaN
          let rawVal = values[i]?.trim();
          if (rawVal) rawVal = rawVal.replace(/^"|"$/g, '').replace(/^'|'$/g, '');
          const numVal = Number(rawVal);
          obj[h] = (rawVal === '' || isNaN(numVal)) ? rawVal : numVal;
        });
        return obj;
      });

      // Add to datasets list
      const newDatasetName = file.name;
      const newDataset = { name: newDatasetName, data: parsedData, isCustom: true };

      setDatasets(prev => {
        // Avoid duplicates by name
        const filtered = prev.filter(d => d.name !== newDatasetName);
        return [...filtered, newDataset];
      });
      setCurrentDatasetName(newDatasetName);

      // Auto-detect target column (usually the last string column or just the last column)
      const keys = Object.keys(parsedData[0] || {});
      const potentialTarget = keys[keys.length - 1];
      setTargetColumn(potentialTarget);

      // Reset metrics
      setBaselineMetrics(null);
    };
    reader.readAsText(file);
  };

  return (
    <div className={`min-h-screen flex flex-col bg-slate-50 transition-all duration-700 ${is3DFocus ? 'bg-slate-900 overflow-hidden' : ''}`}>

      {/* 3D THEATER OVERLAY */}
      {is3DFocus && (
        <div className="fixed inset-0 z-[100] bg-slate-950 flex flex-col animate-in fade-in zoom-in duration-300">
          <div className="p-6 bg-slate-900/90 backdrop-blur-2xl border-b border-white/10 flex justify-between items-center">
            <div className="flex items-center gap-4 text-white">
              <div className="bg-indigo-600 p-2 rounded-lg shadow-lg">
                <Maximize2 size={24} className="text-white" />
              </div>
              <div>
                <h2 className="font-black text-lg tracking-widest uppercase italic">IMMERSIVE 3D SYSTEM</h2>
                <p className="text-indigo-400 text-[10px] font-black uppercase tracking-[0.3em]">Lecture 9 & 10 Protocol Active</p>
              </div>
            </div>
            <button onClick={() => setIs3DFocus(false)} className="text-white hover:bg-white/10 p-3 rounded-full border border-white/10 transition active:scale-90">
              <Minimize2 size={32} />
            </button>
          </div>
          <div className="flex-1 bg-slate-950">
            <PlotlyChart
              data={currentPage === Page.PCA ? pca3DTraces : bayesianTraces.filter(t => t.type === 'surface')}
              layout={{
                scene: {
                  xaxis: { gridcolor: '#1e293b', color: '#fff', title: selectedFeatures[0] || 'Feature 1' },
                  yaxis: { gridcolor: '#1e293b', color: '#fff', title: selectedFeatures[1] || 'Feature 2' },
                  zaxis: { gridcolor: '#1e293b', color: '#fff', title: currentPage === Page.PCA ? 'PC 3' : 'Prob. Density' },
                  bgcolor: '#020617',
                  camera: { eye: { x: 1.8, y: 1.8, z: 1.2 } }
                },
                margin: { l: 0, r: 0, b: 0, t: 0 },
                paper_bgcolor: '#020617',
                font: { color: '#fff', family: 'Inter' },
                showlegend: true,
                legend: { font: { color: '#fff' }, x: 0.05, y: 0.95 }
              }}
              config={{ displayModeBar: true }}
            />
          </div>
        </div>
      )}

      {/* Header */}
      {!is3DFocus && (
        <header className="bg-indigo-950 text-white p-4 shadow-2xl sticky top-0 z-50 border-b border-indigo-900">
          <div className="max-w-7xl mx-auto flex items-center justify-between">
            <div className="flex items-center space-x-3 group cursor-pointer" onClick={() => setCurrentPage(Page.UPLOAD)}>
              <div className="bg-indigo-500 p-2 rounded-xl shadow-lg group-hover:rotate-6 transition">
                <Brain size={24} />
              </div>
              <div>
                <h1 className="text-xl font-black tracking-tighter uppercase italic">BayesPCA <span className="text-indigo-400">v2.1</span></h1>
                <p className="text-[9px] font-black text-indigo-300 uppercase tracking-widest leading-none">Cognitive Analytics</p>
              </div>
            </div>
            <nav className="flex space-x-1 overflow-x-auto no-scrollbar">
              {[
                { page: Page.UPLOAD, label: "Upload" },
                { page: Page.METRICS, label: "Baseline Classification" },
                { page: Page.COVARIANCE, label: "Covariance" },
                { page: Page.PCA, label: "PCA" },
                { page: Page.SELECTION, label: "Selection" },
                { page: Page.DISTRIBUTION, label: "Distribution" }
              ].map((item) => (
                <button
                  key={item.page}
                  onClick={() => setCurrentPage(item.page)}
                  className={`px-4 py-2 rounded-lg text-[10px] font-black uppercase tracking-widest transition whitespace-nowrap ${currentPage === item.page ? 'bg-indigo-600 text-white shadow-lg' : 'text-indigo-300 hover:text-white hover:bg-white/5'}`}
                >
                  {item.label}
                </button>
              ))}
            </nav>
          </div>
        </header>
      )}

      <main className={`flex-1 max-w-7xl w-full mx-auto p-4 md:p-8 space-y-8 ${is3DFocus ? 'hidden' : ''}`}>

        {/* Page Headings */}
        <div className="flex flex-col md:flex-row justify-between items-end gap-4">
          <div className="space-y-1">
            <div className="flex items-center gap-3 text-indigo-600 font-black text-xs uppercase tracking-widest">
              <Activity size={14} /> Rubric Step: {Object.values(Page).indexOf(currentPage) + 1} / 6
            </div>
            <h2 className="text-4xl font-black text-slate-800 tracking-tighter uppercase italic leading-none">
              {currentPage === Page.UPLOAD ? "1. Data Architect" :
                currentPage === Page.METRICS ? "2. Baseline Classification" :
                  currentPage === Page.COVARIANCE ? "3. Covariance Matrix Σ" :
                    currentPage === Page.PCA ? "4. Eigen Analysis" :
                      currentPage === Page.SELECTION ? "5. Feature Selection" :
                        "6. Likelihood Geometry"}
            </h2>
          </div>
          {(currentPage === Page.PCA || currentPage === Page.DISTRIBUTION) && (
            <button onClick={() => setIs3DFocus(true)} className="bg-indigo-600 text-white px-8 py-4 rounded-2xl font-black text-xs uppercase tracking-widest shadow-2xl shadow-indigo-100 flex items-center gap-3 hover:-translate-y-1 transition active:scale-95 group">
              <Maximize2 size={18} className="group-hover:rotate-12 transition" />
              Launch 3D Cinema
            </button>
          )}
        </div>

        {/* --- PAGE: UPLOAD --- */}
        {currentPage === Page.UPLOAD && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 animate-in fade-in duration-500">
            <div className="lg:col-span-4 space-y-6">
              <div className="bg-white p-8 rounded-[2.5rem] shadow-sm border border-slate-200">
                <h3 className="text-xs font-black text-slate-400 uppercase tracking-widest mb-6 flex items-center gap-2">
                  <Layout size={16} className="text-indigo-600" /> Data Management
                </h3>

                <div className="space-y-3 max-h-[300px] overflow-y-auto pr-2 scrollbar-thin">
                  {datasets.map(ds => (
                    <div
                      key={ds.name}
                      onClick={() => setCurrentDatasetName(ds.name)}
                      className={`relative w-full p-4 border rounded-3xl text-left cursor-pointer transition-all group flex items-center justify-between
                        ${currentDatasetName === ds.name
                          ? 'bg-indigo-600 border-indigo-600 shadow-xl scale-[1.02]'
                          : 'bg-slate-50 border-slate-100 hover:border-indigo-300 hover:bg-white'}`}
                    >
                      <div className="flex items-center gap-3 overflow-hidden">
                        <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${currentDatasetName === ds.name ? 'bg-white/20' : 'bg-white shadow-sm'}`}>
                          {ds.isCustom ? <Upload size={14} className={currentDatasetName === ds.name ? 'text-white' : 'text-slate-400'} /> : <Database size={14} className={currentDatasetName === ds.name ? 'text-white' : 'text-indigo-500'} />}
                        </div>
                        <div className="flex flex-col min-w-0">
                          <span className={`text-[10px] font-black uppercase tracking-wide truncate ${currentDatasetName === ds.name ? 'text-white' : 'text-slate-700'}`}>
                            {ds.name}
                          </span>
                          <span className={`text-[9px] font-bold ${currentDatasetName === ds.name ? 'text-indigo-200' : 'text-slate-400'}`}>
                            {ds.data.length} Vectors
                          </span>
                        </div>
                      </div>

                      {ds.isCustom && (
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            setDatasets(prev => prev.filter(p => p.name !== ds.name));
                            if (currentDatasetName === ds.name) setCurrentDatasetName(datasets[0].name);
                          }}
                          className={`p-2 rounded-full hover:bg-red-500 hover:text-white transition z-10 ${currentDatasetName === ds.name ? 'text-indigo-200' : 'text-slate-300'}`}
                          title="Delete Dataset"
                        >
                          <div className="w-3 h-3 flex items-center justify-center">×</div>
                        </button>
                      )}
                    </div>
                  ))}
                </div>

                <div className="mt-8 pt-6 border-t border-slate-100">
                  <div className="flex justify-between items-center mb-4">
                    <h3 className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Processing Configuration</h3>
                    <span className="text-[9px] font-bold px-2 py-1 bg-slate-100 rounded text-slate-500">Settings</span>
                  </div>

                  <div className="space-y-4">
                    {/* Model Selector */}
                    <div className="flex gap-2 p-1 bg-slate-50 rounded-2xl border border-slate-100">
                      <button onClick={() => { setSelectedModel('bayes'); setBaselineMetrics(null); }} className={`flex-1 py-3 rounded-xl font-black text-[9px] uppercase tracking-wider transition ${selectedModel === 'bayes' ? 'bg-indigo-600 text-white shadow-lg' : 'text-slate-400 hover:text-slate-600'}`}>Naive Bayes</button>
                      <button onClick={() => { setSelectedModel('mindist'); setBaselineMetrics(null); }} className={`flex-1 py-3 rounded-xl font-black text-[9px] uppercase tracking-wider transition ${selectedModel === 'mindist' ? 'bg-indigo-600 text-white shadow-lg' : 'text-slate-400 hover:text-slate-600'}`}>Min. Dist.</button>
                    </div>

                    {/* Target Column Selector */}
                    <div>
                      <label className="text-[9px] font-bold text-slate-400 uppercase mb-2 block ml-2">Target Feature (Class)</label>
                      <select
                        value={targetColumn}
                        onChange={(e) => setTargetColumn(e.target.value)}
                        className="w-full text-xs font-bold text-slate-700 bg-slate-50 border border-slate-200 rounded-xl px-4 py-3 outline-none focus:border-indigo-500 focus:ring-2 focus:ring-indigo-200 transition"
                      >
                        {Object.keys(data[0] || {}).map(k => (
                          <option key={k} value={k}>{k}</option>
                        ))}
                      </select>
                    </div>

                    {/* Normalization Toggle */}
                    <div
                      className={`flex items-center justify-between p-4 rounded-xl border transition cursor-pointer ${useNormalization ? 'bg-emerald-50 border-emerald-200' : 'bg-slate-50 border-slate-200'}`}
                      onClick={() => setUseNormalization(!useNormalization)}
                    >
                      <div className="flex items-center gap-3">
                        <div className={`w-4 h-4 rounded flex items-center justify-center ${useNormalization ? 'bg-emerald-500 text-white' : 'bg-slate-300'}`}>
                          {useNormalization && <div className="text-[10px]">✓</div>}
                        </div>
                        <span className="text-[10px] font-black uppercase text-slate-600">Auto-Standardize Data</span>
                      </div>
                      <span className="text-[9px] text-slate-400 italic">Recommended for 3D</span>
                    </div>
                  </div>
                </div>

                <div className="mt-8 pt-8 border-t border-slate-100">
                  <div className="relative group border-4 border-dotted border-slate-100 rounded-3xl p-8 text-center hover:bg-indigo-50/50 hover:border-indigo-200 transition cursor-pointer overflow-hidden">
                    <input type="file" accept=".csv" onChange={handleFileUpload} className="absolute inset-0 opacity-0 cursor-pointer z-10" />
                    <div className="absolute inset-0 bg-indigo-50/0 group-hover:bg-indigo-50/50 transition duration-500" />
                    <Upload size={32} className="mx-auto text-slate-300 mb-3 group-hover:text-indigo-500 group-hover:scale-110 transition duration-500 relative z-0" />
                    <p className="text-[10px] font-black text-slate-400 uppercase tracking-widest relative z-0 group-hover:text-indigo-400">Upload New Dataset</p>
                  </div>
                </div>
              </div>
            </div>
            <div className="lg:col-span-8 bg-white rounded-[3rem] border border-slate-200 overflow-hidden h-[700px] flex flex-col shadow-sm">
              <div className="p-8 border-b bg-slate-50/30 flex justify-between items-center">
                <h3 className="text-xs font-black text-slate-800 uppercase italic flex items-center gap-2">
                  <Table size={16} className="text-indigo-600" />
                  Observational Frame
                </h3>
                <span className="text-[10px] font-black bg-indigo-100 text-indigo-700 px-5 py-2 rounded-full uppercase shadow-sm">{data.length} Vectors Registered</span>
              </div>
              <div className="flex-1 overflow-auto bg-slate-50/10">
                <table className="w-full text-[10px] text-left">
                  <thead className="bg-white sticky top-0 border-b border-slate-100 z-10 shadow-sm">
                    <tr>{Object.keys(data[0] || {}).map(h => <th key={h} className="px-8 py-5 font-black uppercase text-slate-400">{h}</th>)}</tr>
                  </thead>
                  <tbody className="divide-y divide-slate-50">
                    {data.slice(0, 50).map((row, i) => (
                      <tr key={i} className="hover:bg-white transition">
                        {Object.values(row).map((v, j) => <td key={j} className="px-8 py-5 text-slate-600 font-bold">{typeof v === 'number' ? v.toFixed(2) : <span className="px-3 py-1 bg-indigo-50 text-indigo-600 rounded-lg">{v}</span>}</td>)}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* --- PAGE: METRICS (BASELINE) --- */}
        {currentPage === Page.METRICS && baselineMetrics && (
          <div className="space-y-8 animate-in slide-in-from-bottom duration-500">
            <div className="flex justify-between items-center px-4">
              <div className="flex items-center gap-3 bg-indigo-50 px-6 py-3 rounded-full border border-indigo-100">
                <CheckCircle2 size={16} className="text-emerald-500" />
                <span className="text-xs font-black text-indigo-800 uppercase tracking-wider">
                  Model Successfully Applied to {data.length} Vectors
                </span>
              </div>
              <div className="text-xs font-black text-slate-400 uppercase tracking-wider">
                Full Feature Set ({allFeatures.length} Dimensions)
              </div>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
              {[
                { label: 'System Accuracy', val: baselineMetrics.accuracy, color: 'text-indigo-600' },
                { label: 'Precision Factor', val: baselineMetrics.precision, color: 'text-emerald-500' },
                { label: 'Recall Rate', val: baselineMetrics.recall, color: 'text-amber-500' },
                { label: 'F1 Harmonic', val: baselineMetrics.f1, color: 'text-rose-500' }
              ].map(s => (
                <div key={s.label} className="bg-white p-10 rounded-[2.5rem] border border-slate-200 shadow-sm hover:shadow-2xl transition-all">
                  <span className="text-[10px] font-black text-slate-400 uppercase tracking-[0.4em]">{s.label}</span>
                  <p className={`text-6xl font-black mt-4 tracking-tighter ${s.color}`}>{(s.val * 100).toFixed(1)}%</p>
                  <div className="mt-8 w-full h-2 bg-slate-50 rounded-full overflow-hidden">
                    <div className="h-full bg-slate-200" style={{ width: `${s.val * 100}%` }} />
                  </div>
                </div>
              ))}
            </div>
            {/* Confusion Matrix & Details Section */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              {/* Detail Block */}
              <div className="bg-slate-900 text-white p-12 rounded-[3.5rem] relative overflow-hidden flex flex-col justify-center">
                <div className="absolute top-0 right-0 p-12 opacity-10">
                  <Activity size={200} />
                </div>
                <div className="relative z-10 space-y-6">
                  <div className="flex items-center gap-3 mb-2">
                    <div className="bg-indigo-500 w-2 h-2 rounded-full animate-pulse" />
                    <span className="text-indigo-400 text-[10px] font-black uppercase tracking-widest">Statistical Foundation</span>
                  </div>
                  <h3 className="text-3xl font-black uppercase italic tracking-tight">Baseline <br /> Evaluation</h3>
                  <p className="text-slate-400 font-medium leading-relaxed max-w-md">
                    This initial evaluation establishes the performance benchmark using the <strong>full high-dimensional feature set</strong> ({allFeatures.length}D).
                    It serves as the reference point (Ground Truth) to measure how well our PCA dimensionality reduction preserves the separability of the classes later.
                  </p>

                  <div className="flex flex-col gap-4 pt-6">
                    <div className="bg-white/5 p-4 rounded-2xl border border-white/5">
                      <span className="text-[10px] font-bold text-indigo-300 uppercase block mb-1">Classifier Algorithm</span>
                      <span className="text-lg font-black">{selectedModel === 'bayes' ? 'Gaussian Naive Bayes' : 'Minimum Distance'}</span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Confusion Matrix */}
              <div className="bg-white p-12 rounded-[3.5rem] border border-slate-200 shadow-sm flex flex-col">
                <h3 className="text-lg font-black text-slate-800 uppercase italic mb-8 flex items-center gap-3">
                  <Grid3X3 size={20} className="text-indigo-600" />
                  Confusion Matrix
                </h3>

                {baselineMetrics.confusionMatrix ? (
                  <div className="flex-1 flex items-center justify-center">
                    <div className="overflow-auto max-w-full">
                      <div className="grid gap-1" style={{ gridTemplateColumns: `auto repeat(${uniqueClasses.length}, minmax(60px, 1fr))` }}>
                        {/* Header Row */}
                        <div className="h-10"></div> {/* Empty corner */}
                        {uniqueClasses.map(cls => (
                          <div key={cls} className="h-10 flex items-end justify-center pb-2">
                            <span className="text-[9px] font-black uppercase text-slate-400 rotate-0 whitespace-nowrap" title={cls}>
                              {cls.substring(0, 4)}..
                            </span>
                          </div>
                        ))}

                        {/* Rows */}
                        {uniqueClasses.map((actualCls, i) => (
                          <React.Fragment key={actualCls}>
                            {/* Row Label */}
                            <div className="h-14 flex items-center justify-end pr-3">
                              <span className="text-[9px] font-black uppercase text-slate-400 whitespace-nowrap" title={actualCls}>
                                {actualCls.substring(0, 8)}
                              </span>
                            </div>

                            {/* Cells */}
                            {uniqueClasses.map((predCls, j) => {
                              const val = baselineMetrics.confusionMatrix?.[i]?.[j] || 0;
                              const total = baselineMetrics.confusionMatrix?.[i]?.reduce((a: number, b: number) => a + b, 0) || 1;
                              const intensity = val / total;
                              const isDiag = i === j;

                              return (
                                <div
                                  key={`${i}-${j}`}
                                  className={`h-14 relative rounded-xl flex items-center justify-center text-xs font-black transition-all hover:scale-110 hover:z-10 cursor-default group
                                                ${isDiag ? 'ring-2 ring-indigo-100' : ''}`}
                                  style={{
                                    backgroundColor: isDiag
                                      ? `rgba(99, 102, 241, ${Math.max(0.1, intensity)})` // Indigo for diagonal (TP)
                                      : val > 0 ? `rgba(244, 63, 94, ${Math.max(0.1, intensity)})` : '#f8fafc', // Red for errors
                                    color: intensity > 0.5 ? 'white' : (val === 0 ? '#cbd5e1' : '#1e293b')
                                  }}
                                >
                                  {val}
                                  {/* Tooltip */}
                                  <div className="absolute opacity-0 group-hover:opacity-100 bottom-full mb-2 bg-slate-800 text-white text-[9px] py-1 px-2 rounded-lg pointer-events-none whitespace-nowrap z-20">
                                    True: {actualCls} <br /> Pred: {predCls}
                                  </div>
                                </div>
                              );
                            })}
                          </React.Fragment>
                        ))}
                      </div>
                      <p className="text-center text-[10px] text-slate-400 mt-6 font-medium">Rows: Actual Class • Columns: Predicted Class</p>
                    </div>
                  </div>
                ) : (
                  <div className="flex-1 flex items-center justify-center text-slate-400 font-medium text-sm">
                    Matrix Data Unavailable
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* --- PAGE: COVARIANCE Σ --- */}
        {currentPage === Page.COVARIANCE && (
          <div className="space-y-8 animate-in fade-in duration-500">
            {/* Color Legend */}
            <div className="bg-white p-6 rounded-[2rem] border border-slate-200 shadow-sm">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <Grid3X3 size={20} className="text-indigo-600" />
                  <span className="text-sm font-black text-slate-600 uppercase tracking-wider">Heatmap Scale</span>
                </div>
                <div className="flex items-center gap-6">
                  <div className="flex items-center gap-3">
                    <div className="flex items-center gap-1">
                      <div className="w-12 h-6 rounded-lg" style={{ background: 'linear-gradient(to right, rgba(244, 63, 94, 0.2), rgba(244, 63, 94, 1))' }}></div>
                      <span className="text-xs font-bold text-slate-500 ml-2">Negative</span>
                    </div>
                    <div className="w-px h-6 bg-slate-200"></div>
                    <div className="flex items-center gap-1">
                      <div className="w-12 h-6 rounded-lg" style={{ background: 'linear-gradient(to right, rgba(99, 102, 241, 0.2), rgba(99, 102, 241, 1))' }}></div>
                      <span className="text-xs font-bold text-slate-500 ml-2">Positive</span>
                    </div>
                  </div>
                  <div className="text-xs text-slate-400 font-medium">
                    Diagonal = Variance
                  </div>
                </div>
              </div>
            </div>

            {/* Matrices Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              {Array.from(classStats.entries()).map(([cls, stats], idx) => (
                <div key={cls} className="bg-gradient-to-br from-white to-slate-50/30 p-8 rounded-[3rem] border-2 border-slate-200 shadow-xl flex flex-col">
                  {/* Header */}
                  <div className="flex justify-between items-center mb-6 pb-6 border-b-2 border-slate-100">
                    <div className="flex items-center gap-4">
                      <div className="w-3 h-3 rounded-full shadow-lg ring-4 ring-white" style={{ backgroundColor: COLORS[idx % COLORS.length] }} />
                      <h3 className="text-2xl font-black text-slate-800 uppercase italic tracking-tight">{cls}</h3>
                    </div>
                    <div className="bg-slate-100 px-5 py-2 rounded-full">
                      <span className="text-xs font-black text-slate-500 uppercase tracking-wider">{stats.cov.length}×{stats.cov.length} Matrix</span>
                    </div>
                  </div>

                  {/* Matrix Content Container */}
                  <div className="flex-1 overflow-auto max-h-[500px] scrollbar-thin scrollbar-thumb-slate-300 scrollbar-track-slate-100 rounded-2xl relative">
                    <div className="inline-block min-w-full align-middle">
                      {/* Top Header Row (Sticky) */}
                      <div className="flex items-center gap-1 sticky top-0 z-20 bg-white/95 backdrop-blur-sm py-2 mb-1 border-b border-slate-100">
                        {/* Empty Corner */}
                        <div className="w-20 flex-shrink-0 sticky left-0 z-30 bg-white/95 backdrop-blur-sm"></div>

                        {/* Feature Columns */}
                        <div className="flex gap-1">
                          {selectedFeatures.map((f, i) => (
                            <div key={i} className="w-16 text-center flex-shrink-0">
                              <span className="text-[9px] font-black uppercase text-slate-500 block truncate px-1" title={f}>
                                {f.substring(0, 10)}
                              </span>
                            </div>
                          ))}
                        </div>
                      </div>

                      {/* Data Rows */}
                      <div className="space-y-1 pb-2">
                        {stats.cov.map((row, i) => (
                          <div key={i} className="flex items-center gap-1">
                            {/* Row Label (Left Sticky) */}
                            <div className="w-20 text-right pr-3 flex-shrink-0 sticky left-0 z-10 bg-white/90 backdrop-blur-sm h-12 flex items-center justify-end border-r border-slate-50">
                              <span className="text-[9px] font-black uppercase text-slate-500 truncate block w-full" title={selectedFeatures[i]}>
                                {selectedFeatures[i]}
                              </span>
                            </div>

                            {/* Cells */}
                            <div className="flex gap-1">
                              {row.map((val, j) => {
                                const max = Math.max(...stats.cov.flat().map(Math.abs)) || 1;
                                const intensity = Math.pow(Math.abs(val) / max, 0.6);
                                const isDiag = i === j;
                                return (
                                  <div
                                    key={j}
                                    className={`w-16 h-12 flex-shrink-0 flex items-center justify-center text-[10px] font-black transition-all hover:scale-105 hover:shadow-lg hover:z-0 relative rounded-lg ${isDiag ? 'ring-2 ring-slate-900/20 ring-offset-1' : ''}`}
                                    style={{
                                      backgroundColor: val >= 0
                                        ? `rgba(99, 102, 241, ${Math.max(intensity, 0.15)})`
                                        : `rgba(244, 63, 94, ${Math.max(intensity, 0.15)})`,
                                      color: intensity > 0.55 ? 'white' : '#1e293b',
                                      fontWeight: isDiag ? 900 : 700
                                    }}
                                    title={`Cov(${selectedFeatures[i]}, ${selectedFeatures[j]}) = ${val.toFixed(4)}`}
                                  >
                                    {Math.abs(val) >= 0.01 ? val.toFixed(2) : '0.00'}
                                  </div>
                                );
                              })}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* --- PAGE: PCA --- */}
        {currentPage === Page.PCA && (
          <>
            {pcaResult && selectedFeatures.length >= 2 ? (
              <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 animate-in zoom-in duration-500">
                <div className="lg:col-span-8 bg-white p-12 rounded-[3.5rem] border border-slate-200 h-[650px] flex flex-col shadow-sm">
                  <div className="flex justify-between items-center mb-10">
                    <div>
                      <h3 className="text-xl font-black text-slate-800 uppercase italic">Component Projection</h3>
                      <p className="text-slate-400 text-[10px] font-black uppercase tracking-widest mt-1">Global Clustering Matrix</p>
                    </div>
                    <div className="flex items-center gap-4">
                      {/* Toggle 2D/3D */}
                      <div className="flex bg-slate-100 p-1 rounded-full">
                        <button
                          onClick={() => setPcaViewMode('2D')}
                          className={`px-4 py-2 rounded-full text-[10px] font-black uppercase transition-all ${pcaViewMode === '2D' ? 'bg-white text-indigo-600 shadow-sm' : 'text-slate-400 hover:text-slate-600'}`}
                        >
                          2D View
                        </button>
                        <button
                          onClick={() => setPcaViewMode('3D')}
                          className={`px-4 py-2 rounded-full text-[10px] font-black uppercase transition-all ${pcaViewMode === '3D' ? 'bg-white text-indigo-600 shadow-sm' : 'text-slate-400 hover:text-slate-600'}`}
                          disabled={selectedFeatures.length < 3}
                          title={selectedFeatures.length < 3 ? "Requires 3+ features" : ""}
                        >
                          3D View
                        </button>
                      </div>

                      <div className="h-8 w-px bg-slate-200 mx-2"></div>

                      <div className="flex items-center gap-2">
                        {uniqueClasses.map((cls, i) => (
                          <div key={cls} className="flex items-center gap-2 px-3 py-1 bg-slate-50 rounded-full border border-slate-100">
                            <div className="w-2 h-2 rounded-full" style={{ backgroundColor: COLORS[i % COLORS.length] }} />
                            <span className="text-[9px] font-black uppercase text-slate-500">{cls}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>

                  <div className="flex-1 bg-slate-50/50 rounded-[3rem] border border-slate-100 p-8 shadow-inner overflow-hidden min-h-[450px] flex items-center justify-center relative">
                    {projectedData.length > 0 ? (
                      pcaViewMode === '2D' ? (
                        <div className="overflow-auto w-full h-full flex items-center justify-center">
                          <ScatterChart width={750} height={450} data={projectedData} margin={{ top: 20, right: 20, bottom: 40, left: 40 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                            <XAxis type="number" dataKey="pc1" name="PC1" unit="" label={{ value: 'Principal Component 1', position: 'bottom', offset: 0, fill: '#94a3b8', fontSize: 10, fontWeight: 900 }} tick={{ fontSize: 10, fill: '#64748b' }} />
                            <YAxis type="number" dataKey="pc2" name="PC2" unit="" label={{ value: 'Principal Component 2', angle: -90, position: 'left', fill: '#94a3b8', fontSize: 10, fontWeight: 900 }} tick={{ fontSize: 10, fill: '#64748b' }} />
                            <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ borderRadius: '16px', border: 'none', boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1)' }} />
                            {uniqueClasses.map((cls, idx) => (
                              <Scatter key={cls} name={cls} data={projectedData.filter((_, i) => data[i][targetColumn] === cls)} fill={COLORS[idx % COLORS.length]} opacity={0.6} shape="circle" />
                            ))}
                          </ScatterChart>
                        </div>
                      ) : (
                        <div className="w-full h-full rounded-[2rem] overflow-hidden">
                          {selectedFeatures.length >= 3 ? (
                            <PlotlyChart
                              data={pca3DTraces}
                              layout={{
                                scene: {
                                  xaxis: { title: 'PC1' },
                                  yaxis: { title: 'PC2' },
                                  zaxis: { title: 'PC3' },
                                  camera: { eye: { x: 1.5, y: 1.5, z: 1.5 } }
                                },
                                margin: { l: 0, r: 0, b: 0, t: 0 },
                                paper_bgcolor: 'rgba(0,0,0,0)',
                                showlegend: true,
                                legend: { x: 0, y: 1 }
                              }}
                              config={{ displayModeBar: true }}
                            />
                          ) : (
                            <div className="flex flex-col items-center justify-center h-full text-slate-400">
                              <Info size={32} className="mb-4 opacity-50" />
                              <p className="font-bold">Not enough dimensions</p>
                              <p className="text-sm">3D view requires at least 3 features.</p>
                            </div>
                          )}
                        </div>
                      )
                    ) : (
                      <div className="flex items-center justify-center h-full">
                        <p className="text-slate-400 font-medium">No data to display</p>
                      </div>
                    )}
                  </div>
                </div>
                <div className="lg:col-span-4 flex flex-col gap-6">
                  <div className="bg-indigo-900 p-10 rounded-[3rem] text-white shadow-xl">
                    <h3 className="text-[10px] font-black uppercase tracking-widest mb-6 opacity-70 flex items-center gap-2">
                      <Layers size={14} /> Explained Variance
                    </h3>
                    <div className="space-y-6">
                      {pcaResult.explainedVariance.slice(0, 5).map((val, i) => (
                        <div key={i} className="space-y-2">
                          <div className="flex justify-between items-end">
                            <span className="text-[10px] font-black italic text-indigo-300">Principal {i + 1}</span>
                            <span className="text-xl font-black">{(val * 100).toFixed(1)}%</span>
                          </div>
                          <div className="w-full h-1.5 bg-white/10 rounded-full overflow-hidden">
                            <div className="h-full bg-indigo-400 shadow-[0_0_10px_rgba(129,140,248,0.8)]" style={{ width: `${val * 100}%` }} />
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                  <div className="bg-white p-10 rounded-[3rem] shadow-sm border border-slate-200 flex-1">
                    <h3 className="text-xs font-black text-slate-400 uppercase tracking-widest mb-6">Cumulative Impact</h3>
                    <div className="flex items-center justify-center">
                      {pcaResult.cumulativeVariance && pcaResult.cumulativeVariance.length > 0 ? (
                        <LineChart width={350} height={120} data={pcaResult.cumulativeVariance.slice(0, 5).map((v, i) => ({ name: `PC${i + 1}`, val: v * 100 }))}>
                          <CartesianGrid stroke="#f1f5f9" vertical={false} />
                          <XAxis dataKey="name" hide />
                          <YAxis domain={[0, 100]} hide />
                          <Tooltip contentStyle={{ borderRadius: '12px', border: 'none', boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)' }} />
                          <Line type="monotone" dataKey="val" stroke="#6366f1" strokeWidth={4} dot={{ fill: '#6366f1', r: 6, stroke: '#fff', strokeWidth: 2 }} activeDot={{ r: 8 }} />
                        </LineChart>
                      ) : (
                        <div className="flex items-center justify-center h-32">
                          <p className="text-slate-400 text-xs">No variance data</p>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="flex items-center justify-center min-h-[600px]">
                <div className="bg-white p-16 rounded-[4rem] border-2 border-slate-200 shadow-xl max-w-2xl">
                  <div className="flex flex-col items-center text-center space-y-6">
                    <div className="w-20 h-20 bg-amber-100 rounded-full flex items-center justify-center">
                      <Info size={40} className="text-amber-600" />
                    </div>
                    <h3 className="text-3xl font-black text-slate-800 uppercase italic tracking-tight">PCA Analysis Unavailable</h3>
                    <div className="space-y-3 text-slate-600">
                      <p className="text-lg font-medium">
                        Principal Component Analysis requires:
                      </p>
                      <ul className="text-left space-y-2 bg-slate-50 p-6 rounded-2xl">
                        <li className="flex items-start gap-3">
                          <CheckCircle2 size={20} className={selectedFeatures.length >= 2 ? "text-emerald-500 flex-shrink-0" : "text-slate-300 flex-shrink-0"} />
                          <span className="font-bold">
                            At least <span className="text-indigo-600">2 numeric features</span>
                            {selectedFeatures.length > 0 && ` (Current: ${selectedFeatures.length})`}
                          </span>
                        </li>
                        <li className="flex items-start gap-3">
                          <CheckCircle2 size={20} className={data.length >= 3 ? "text-emerald-500 flex-shrink-0" : "text-slate-300 flex-shrink-0"} />
                          <span className="font-bold">
                            At least <span className="text-indigo-600">3 data points</span>
                            {data.length > 0 && ` (Current: ${data.length})`}
                          </span>
                        </li>
                      </ul>
                      <p className="text-sm text-slate-500 italic mt-4">
                        {selectedFeatures.length === 0
                          ? "Your dataset doesn't contain numeric features. Please upload a CSV with numerical columns."
                          : selectedFeatures.length === 1
                            ? "Only one numeric feature detected. PCA needs at least 2 features for dimensionality reduction."
                            : "Please check your data and try again."}
                      </p>
                    </div>
                    <div className="pt-4">
                      <button
                        onClick={() => setCurrentPage(Page.UPLOAD)}
                        className="px-8 py-4 bg-indigo-600 text-white rounded-2xl font-black text-sm uppercase tracking-widest shadow-lg hover:bg-indigo-700 transition flex items-center gap-3"
                      >
                        <Database size={20} />
                        Back to Upload
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </>
        )}

        {/* --- PAGE: SELECTION & COMPARISON (Preserved Logic, New Style) --- */}
        {currentPage === Page.SELECTION && (
          <div className="space-y-8 animate-in slide-in-from-bottom duration-500">
            <div className="bg-white p-12 rounded-[4rem] border border-slate-200 shadow-sm">
              <h3 className="text-2xl font-black text-slate-800 uppercase italic mb-8">Feature Selection</h3>
              <div className="flex flex-wrap gap-3">
                {allFeatures.map(f => (
                  <button
                    key={f}
                    onClick={() => setSelectedFeatures(prev => prev.includes(f) ? (prev.length > 1 ? prev.filter(x => x !== f) : prev) : [...prev, f])}
                    className={`px-6 py-3 rounded-2xl text-[10px] font-black uppercase tracking-widest transition-all border ${selectedFeatures.includes(f) ? 'bg-indigo-600 border-indigo-600 text-white shadow-lg scale-105' : 'bg-slate-50 border-slate-200 text-slate-400 hover:bg-slate-100'}`}
                  >
                    {f}
                  </button>
                ))}
              </div>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <div className="bg-white p-10 rounded-[3rem] border border-slate-200 shadow-sm">
                <h3 className="font-black text-slate-800 uppercase italic mb-8">Performance Comparison</h3>
                <div className="space-y-4">
                  <div className="flex justify-between p-6 bg-slate-50 rounded-3xl font-black text-xs items-center">
                    <span className="text-slate-400">BASELINE ACCURACY</span>
                    <span className="text-slate-600 text-2xl">{((baselineMetrics?.accuracy || 0) * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between p-6 bg-indigo-50 rounded-3xl font-black text-xs items-center border border-indigo-100">
                    <span className="text-indigo-600 uppercase italic">OPTIMIZED ACCURACY</span>
                    <span className="text-indigo-700 text-2xl">{((currentMetrics?.accuracy || 0) * 100).toFixed(1)}%</span>
                  </div>
                </div>
              </div>
              <div className="bg-slate-900 p-10 rounded-[3rem] text-white flex flex-col justify-between relative overflow-hidden">
                <div className="absolute top-0 right-0 p-10 opacity-10">
                  <Sparkles size={120} />
                </div>
                <h3 className="font-black text-indigo-400 uppercase italic mb-6 relative z-10">Feature Variance Logic</h3>
                <p className="text-slate-400 text-sm font-medium leading-relaxed mb-6 italic relative z-10">
                  Select features that maximize the eigenvalues (variance). High variance features contribute most to the information content and separability of the classes in the projected space.
                </p>
              </div>
            </div>
          </div>
        )}

        {/* --- PAGE: DISTRIBUTION (BAYESIAN LANDSCAPES) --- */}
        {currentPage === Page.DISTRIBUTION && (
          <div className="space-y-12 animate-in fade-in zoom-in duration-700">
            <div className="bg-slate-950 rounded-[5rem] p-20 shadow-[0_60px_120px_-30px_rgba(0,0,0,0.6)] overflow-hidden relative">
              <div className="absolute inset-0 bg-gradient-to-br from-indigo-500/10 to-transparent pointer-events-none" />

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-20 items-center">
                <div className="space-y-10 relative z-10 text-center lg:text-left">
                  <div className="inline-flex items-center gap-3 px-8 py-3 bg-white/5 rounded-full border border-white/10 backdrop-blur-xl">
                    <div className="w-2 h-2 rounded-full bg-indigo-500 animate-pulse" />
                    <span className="text-[10px] font-black tracking-[0.4em] uppercase text-indigo-300">Stochastic Density Map</span>
                  </div>
                  <h3 className="text-7xl font-black text-white tracking-tighter uppercase italic leading-none">Bayesian <br /> Landscapes</h3>
                  <p className="text-slate-400 text-xl font-medium leading-relaxed max-w-2xl mx-auto lg:mx-0">
                    High-fidelity visualization of the Multivariate Bayesian Likelihood P(x|ω). Each surface maps the precise probability density using the calculated covariance matrix Σ.
                  </p>
                  <button onClick={() => setIs3DFocus(true)} className="px-14 py-7 bg-indigo-600 text-white rounded-[3rem] font-black text-sm uppercase tracking-[0.3em] shadow-[0_20px_40px_-10px_rgba(79,70,229,0.5)] hover:bg-indigo-500 hover:scale-105 transition-all flex items-center gap-5 mx-auto lg:mx-0 group">
                    <Maximize2 size={28} className="group-hover:rotate-12 transition" />
                    Engage 3D Simulation
                  </button>
                </div>

                {/* --- 2D VISUALIZATION (FULL WIDTH) --- */}
                <div className="lg:col-span-2 relative h-[500px] w-full bg-white rounded-[4rem] border border-slate-200 shadow-xl overflow-hidden flex flex-col">
                  <div className="p-8 border-b border-slate-100 flex justify-between items-center bg-slate-50">
                    <span className="text-xs font-black tracking-[0.3em] uppercase italic text-slate-500">2D Contour Analysis</span>
                    <div className="flex gap-2">
                      <span className="w-3 h-3 rounded-full bg-indigo-500"></span>
                      <span className="w-3 h-3 rounded-full bg-emerald-500"></span>
                    </div>
                  </div>
                  <div className="flex-1 bg-white">
                    <PlotlyChart
                      data={bayesianTraces.filter(t => t.type === 'contour')}
                      layout={{
                        xaxis: { visible: true, gridcolor: '#f1f5f9', color: '#64748b', title: { text: selectedFeatures[0], font: { size: 12, weight: 900 } }, zeroline: false },
                        yaxis: { visible: true, gridcolor: '#f1f5f9', color: '#64748b', title: { text: selectedFeatures[1], font: { size: 12, weight: 900 } }, zeroline: false },
                        margin: { l: 60, r: 20, b: 60, t: 20 },
                        paper_bgcolor: 'rgba(0,0,0,0)',
                        showlegend: true,
                        legend: { orientation: 'h', y: 1.1 }
                      }}
                      config={{ staticPlot: false, displayModeBar: true }}
                    />
                  </div>
                </div>

                {/* --- 3D VISUALIZATION (BOTTOM) --- */}
                <div className="lg:col-span-2 relative h-[500px] w-full bg-slate-900/40 rounded-[4rem] border border-white/5 shadow-inner overflow-hidden flex flex-col">
                  <div className="p-6 border-b border-white/5 flex justify-between items-center text-white/40">
                    <span className="text-[10px] font-black tracking-[0.3em] uppercase italic">3D Stochastic Manifolds</span>
                  </div>
                  <div className="flex-1">
                    <PlotlyChart
                      data={bayesianTraces.filter(t => t.type === 'surface')}
                      layout={{
                        scene: {
                          xaxis: { visible: true, gridcolor: '#334155', color: '#94a3b8', title: { text: selectedFeatures[0] || 'x1', font: { size: 10 } } },
                          yaxis: { visible: true, gridcolor: '#334155', color: '#94a3b8', title: { text: selectedFeatures[1] || 'x2', font: { size: 10 } } },
                          zaxis: { visible: true, gridcolor: '#334155', color: '#94a3b8', title: '' },
                          bgcolor: 'rgba(0,0,0,0)',
                          camera: { eye: { x: 1.5, y: 1.5, z: 0.5 } }
                        },
                        margin: { l: 0, r: 0, b: 0, t: 0 },
                        paper_bgcolor: 'rgba(0,0,0,0)',
                        showlegend: false,
                      }}
                      config={{ staticPlot: false, displayModeBar: true }}
                    />
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

      </main>

      {/* Footer System HUD */}
      {
        !is3DFocus && (
          <footer className="bg-white border-t border-slate-200 p-8 sticky bottom-0 z-40 shadow-[0_-30px_60px_-15px_rgba(0,0,0,0.08)]">
            <div className="max-w-7xl mx-auto flex flex-col md:flex-row justify-between items-center gap-10">
              <div className="flex items-center gap-10">
                <div className="flex items-center -space-x-4">
                  {COLORS.map((c, i) => <div key={i} className="w-7 h-7 rounded-full border-4 border-white shadow-xl" style={{ backgroundColor: c }} />)}
                </div>
                <div className="h-10 w-px bg-slate-100 hidden md:block" />
                <div>
                  <p className="text-[10px] font-black text-slate-300 uppercase tracking-widest">Active Model</p>
                  <div className="flex items-center gap-3 mt-1">
                    <Layout size={14} className="text-indigo-600" />
                    <span className="text-xs font-black text-slate-600 uppercase italic">
                      {selectedModel === 'bayes' ? 'Naive Bayesian Kernel v1' : 'Euclidean Distance Metrics'}
                    </span>
                  </div>
                </div>
              </div>
              <div className="flex items-center gap-6">
                <button onClick={() => {
                  const pages = Object.values(Page);
                  const currentIdx = pages.indexOf(currentPage);
                  const prevIdx = (currentIdx - 1 + pages.length) % pages.length;
                  setCurrentPage(pages[prevIdx]);
                }} className="px-10 py-5 text-[10px] font-black text-slate-400 bg-slate-50 rounded-[2rem] hover:bg-slate-100 transition uppercase tracking-[0.3em] border border-slate-100">Back</button>
                <button onClick={() => {
                  const pages = Object.values(Page);
                  const nextIdx = (pages.indexOf(currentPage) + 1) % pages.length;
                  setCurrentPage(pages[nextIdx]);
                }} className="px-14 py-5 bg-slate-900 text-white rounded-[2rem] font-black text-[10px] uppercase tracking-[0.4em] shadow-2xl hover:bg-black transition-all flex items-center gap-4">
                  Next Step <ChevronRight size={20} className="text-indigo-400" />
                </button>
              </div>
            </div>
          </footer>
        )
      }
    </div >
  );
};

export default App;
