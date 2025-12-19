
import { DataPoint, PCAResult, Metrics } from '../types';

/**
 * Calculates the mean vector of a dataset
 */
export const calculateMean = (data: number[][]): number[] => {
  if (!data.length) return [];
  const n = data.length;
  const p = data[0].length;
  const means = new Array(p).fill(0);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < p; j++) {
      means[j] += data[i][j];
    }
  }
  return means.map(m => m / n);
};

/**
 * Calculates the covariance matrix (Σ)
 */
export const calculateCovariance = (data: number[][], means: number[]): number[][] => {
  const n = data.length;
  const p = data[0].length;
  if (n < 2) return Array.from({ length: p }, () => new Array(p).fill(0));
  const cov = Array.from({ length: p }, () => new Array(p).fill(0));

  for (let i = 0; i < p; i++) {
    for (let j = 0; j < p; j++) {
      let sum = 0;
      for (let k = 0; k < n; k++) {
        sum += (data[k][i] - (means[i] || 0)) * (data[k][j] - (means[j] || 0));
      }
      cov[i][j] = sum / (n - 1);
    }
  }
  return cov;
};

/**
 * Simplified Jacobi eigenvalue algorithm for symmetric matrices
 */
export const eigenSolveSymmetric = (A: number[][]): { eigenvalues: number[], eigenvectors: number[][] } => {
  const n = A.length;
  let V: number[][] = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
  let D: number[][] = A.map(row => [...row]);
  const maxIterations = 100;

  for (let iter = 0; iter < maxIterations; iter++) {
    let maxVal = 0;
    let p = 0;
    let q = 1;

    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        if (Math.abs(D[i][j]) > maxVal) {
          maxVal = Math.abs(D[i][j]);
          p = i;
          q = j;
        }
      }
    }

    if (maxVal < 1e-9) break;

    const phi = 0.5 * Math.atan2(2 * D[p][q], D[q][q] - D[p][p]);
    const c = Math.cos(phi);
    const s = Math.sin(phi);

    const Jp_new = D[p][p] * c * c - 2 * D[p][q] * c * s + D[q][q] * s * s;
    const Jq_new = D[p][p] * s * s + 2 * D[p][q] * c * s + D[q][q] * c * c;
    D[p][q] = 0;
    D[q][p] = 0;
    D[p][p] = Jp_new;
    D[q][q] = Jq_new;

    for (let i = 0; i < n; i++) {
      if (i !== p && i !== q) {
        const Dip = D[i][p];
        const Diq = D[i][q];
        D[i][p] = D[p][i] = c * Dip - s * Diq;
        D[i][q] = D[q][i] = s * Dip + c * Diq;
      }
      const Vip = V[i][p];
      const Viq = V[i][q];
      V[i][p] = c * Vip - s * Viq;
      V[i][q] = s * Vip + c * Viq;
    }
  }

  const eigenvalues = D.map((_, i) => D[i][i]);
  const eigenvectors = V[0].map((_, colIndex) => V.map(row => row[colIndex]));
  const indices = eigenvalues.map((_, i) => i).sort((a, b) => eigenvalues[b] - eigenvalues[a]);
  
  return {
    eigenvalues: indices.map(i => eigenvalues[i]),
    eigenvectors: indices.map(i => eigenvectors[i])
  };
};

export const runPCA = (data: number[][]): PCAResult => {
  const means = calculateMean(data);
  const cov = calculateCovariance(data, means);
  const { eigenvalues, eigenvectors } = eigenSolveSymmetric(cov);
  
  const totalVariance = eigenvalues.reduce((a, b) => a + b, 0) || 1;
  const explainedVariance = eigenvalues.map(v => v / totalVariance);
  const cumulativeVariance: number[] = [];
  explainedVariance.reduce((acc, v, i) => {
    cumulativeVariance[i] = acc + v;
    return acc + v;
  }, 0);

  return {
    eigenvalues,
    eigenvectors,
    explainedVariance,
    cumulativeVariance,
    covarianceMatrix: cov
  };
};

export interface ClassStats {
  means: number[];
  vars: number[];
  prior: number;
}

/**
 * Gaussian Naive Bayes Classifier Implementation
 */
export class GaussianNaiveBayes {
  public classStats: Map<string, ClassStats> = new Map();
  public classes: string[] = [];

  fit(data: number[][], labels: string[]) {
    const classGroups = new Map<string, number[][]>();
    labels.forEach((label, i) => {
      if (!classGroups.has(label)) classGroups.set(label, []);
      classGroups.get(label)!.push(data[i]);
    });

    this.classes = Array.from(classGroups.keys());
    const totalCount = labels.length;

    classGroups.forEach((groupData, label) => {
      const n = groupData.length;
      const p = groupData[0].length;
      const means = calculateMean(groupData);
      const vars = new Array(p).fill(0);
      
      for (let j = 0; j < p; j++) {
        let sumSq = 0;
        for (let i = 0; i < n; i++) {
          sumSq += Math.pow(groupData[i][j] - means[j], 2);
        }
        vars[j] = (sumSq / n) + 1e-9;
      }

      this.classStats.set(label, { means, vars, prior: n / totalCount });
    });
  }

  predict(data: number[][]): string[] {
    return data.map(point => {
      let bestClass = '';
      let maxPost = -Infinity;

      this.classes.forEach(label => {
        const stats = this.classStats.get(label)!;
        let logPost = Math.log(stats.prior);

        for (let j = 0; j < point.length; j++) {
          const exponent = -Math.pow(point[j] - stats.means[j], 2) / (2 * stats.vars[j]);
          const logGaussian = exponent - 0.5 * Math.log(2 * Math.PI * stats.vars[j]);
          logPost += logGaussian;
        }

        if (logPost > maxPost) {
          maxPost = logPost;
          bestClass = label;
        }
      });

      return bestClass;
    });
  }
}

export const calculateMetrics = (actual: string[], predicted: string[]): Metrics => {
  const classes = Array.from(new Set(actual)).sort();
  const n = actual.length;
  let correct = 0;
  const cm = Array.from({ length: classes.length }, () => new Array(classes.length).fill(0));

  for (let i = 0; i < n; i++) {
    if (actual[i] === predicted[i]) correct++;
    const actualIdx = classes.indexOf(actual[i]);
    const predIdx = classes.indexOf(predicted[i]);
    if (actualIdx !== -1 && predIdx !== -1) cm[actualIdx][predIdx]++;
  }

  const accuracy = correct / n;
  
  let totalP = 0, totalR = 0;
  classes.forEach((cls, idx) => {
    const tp = cm[idx][idx];
    const fp = cm.reduce((sum, row, rIdx) => rIdx !== idx ? sum + row[idx] : sum, 0);
    const fn = cm[idx].reduce((sum, val, cIdx) => cIdx !== idx ? sum + val : sum, 0);
    
    totalP += tp / (tp + fp || 1);
    totalR += tp / (tp + fn || 1);
  });

  const precision = totalP / classes.length;
  const recall = totalR / classes.length;
  const f1 = 2 * (precision * recall) / (precision + recall || 1);

  return { accuracy, precision, recall, f1, confusionMatrix: cm };
};

export class MinimumDistanceClassifier {
  public classMeans: Map<string, number[]> = new Map();

  fit(data: number[][], labels: string[]) {
    const classGroups = new Map<string, number[][]>();
    labels.forEach((label, i) => {
      if (!classGroups.has(label)) classGroups.set(label, []);
      classGroups.get(label)!.push(data[i]);
    });

    classGroups.forEach((groupData, label) => {
      this.classMeans.set(label, calculateMean(groupData));
    });
  }

  predict(data: number[][]): string[] {
    return data.map(point => {
      let bestClass = '';
      let minDist = Infinity;

      this.classMeans.forEach((mean, label) => {
        const dist = point.reduce((sum, val, i) => sum + Math.pow(val - (mean[i] || 0), 2), 0);
        if (dist < minDist) {
          minDist = dist;
          bestClass = label;
        }
      });

      return bestClass;
    });
  }
}

export const getMultivariateGaussianPDF = (point: number[], mean: number[], cov: number[][]): number => {
  const k = point.length;
  // Simple 2x2 determinant and inverse for visualization speed
  if (k !== 2) return 0; // Only support 2D for viz

  const a = cov[0][0], b = cov[0][1];
  const c = cov[1][0], d = cov[1][1];
  const det = a * d - b * c;
  
  if (det <= 1e-9) return 0; // Singular

  const invDet = 1 / det;
  const invCov = [
    [d * invDet, -b * invDet],
    [-c * invDet, a * invDet]
  ];

  const dx = point[0] - mean[0];
  const dy = point[1] - mean[1];

  // (x-mu)^T * Sigma^-1 * (x-mu)
  const mahalanobisSq = 
    (dx * invCov[0][0] + dy * invCov[1][0]) * dx +
    (dx * invCov[0][1] + dy * invCov[1][1]) * dy;

  const normalization = 1 / (2 * Math.PI * Math.sqrt(det));
  
  return normalization * Math.exp(-0.5 * mahalanobisSq);
};
