
export interface DataPoint {
  [key: string]: number | string;
}

export interface PCAResult {
  eigenvalues: number[];
  eigenvectors: number[][];
  explainedVariance: number[];
  cumulativeVariance: number[];
  covarianceMatrix: number[][];
}

export interface Metrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  confusionMatrix: number[][];
}

export enum Page {
  UPLOAD = 'upload',
  METRICS = 'metrics',
  COVARIANCE = 'covariance',
  PCA = 'pca',
  SELECTION = 'selection',
  DISTRIBUTION = 'distribution'
}

export interface ClassificationResult {
  predictions: string[] | number[];
  metrics: Metrics;
}
