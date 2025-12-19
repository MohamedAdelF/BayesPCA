
import { GoogleGenAI } from "@google/genai";
import { Metrics, PCAResult } from "../types";

const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

export const getInsight = async (
  metrics: Metrics, 
  pca: PCAResult, 
  featureNames: string[], 
  targetName: string
) => {
  const prompt = `
    Analyze the following machine learning results for a dataset with target variable "${targetName}".
    
    Model Performance:
    - Accuracy: ${(metrics.accuracy * 100).toFixed(2)}%
    - Precision: ${(metrics.precision * 100).toFixed(2)}%
    - Recall: ${(metrics.recall * 100).toFixed(2)}%
    
    PCA Analysis:
    - Top 3 Features contribution: ${pca.explainedVariance.slice(0, 3).map(v => (v * 100).toFixed(1) + '%').join(', ')}
    - Total variance explained by top 3 components: ${(pca.cumulativeVariance[2] * 100).toFixed(2)}%
    
    Data Context:
    The features are: ${featureNames.join(', ')}.
    
    Task:
    Provide a concise technical summary (max 150 words) suitable for a "Cognitive Science" or "Data Science" project. 
    Explain if PCA is effective here and what the eigenvalues suggest about feature redundancy. 
    Mention the relationship between the Covariance matrix and Bayesian Decision Surfaces (Lecture 9 & 10 topics).
  `;

  try {
    // Fix: Use gemini-3-pro-preview for complex text tasks involving advanced reasoning and math
    const response = await ai.models.generateContent({
      model: 'gemini-3-pro-preview',
      contents: prompt,
    });
    // Fix: Directly access .text property from GenerateContentResponse
    return response.text;
  } catch (error) {
    console.error("Gemini Error:", error);
    return "Could not generate AI insights at this time.";
  }
};
