# Bayesian PCA Explorer

A high-performance, interactive educational platform designed to visualize and decode the mathematical relationship between **Covariance (Σ)**, **Principal Component Analysis (PCA)**, and **Bayesian Classification Boundaries**.

Built with React, Vite, Recharts, and Plotly, this tool provides a premium "Glassmorphic" UI for exploring high-dimensional data in real-time.

## 🚀 Features

### 1. Data Management
- **Prebuilt Datasets**: Instant access to classic datasets:
  - 🍷 **Wine Quality** (Chemical analysis)
  - 🌸 **Iris Flowers** (Botanical measurements)
  - 🧬 **Breast Cancer** (Diagnostic metrics)
- **Custom Uploads**: Drag & Drop your own CSV files.
- **Auto-Standardization**: Automatic Z-Score normalization to ensure robust 3D visualizations regardless of data scale.

### 2. Baseline Classification (Ground Truth)
- Evaluate model performance on the **full feature set** before dimensionality reduction.
- **Confusion Matrix**: Interactive heatmap to visualize True vs Predicted classes.
- **Metrics**: Accuracy, Precision, Recall, and F1-Score real-time calculation.
- **Algorithms**: Switch between **Gaussian Naive Bayes** and **Minimum Distance Classifier**.

### 3. Covariance Matrix (Σ)
- Deep dive into feature relationships with a **heatmap-styled interactive matrix**.
- **Sticky Headers**: Easily navigate large matrices with pinned row/column labels.
- **Correlation Intensity**: Visual color coding (`Red`: Negative, `Blue`: Positive) to spot patterns instantly.

### 4. Eigen Analysis (PCA)
- **3D & 2D Projection**: Visualize high-dimensional data projected onto the top Principal Components (PC1, PC2, PC3).
- **Interactive 3D Plot**: Rotate, zoom, and explore the data manifold in a cinema-grade 3D environment.
- **Variance Explained**: Scree plots showing the impact of each component.
- **Robust Rendering**: Handles edge cases (e.g., fewer than 3 components) gracefully.

### 5. Likelihood Geometry
- Visualize the **Gaussian Probability Density Functions (PDF)**.
- **Contour Plots**: 2D decision boundaries.
- **3D Surface Plots**: View the "mountains" of probability density for each class.

---

## 🛠️ Technology Stack

- **Core**: [React 19](https://react.dev/), TypeScript, [Vite](https://vitejs.dev/)
- **Visualization**: 
  - [Plotly.js](https://plotly.com/javascript/) (3D & Surfaces)
  - [Recharts](https://recharts.org/) (2D Analytics)
- **Styling**: Tailwind CSS (Custom Glassmorphism Design System)
- **Deployment**: Node.js/Express (Ready for Railway/Vercel)

---

## 📦 Installation & Usage

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/bayesian-pca-explorer.git
   cd bayesian-pca-explorer
   ```

2. **Install Dependencies**
   ```bash
   npm install
   ```

3. **Run Locally**
   ```bash
   npm run dev
   ```

4. **Build for Production**
   ```bash
   npm run build
   npm start
   ```

---

## ☁️ Deployment (Railway)

The project is pre-configured for **Railway**.

### Quick Deploy Steps:

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Ready for Railway deployment"
   git push origin main
   ```

2. **Deploy on Railway**
   - Go to [Railway.app](https://railway.app)
   - Click "New Project" → "Deploy from GitHub repo"
   - Select your repository
   - Railway will automatically:
     - Detect Node.js project
     - Run `npm install`
     - Run `npm run build` (from railway.json)
     - Run `npm start` to launch the server

3. **Environment Variables** (Optional)
   - If you need API keys, add them in Railway dashboard under "Variables"
   - The app will use `PORT` automatically (Railway sets this)

### Manual Build & Test Locally:
```bash
npm run build
npm start
# Server will run on http://localhost:3000
```

### Railway Configuration:
- Build Command: `npm run build` (auto-detected)
- Start Command: `npm start` (from package.json)
- Port: Automatically set by Railway via `PORT` environment variable

---

## 📝 License

MIT License. Free for educational and research use.
