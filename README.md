# 🧠 Bayesian PCA Explorer

<div dir="rtl">

# مستكشف التحليل الإحصائي: PCA و Bayesian Classification

منصة تعليمية تفاعلية عالية الأداء مصممة لتصور وفهم العلاقة الرياضية بين **مصفوفة التباين المشترك (Σ)**، **تحليل المكونات الرئيسية (PCA)**، و**حدود التصنيف البايزي**.

</div>

A high-performance, interactive educational platform designed to visualize and decode the mathematical relationship between **Covariance (Σ)**, **Principal Component Analysis (PCA)**, and **Bayesian Classification Boundaries**.

Built with React, Vite, Recharts, and Plotly, this tool provides a premium "Glassmorphic" UI for exploring high-dimensional data in real-time.

---

<div dir="rtl">

## 📋 وصف المشروع

هذا المشروع يطبق متطلبات مشروع المقرر الدراسي في تعلم الآلة، ويتضمن:

1. ✅ **جمع البيانات**: مجموعات بيانات جاهزة (Wine, Iris, Cancer) + رفع ملفات CSV مخصصة
2. ✅ **تطبيق نموذج تصنيف**: Gaussian Naive Bayes و Minimum Distance Classifier
3. ✅ **حساب مصفوفة التباين المشترك**: لكل فئة على حدة
4. ✅ **إيجاد القيم الذاتية والمتجهات الذاتية**: من مصفوفة التباين المشترك
5. ✅ **اختيار أهم الميزات**: بناءً على تحليل PCA
6. ✅ **إعادة التصنيف**: باستخدام الميزات المختارة
7. ✅ **مقارنة النتائج**: بين التصنيف الكامل والتصنيف بالميزات المختارة
8. ✅ **رسم توزيع الاحتمالات**: باستخدام أهم ميزتين

</div>

---

## 🚀 المميزات / Features

### 1. إدارة البيانات / Data Management

<div dir="rtl">

- **مجموعات بيانات جاهزة**: وصول فوري لمجموعات بيانات كلاسيكية:
  - 🍷 **جودة النبيذ** (تحليل كيميائي)
  - 🌸 **زهور السوسن** (قياسات نباتية)
  - 🧬 **سرطان الثدي** (مقاييس تشخيصية)
- **رفع ملفات مخصصة**: سحب وإفلات ملفات CSV الخاصة بك
- **التوحيد التلقائي**: تطبيع Z-Score تلقائي لضمان تصورات ثلاثية الأبعاد قوية بغض النظر عن مقياس البيانات

</div>

- **Prebuilt Datasets**: Instant access to classic datasets:
  - 🍷 **Wine Quality** (Chemical analysis)
  - 🌸 **Iris Flowers** (Botanical measurements)
  - 🧬 **Breast Cancer** (Diagnostic metrics)
- **Custom Uploads**: Drag & Drop your own CSV files
- **Auto-Standardization**: Automatic Z-Score normalization to ensure robust 3D visualizations regardless of data scale

---

### 2. التصنيف الأساسي / Baseline Classification

<div dir="rtl">

- تقييم أداء النموذج على **مجموعة الميزات الكاملة** قبل تقليل الأبعاد
- **مصفوفة الارتباك**: خريطة حرارية تفاعلية لتصور الفئات الحقيقية مقابل المتوقعة
- **المقاييس**: حساب دقيق في الوقت الفعلي للدقة (Accuracy)، الدقة (Precision)، الاستدعاء (Recall)، و F1-Score
- **الخوارزميات**: التبديل بين **Gaussian Naive Bayes** و **Minimum Distance Classifier**

</div>

- Evaluate model performance on the **full feature set** before dimensionality reduction
- **Confusion Matrix**: Interactive heatmap to visualize True vs Predicted classes
- **Metrics**: Accuracy, Precision, Recall, and F1-Score real-time calculation
- **Algorithms**: Switch between **Gaussian Naive Bayes** and **Minimum Distance Classifier**

---

### 3. مصفوفة التباين المشترك (Σ) / Covariance Matrix

<div dir="rtl">

- تعمق في علاقات الميزات مع **مصفوفة تفاعلية على شكل خريطة حرارية**
- **عناوين ثابتة**: التنقل بسهولة في المصفوفات الكبيرة مع تثبيت تسميات الصفوف/الأعمدة
- **كثافة الارتباط**: ترميز لوني بصري (أحمر: سالب، أزرق: موجب) لاكتشاف الأنماط فوراً

</div>

- Deep dive into feature relationships with a **heatmap-styled interactive matrix**
- **Sticky Headers**: Easily navigate large matrices with pinned row/column labels
- **Correlation Intensity**: Visual color coding (`Red`: Negative, `Blue`: Positive) to spot patterns instantly

---

### 4. تحليل القيم الذاتية (PCA) / Eigen Analysis

<div dir="rtl">

- **إسقاط ثلاثي وثنائي الأبعاد**: تصور البيانات عالية الأبعاد المسقطة على المكونات الرئيسية (PC1, PC2, PC3)
- **رسم تفاعلي ثلاثي الأبعاد**: تدوير، تكبير، واستكشاف متعدد البيانات في بيئة ثلاثية الأبعاد عالية الجودة
- **التباين الموضح**: رسوم بيانية تظهر تأثير كل مكون
- **عرض قوي**: يتعامل مع الحالات الخاصة (مثل أقل من 3 مكونات) بسلاسة

</div>

- **3D & 2D Projection**: Visualize high-dimensional data projected onto the top Principal Components (PC1, PC2, PC3)
- **Interactive 3D Plot**: Rotate, zoom, and explore the data manifold in a cinema-grade 3D environment
- **Variance Explained**: Scree plots showing the impact of each component
- **Robust Rendering**: Handles edge cases (e.g., fewer than 3 components) gracefully

---

### 5. هندسة الاحتمالية / Likelihood Geometry

<div dir="rtl">

- تصور **دوال كثافة الاحتمال الغوسية متعددة المتغيرات (PDF)**
- **رسوم كفافية**: حدود قرار ثنائية الأبعاد
- **رسوم سطحية ثلاثية الأبعاد**: عرض "الجبال" من كثافة الاحتمال لكل فئة

</div>

- Visualize the **Gaussian Probability Density Functions (PDF)**
- **Contour Plots**: 2D decision boundaries
- **3D Surface Plots**: View the "mountains" of probability density for each class

---

## 🛠️ التقنيات المستخدمة / Technology Stack

- **Core**: [React 19](https://react.dev/), TypeScript, [Vite](https://vitejs.dev/)
- **Visualization**: 
  - [Plotly.js](https://plotly.com/javascript/) (3D & Surfaces)
  - [Recharts](https://recharts.org/) (2D Analytics)
- **Styling**: Tailwind CSS (Custom Glassmorphism Design System)
- **Deployment**: Node.js/Express (Ready for Railway/Vercel)

---

## 📦 التثبيت والاستخدام / Installation & Usage

<div dir="rtl">

### 1. استنساخ المستودع
```bash
git clone https://github.com/MohamedAdelF/BayesPCA.git
cd BayesPCA
```

### 2. تثبيت الحزم
```bash
npm install
```

### 3. التشغيل محلياً
```bash
npm run dev
```

### 4. البناء للإنتاج
```bash
npm run build
npm start
```

</div>

### 1. Clone the repository
```bash
git clone https://github.com/MohamedAdelF/BayesPCA.git
cd BayesPCA
```

### 2. Install Dependencies
```bash
npm install
```

### 3. Run Locally
```bash
npm run dev
```

### 4. Build for Production
```bash
npm run build
npm start
```

---

## ☁️ النشر على Railway / Deployment

<div dir="rtl">

المشروع جاهز للنشر على **Railway**.

### خطوات النشر السريعة:

1. **رفع على GitHub**
   ```bash
   git add .
   git commit -m "Ready for Railway deployment"
   git push origin main
   ```

2. **النشر على Railway**
   - اذهب إلى [Railway.app](https://railway.app)
   - اضغط "New Project" → "Deploy from GitHub repo"
   - اختر المستودع الخاص بك
   - Railway سيقوم تلقائياً بـ:
     - اكتشاف مشروع Node.js
     - تشغيل `npm install`
     - تشغيل `npm run build` (من railway.json)
     - تشغيل `npm start` لبدء السيرفر

3. **متغيرات البيئة** (اختياري)
   - إذا كنت تحتاج مفاتيح API، أضفها في لوحة Railway تحت "Variables"
   - التطبيق سيستخدم `PORT` تلقائياً (Railway يضبط هذا)

### البناء والاختبار محلياً:
```bash
npm run build
npm start
# السيرفر سيعمل على http://localhost:3000
```

### إعدادات Railway:
- أمر البناء: `npm run build` (يتم اكتشافه تلقائياً)
- أمر البدء: `npm start` (من package.json)
- المنفذ: يتم ضبطه تلقائياً من قبل Railway عبر متغير البيئة `PORT`

</div>

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

## 📚 المتطلبات المنجزة / Project Requirements

<div dir="rtl">

✅ **1. جمع البيانات**: مجموعات بيانات جاهزة + رفع CSV مخصص  
✅ **2. تطبيق نموذج ML**: Gaussian Naive Bayes / Minimum Distance Classifier  
✅ **3. حساب مصفوفة التباين المشترك**: لكل فئة على حدة  
✅ **4. إيجاد القيم الذاتية والمتجهات الذاتية**: من مصفوفة التباين  
✅ **5. اختيار أهم الميزات**: مع شرح أسباب الاختيار  
✅ **6. إعادة التصنيف**: باستخدام الميزات المختارة  
✅ **7. مقارنة النتائج**: بين التصنيف الكامل والمختصر  
✅ **8. رسم توزيع الاحتمالات**: باستخدام أهم ميزتين  

</div>

✅ **1. Collect Dataset**: Prebuilt datasets + custom CSV upload  
✅ **2. Apply ML Classifier**: Gaussian Naive Bayes / Minimum Distance Classifier  
✅ **3. Compute Covariance Matrix**: Per class  
✅ **4. Find Eigenvalues & Eigenvectors**: From covariance matrix  
✅ **5. Select Important Features**: With explanation  
✅ **6. Re-classify**: Using selected features  
✅ **7. Compare Results**: Baseline vs optimized  
✅ **8. Sketch Probability Distribution**: Using top 2 features  

---

## 📝 الترخيص / License

MIT License. Free for educational and research use.

---

<div dir="rtl">

## 👤 المطور / Developer

**Mohamed Adel**  
[GitHub Profile](https://github.com/MohamedAdelF)

---

## 🔗 الروابط / Links

- **المستودع**: [https://github.com/MohamedAdelF/BayesPCA](https://github.com/MohamedAdelF/BayesPCA)
- **النشر**: متاح على Railway بعد النشر

</div>

## 👤 Developer

**Mohamed Adel**  
[GitHub Profile](https://github.com/MohamedAdelF)

---

## 🔗 Links

- **Repository**: [https://github.com/MohamedAdelF/BayesPCA](https://github.com/MohamedAdelF/BayesPCA)
- **Deployment**: Available on Railway after deployment
