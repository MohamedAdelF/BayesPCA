import express from 'express';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { existsSync } from 'fs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const app = express();
const port = process.env.PORT || 3000;

// Serve static files from dist directory
const distPath = join(__dirname, 'dist');
if (!existsSync(distPath)) {
    console.error('ERROR: dist directory not found! Please run "npm run build" first.');
    process.exit(1);
}

app.use(express.static(distPath));

// Handle all routes - serve index.html for SPA routing
app.get('*', (req, res) => {
    const indexPath = join(distPath, 'index.html');
    if (existsSync(indexPath)) {
        res.sendFile(indexPath);
    } else {
        res.status(404).send('Build files not found. Please run "npm run build" first.');
    }
});

app.listen(port, '0.0.0.0', () => {
    console.log(`🚀 Server running on port ${port}`);
    console.log(`📦 Serving files from: ${distPath}`);
});
