# BCI Classifier Website

Professional website for the BCI Classifier project - a machine learning system for predicting Brain-Computer Interface decoder performance.

## Overview

This website provides:
- **Home**: Project overview and pipeline architecture
- **Methodology**: Detailed explanation of the four-phase approach
- **Results**: Interactive dashboard for viewing ground truth labels
- **Documentation**: Setup guides, API reference, and troubleshooting

## Features

- 🎨 Modern, responsive design
- 📊 Interactive results dashboard
- 📈 Performance distribution visualization
- 💾 JSON data loading and display
- 📋 Code copy-to-clipboard functionality
- 🔔 Toast notifications
- ♿ Accessibility-friendly

## Usage

### Local Development

Simply open the HTML files in a web browser:

```bash
# Open the homepage
open website/index.html

# Or use a simple HTTP server
cd website
python -m http.server 8000
# Then visit http://localhost:8000
```

### Viewing Results

1. Run Phase 1 ground truth generation:
   ```bash
   python src/generate_ground_truth_labels.py
   ```

2. Open the Results page in your browser:
   ```bash
   open website/results.html
   ```

3. Click "Load Results" to view the data

## File Structure

```
website/
├── index.html           # Homepage
├── methodology.html     # Methodology page
├── results.html         # Results dashboard
├── documentation.html   # Documentation
├── css/
│   └── style.css       # All styling
├── js/
│   ├── main.js         # Main JavaScript utilities
│   └── results.js      # Results page functionality
└── assets/             # Images and other assets (future)
```

## Technologies

- **HTML5**: Semantic markup
- **CSS3**: Custom styling with CSS variables
- **Vanilla JavaScript**: No frameworks, lightweight and fast
- **Responsive Design**: Mobile-friendly layout

## Customization

### Colors

Edit CSS variables in `css/style.css`:

```css
:root {
    --primary-color: #2563eb;      /* Primary brand color */
    --secondary-color: #7c3aed;    /* Secondary accent */
    --accent-color: #10b981;       /* Success/accent color */
    --dark-bg: #1e293b;            /* Dark background */
    --light-bg: #f8fafc;           /* Light background */
}
```

### Content

Edit the HTML files directly to modify content, sections, or layout.

## Browser Support

- Chrome/Edge (latest)
- Firefox (latest)
- Safari (latest)
- Mobile browsers

## Performance

- Lightweight: ~50KB total (HTML + CSS + JS)
- No external dependencies
- Fast load times
- Optimized for performance

## Future Enhancements

- [ ] Add charts for Phase 2-4 results
- [ ] Interactive feature importance plots
- [ ] Real-time processing status updates
- [ ] Export results to PDF
- [ ] Dark mode toggle
- [ ] Advanced filtering and sorting

## Contributing

To add new pages or features:

1. Create HTML file with consistent navigation
2. Add styles to `css/style.css`
3. Add JavaScript to `js/` directory
4. Update this README

## License

Part of the BCI-Classifer project.
