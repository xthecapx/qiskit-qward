# qWard Presentation - ChileCon 2025

A professional, projector-optimized presentation for the qWard quantum computing toolkit, built with [reveal.js](https://revealjs.com).

## 📁 File Structure

```
chilecon/
├── index.html                 # Main reveal.js presentation file
├── theme-qward.css            # Custom quantum-inspired theme
├── package.json               # npm configuration
├── node_modules/              # reveal.js and dependencies
├── README.md                  # This file
├── PLAN.md                    # Presentation planning document
└── *.png                      # Images and diagrams
```

## 🚀 Quick Start

### Installation

```bash
# Install dependencies (reveal.js)
npm install

# Start the presentation server
npm start
```

The presentation will automatically open in your browser at `http://localhost:8000`

### Building for Production

```bash
# Build the presentation for deployment
npm run build
```

This will create a `dist/` folder with all necessary files bundled and ready for deployment.

### GitHub Pages Deployment

The presentation is automatically deployed to GitHub Pages when changes are pushed to the `main` branch:

1. **Automatic Deployment**: Push changes to `docs/papers/chilecon/` in the main branch
2. **GitHub Actions**: The workflow builds and deploys automatically
3. **Access**: Visit your GitHub Pages URL (e.g., `https://username.github.io/qiskit-qward/`)

**Manual Deployment**:
```bash
# Build the presentation
npm run build

# The dist/ folder can be deployed to any static hosting service
```

## 🎨 Features

### Quantum-Inspired Color Palette
Inspired by quantum computing aesthetics with high projector visibility:
- **Dark Background**: `#0f1419` with subtle gradient overlays
- **Vibrant Accents**:
  - Cyan (`#00d4ff`) - Primary accent, gradient headers
  - Blue (`#58a6ff`) - Secondary elements
  - Purple (`#bc8cff`) - Special emphasis
  - Amber (`#fbbf24`) - Highlights and insights
  - Red (`#ff6b6b`) - Alerts

### Reveal.js Power Features
Built on reveal.js for professional presentations:
- **Smooth Transitions**: Slide, fade, and zoom effects
- **Speaker Notes**: Press `S` for speaker view
- **Overview Mode**: Press `Esc` for slide overview
- **PDF Export**: Add `?print-pdf` to URL for printing
- **Auto-Slide**: Optional automatic progression
- **Progress Bar**: Visual indication of presentation progress

### Navigation
- **Arrow Keys**: Navigate between slides
- **Space**: Next slide
- **Shift+Space**: Previous slide
- **N / P**: Next/Previous
- **Home / End**: First/Last slide
- **Overview**: ESC or O
- **Speaker View**: S
- **Fullscreen**: F

## 🎨 Customization

### Modifying the Theme
The presentation uses the built-in **serif theme** with custom overrides in `custom-overrides.css`:

**To change the base theme**, edit `index.html` and replace `serif.css` with any of:
- `black.css` - Dark theme (default)
- `white.css` - Clean white
- `league.css` - Gray background
- `beige.css` - Warm beige
- `night.css` - Black with thick text
- `moon.css` - Dark blue
- `sky.css` - Blue background
- `blood.css` - Dark with red links
- `solarized.css` - Solarized colors
- `dracula.css` - Popular dark theme

**To customize styling**, edit `custom-overrides.css`:
- Typography sizes (via CSS variables)
- Colors for highlights and accents
- Component styles (cards, grids, boxes)
- Spacing and layout

### Adjusting Content
Edit `index.html` to:
- Add/remove slides (wrap content in `<section>` tags)
- Update text and images
- Modify slide layouts
- Add fragments for progressive disclosure

### Reveal.js Configuration
Edit the `Reveal.initialize()` call in `index.html` to:
- Change transitions
- Enable/disable controls
- Add plugins
- Configure timing

See [reveal.js documentation](https://revealjs.com/config/) for all options.

## 🎯 CSS Variables for Customization

The `custom-overrides.css` file uses [reveal.js CSS variables](https://github.com/hakimel/reveal.js/blob/master/css/theme/template/exposer.scss) for easy customization:

### Typography
```css
--r-heading1-size: 2.2em;         /* Main title size */
--r-heading2-size: 1.8em;         /* Slide title size */
--r-heading3-size: 1.4em;         /* Section heading */
--r-heading4-size: 1.1em;         /* Subsection */
--r-main-font-size: 38px;         /* Body text size */
--r-main-line-height: 1.6;        /* Line spacing */
```

### Custom Accent Colors
```css
--color-cyan: #006b7d;            /* Primary accent */
--color-blue: #2563eb;            /* Secondary accent */
--color-purple: #7c3aed;          /* Special emphasis */
--color-amber: #d97706;           /* Highlights */
--color-red: #dc2626;             /* Alerts */
--color-green: #059669;           /* Success */
```

### Usage in HTML
Use utility classes for colored text:
```html
<strong class="highlight-cyan">Important point</strong>
<strong class="highlight-amber">Warning</strong>
```

## 📊 Slide Structure

The presentation includes:
1. Title slide
2. Agenda
3. Introduction & Motivation
4. Quantum Programming Journey
5. Research Questions
6. Methodology
7. Findings (RQ1, RQ2, RQ3)
8. qWard Solution
9. Architecture diagrams
10. Practical example
11. Results (pre-runtime & post-runtime metrics)
12. Conclusion & Future Work
13. Q&A
14. Backup slides (for detailed Q&A)

## 🔧 Technical Details

### Dependencies
- **reveal.js 5.2.1**: Modern HTML presentation framework
- **Google Fonts**: Inter for UI, Fira Code for code blocks
- **Node.js**: For development server

### Browser Compatibility
- Chrome/Edge 90+
- Firefox 88+
- Safari 14+
- Modern mobile browsers

### Performance
- Lazy loading of images
- Hardware-accelerated transitions
- Optimized for large presentations
- Progressive enhancement

## 📝 Tips & Best Practices

### Speaker View
Press `S` during presentation to open speaker view in a new window. This shows:
- Current slide
- Next slide preview
- Speaker notes
- Timer

### PDF Export
Add `?print-pdf` to the URL and print to PDF:
```
http://localhost:8000/?print-pdf
```

### Adding Speaker Notes
Add notes to any slide:
```html
<section>
    <h2>Slide Title</h2>
    <p>Content here...</p>
    <aside class="notes">
        These are speaker notes - only visible in speaker view
    </aside>
</section>
```

### Progressive Disclosure (Fragments)
Reveal content step by step:
```html
<ul>
    <li class="fragment">Appears first</li>
    <li class="fragment">Appears second</li>
    <li class="fragment">Appears third</li>
</ul>
```

## 🤝 Contributing

To modify this presentation:
1. Edit `index.html` for content
2. Edit `theme-qward.css` for styling
3. Test with `npm start`
4. Verify in speaker view and PDF export mode

## 📄 License

Part of the qWard project. See main project license for details.

---

**For questions or issues**: Contact the qWard team
**GitHub**: xthecapx/qiskit-qward
**PyPI**: qiskit-qward

