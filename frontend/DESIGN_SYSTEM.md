# Design System

A comprehensive design system inspired by [grainrad.com](https://grainrad.com) - a dark, minimalist, utility-focused aesthetic designed for professional audio/visual tools.

## Design Philosophy

**Core Principles:**
1. **Radical Simplicity** - Remove everything that doesn't serve the user's primary goal
2. **Information Hierarchy** - Most important information gets the most visual weight through brightness
3. **Purposeful Color** - Color indicates meaning, not decoration (pure grayscale with rare accent)
4. **Breathing Room** - White space creates clarity
5. **Professional Utility** - Every element feels like part of a serious tool

**Aesthetic Direction:**
- Terminal/CLI aesthetic with monospace typography
- Near-pure black backgrounds
- High-contrast text hierarchy using grays
- Minimal borders (very dark gray)
- No gradients, shadows, or decorative elements
- Clean, functional UI components

---

## Color System

### Core Palette (CSS Custom Properties)

```css
:root {
  /* Backgrounds */
  --background: 0 0% 4%;           /* #0a0a0a - Page background */
  --background-elevated: 0 0% 6%;  /* #0f0f0f - Sidebar, panels */
  --background-card: 0 0% 8%;      /* #141414 - Cards, inputs */

  /* Text Hierarchy (brightest = most important) */
  --foreground: 0 0% 53%;          /* #888888 - Primary text */
  --foreground-bright: 0 0% 80%;   /* #cccccc - Headings, emphasis */
  --foreground-muted: 0 0% 33%;    /* #555555 - Secondary text, labels */
  --foreground-subtle: 0 0% 20%;   /* #333333 - Tertiary, disabled */

  /* Borders */
  --border: 0 0% 13%;              /* #222222 - Primary border */
  --border-subtle: 0 0% 10%;       /* #1a1a1a - Subtle dividers */

  /* Interactive */
  --accent: 0 0% 80%;              /* #cccccc - Active states, indicators */
  --accent-muted: 0 0% 53%;        /* #888888 - Hover states */

  /* Status (use sparingly) */
  --success: 142 76% 36%;          /* Green for success states */
  --warning: 38 92% 50%;           /* Amber for warnings */
  --destructive: 0 84% 60%;        /* Red for errors/destructive */
}
```

### Hex Reference Chart

| Token | Hex | Usage |
|-------|-----|-------|
| `--background` | `#0a0a0a` | Page background |
| `--background-elevated` | `#0f0f0f` | Sidebars, elevated panels |
| `--background-card` | `#141414` | Cards, input backgrounds |
| `--foreground` | `#888888` | Primary body text |
| `--foreground-bright` | `#cccccc` | Headings, active items |
| `--foreground-muted` | `#555555` | Labels, secondary text |
| `--foreground-subtle` | `#333333` | Disabled, tertiary |
| `--border` | `#222222` | Primary borders |
| `--border-subtle` | `#1a1a1a` | Subtle dividers |

### Color Usage Guidelines

**DO:**
- Use `foreground-bright` (#cccccc) for section headers and active navigation
- Use `foreground` (#888888) for primary content text
- Use `foreground-muted` (#555555) for labels and secondary information
- Use borders sparingly - only where needed to separate sections

**DON'T:**
- Never use pure white (#ffffff) for text - it's too harsh
- Avoid colored text except for status indicators
- Don't use background colors for hover states on dark elements
- Never use gradients or shadows for depth

---

## Typography

### Font Stack

```css
font-family: "IBM Plex Mono", "JetBrains Mono", "Fira Code", "SF Mono", monospace;
```

**Font Source:** Import via Google Fonts or next/font/google

```tsx
import { IBM_Plex_Mono } from 'next/font/google';

const ibmPlexMono = IBM_Plex_Mono({
  weight: ['400', '500', '600'],
  subsets: ['latin'],
  variable: '--font-mono',
});
```

### Type Scale

| Level | Size | Weight | Color | Usage |
|-------|------|--------|-------|-------|
| Heading 1 | 18px | 400 | `--foreground-bright` | Page titles |
| Heading 2 | 16px | 400 | `--foreground-bright` | Section headers |
| Heading 3 | 14px | 400 | `--foreground` | Subsection headers |
| Body | 14px | 400 | `--foreground` | Primary content |
| Label | 12px | 400 | `--foreground-muted` | Form labels, metadata |
| Caption | 10px | 400 | `--foreground-subtle` | Timestamps, hints |

### Tailwind Classes

```html
<!-- Headings -->
<h1 class="text-lg font-normal text-foreground-bright">Page Title</h1>
<h2 class="text-base font-normal text-foreground-bright">Section</h2>

<!-- Body text -->
<p class="text-sm text-foreground">Primary content</p>
<span class="text-xs text-muted-foreground">Label text</span>

<!-- Captions -->
<span class="text-[10px] text-foreground-subtle">Hint text</span>
```

---

## Spacing System

### Base Scale

Use multiples of 4px (Tailwind's default scale):

| Name | Value | Tailwind | Usage |
|------|-------|----------|-------|
| xs | 4px | `p-1`, `gap-1` | Icon spacing, tight elements |
| sm | 8px | `p-2`, `gap-2` | Form element internal padding |
| md | 12px | `p-3`, `gap-3` | Card internal padding |
| lg | 16px | `p-4`, `gap-4` | Section spacing |
| xl | 24px | `p-6`, `gap-6` | Major section gaps |
| 2xl | 32px | `p-8`, `gap-8` | Page-level spacing |

### Component Spacing

```css
/* Sidebar */
.sidebar {
  padding: 16px 0;
  width: 280px;
}

/* Section within sidebar */
.section {
  padding: 0 16px;
  margin-bottom: 16px;
}

/* Form row (label + control) */
.form-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px 0;
}

/* Slider track padding */
.slider-container {
  padding: 4px 0;
}
```

---

## Component Patterns

### Collapsible Section

The signature UI pattern from grainrad - sections with +/- toggle indicators.

```tsx
interface CollapsibleSectionProps {
  title: string;
  defaultOpen?: boolean;
  children: React.ReactNode;
}

function CollapsibleSection({ title, defaultOpen = false, children }: CollapsibleSectionProps) {
  const [isOpen, setIsOpen] = useState(defaultOpen);

  return (
    <div className="border-b border-border">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center justify-between py-3 px-4 text-foreground-bright hover:text-foreground-bright/80"
      >
        <span className="text-sm font-normal">{title}</span>
        <span className="text-foreground-muted">{isOpen ? '-' : '+'}</span>
      </button>
      {isOpen && (
        <div className="px-4 pb-4">
          {children}
        </div>
      )}
    </div>
  );
}
```

### Form Controls

#### Slider

```tsx
function Slider({ label, value, min, max, unit = '' }: SliderProps) {
  return (
    <div className="flex items-center justify-between gap-4 py-2">
      <span className="text-sm text-foreground-muted flex-shrink-0">{label}</span>
      <span className="text-sm text-foreground w-12 text-right">{value}{unit}</span>
      <input
        type="range"
        min={min}
        max={max}
        value={value}
        className="flex-1 h-1 bg-border rounded-full appearance-none cursor-pointer
                   [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3
                   [&::-webkit-slider-thumb]:bg-foreground [&::-webkit-slider-thumb]:rounded-full
                   [&::-webkit-slider-thumb]:appearance-none"
      />
    </div>
  );
}
```

#### Checkbox

```tsx
function Checkbox({ label, checked, onChange }: CheckboxProps) {
  return (
    <div className="flex items-center justify-between py-2">
      <span className="text-sm text-foreground-muted">{label}</span>
      <input
        type="checkbox"
        checked={checked}
        onChange={onChange}
        className="w-4 h-4 bg-background-card border border-border rounded
                   checked:bg-foreground checked:border-foreground
                   focus:ring-0 focus:ring-offset-0 cursor-pointer"
      />
    </div>
  );
}
```

#### Select/Dropdown

```tsx
function Select({ label, value, options }: SelectProps) {
  return (
    <div className="flex items-center justify-between py-2">
      <span className="text-sm text-foreground-muted">{label}</span>
      <select
        value={value}
        className="bg-background-card border border-border text-foreground-bright
                   text-sm py-1.5 px-3 rounded focus:outline-none focus:border-foreground-muted"
      >
        {options.map(opt => (
          <option key={opt.value} value={opt.value}>{opt.label}</option>
        ))}
      </select>
    </div>
  );
}
```

#### Text Input

```tsx
function TextInput({ label, value, placeholder }: TextInputProps) {
  return (
    <div className="flex items-center justify-between py-2 gap-4">
      <span className="text-sm text-foreground-muted flex-shrink-0">{label}</span>
      <input
        type="text"
        value={value}
        placeholder={placeholder}
        className="flex-1 bg-background border border-border text-foreground
                   text-sm py-1.5 px-3 rounded focus:outline-none focus:border-foreground-muted"
      />
    </div>
  );
}
```

### Card/Panel Selection Grid

For format selection (like export formats):

```tsx
function SelectionGrid({ options, selected, onSelect }: SelectionGridProps) {
  return (
    <div className="grid grid-cols-2 gap-px border border-border rounded overflow-hidden">
      {options.map(option => (
        <button
          key={option.id}
          onClick={() => onSelect(option.id)}
          className={cn(
            "p-3 text-left transition-colors",
            selected === option.id
              ? "bg-foreground-muted/20 text-foreground-bright"
              : "bg-background-card text-foreground-muted hover:bg-background-elevated"
          )}
        >
          <div className="text-sm font-normal">{option.label}</div>
          <div className="text-xs text-foreground-subtle">{option.description}</div>
        </button>
      ))}
    </div>
  );
}
```

### Navigation Sidebar

```tsx
function Sidebar() {
  return (
    <aside className="w-[280px] h-screen bg-background-elevated border-r border-border overflow-y-auto">
      {/* Logo */}
      <div className="px-4 py-4 border-b border-border">
        <span className="text-foreground-bright text-base font-normal">Voice Clone Pipeline</span>
      </div>

      {/* Navigation Items */}
      <nav className="py-2">
        <CollapsibleSection title="Input" defaultOpen>
          <div className="text-xs text-foreground-muted">Standby</div>
          <DropZone />
        </CollapsibleSection>

        <CollapsibleSection title="Effects">
          <NavItem icon="circle" label="Prosody" active />
          <NavItem icon="circle" label="Emotion" />
          <NavItem icon="circle" label="Style" />
        </CollapsibleSection>

        <CollapsibleSection title="Settings" defaultOpen>
          {/* Form controls */}
        </CollapsibleSection>
      </nav>

      {/* Footer */}
      <div className="absolute bottom-0 left-0 right-0 px-4 py-3 border-t border-border">
        <div className="flex gap-4 text-xs text-foreground-muted">
          <a href="#" className="hover:text-foreground">About</a>
          <a href="#" className="hover:text-foreground">Changelog</a>
        </div>
      </div>
    </aside>
  );
}
```

### Drop Zone

```tsx
function DropZone() {
  return (
    <div className="border border-dashed border-border rounded-lg p-6 text-center">
      <p className="text-sm text-foreground-muted">Drop file or click to browse</p>
      <p className="text-xs text-foreground-subtle mt-1">WAV, MP3, FLAC</p>
    </div>
  );
}
```

### Button Variants

```tsx
// Primary action (rare - most actions are form controls)
<button className="bg-foreground text-background px-4 py-2 text-sm rounded hover:bg-foreground-bright">
  Export
</button>

// Secondary/Ghost
<button className="text-foreground-muted hover:text-foreground text-sm py-2 px-3">
  Reset
</button>

// Icon button
<button className="p-2 text-foreground-muted hover:text-foreground">
  <Icon className="w-4 h-4" />
</button>
```

---

## Layout Patterns

### Three-Panel Layout

```
+------------------+------------------------+------------------+
|    Sidebar       |    Main Content        |   Right Panel    |
|    (280px)       |    (flexible)          |   (300px)        |
+------------------+------------------------+------------------+
```

```tsx
function AppLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex h-screen bg-background text-foreground font-mono">
      <Sidebar />
      <main className="flex-1 overflow-auto">
        {children}
      </main>
      <RightPanel />
    </div>
  );
}
```

### Content Area States

```tsx
// Empty state
<div className="flex-1 flex flex-col items-center justify-center text-center">
  <p className="text-foreground-muted text-sm">Awaiting input</p>
  <p className="text-foreground-subtle text-xs mt-1">Drop a file or select a source</p>
</div>

// Loading state
<div className="flex-1 flex items-center justify-center">
  <div className="text-foreground-muted text-sm">Processing...</div>
</div>
```

### Zoom Controls (Bottom Bar)

```tsx
function ZoomControls({ zoom, onZoomChange }: ZoomControlsProps) {
  return (
    <div className="flex items-center gap-4 px-4 py-3 border-t border-border text-sm">
      <button className="text-foreground-muted hover:text-foreground">-</button>
      <span className="text-foreground w-12 text-center">{zoom}%</span>
      <button className="text-foreground-muted hover:text-foreground">+</button>
      <span className="text-foreground-subtle">|</span>
      <button className="text-foreground-muted hover:text-foreground">Reset</button>
      <span className="text-foreground">{zoom}%</span>
    </div>
  );
}
```

---

## Animation & Transitions

### Timing

```css
/* Default transition */
transition: all 200ms ease;

/* Color transitions only */
transition: color 150ms ease, background-color 150ms ease;
```

### Hover States

```css
/* Text hover */
.nav-item {
  color: var(--foreground-muted);
}
.nav-item:hover {
  color: var(--foreground);
}

/* Active indicator */
.nav-item.active::before {
  content: "";
  width: 6px;
  height: 6px;
  background: var(--foreground-bright);
  border-radius: 50%;
}
```

### Do's and Don'ts

**DO:**
- Use simple opacity/color transitions
- Keep all transitions under 200ms
- Animate only color and opacity

**DON'T:**
- No scale animations
- No slide-in/out animations
- No bounce or spring effects
- No loading spinners (use text states)

---

## Icons

### Style

- Use simple line icons (Lucide icons work well)
- Size: 16px (w-4 h-4) for navigation, 14px for inline
- Color: Match text color of context (--foreground-muted for labels)

### Usage

```tsx
// Navigation icon
<Circle className="w-2 h-2 fill-current" /> // Active indicator

// Settings icon
<Settings className="w-4 h-4" />

// Inline with text
<span className="flex items-center gap-2">
  <Download className="w-4 h-4" />
  Export
</span>
```

---

## Accessibility

### Focus States

```css
/* Custom focus ring for dark mode */
*:focus-visible {
  outline: 1px solid var(--foreground-muted);
  outline-offset: 2px;
}

/* Remove default focus ring */
*:focus {
  outline: none;
}
```

### Contrast

All text colors meet WCAG AA contrast requirements against the dark background:
- #888888 on #0a0a0a = 5.9:1 (AA)
- #cccccc on #0a0a0a = 12.6:1 (AAA)
- #555555 on #0a0a0a = 3.2:1 (AA Large Text)

---

## Implementation Checklist

### globals.css
- [ ] Update CSS custom properties with new color tokens
- [ ] Remove light mode variables
- [ ] Add font-family to body
- [ ] Add base styles for inputs, buttons

### tailwind.config.js
- [ ] Add fontFamily.mono with IBM Plex Mono
- [ ] Extend colors with new tokens
- [ ] Add custom spacing if needed

### layout.tsx
- [ ] Add 'dark' class to html
- [ ] Import IBM Plex Mono via next/font
- [ ] Apply font class to body

### Components to Update
- [ ] Navigation.tsx - Sidebar style
- [ ] page.tsx - Landing page redesign
- [ ] All shadcn/ui components - Update variants

---

## Quick Reference

### Tailwind Classes Cheatsheet

```
Background:   bg-background         -> #0a0a0a
              bg-background-elevated -> #0f0f0f
              bg-background-card    -> #141414

Text:         text-foreground-bright -> #cccccc (headings)
              text-foreground       -> #888888 (body)
              text-muted-foreground -> #555555 (labels)

Border:       border-border         -> #222222
              border-border-subtle  -> #1a1a1a

Spacing:      p-4 gap-4            -> Standard (16px)
              p-3 gap-3            -> Compact (12px)
              p-2 gap-2            -> Tight (8px)
```

### Color Conversion

```
#0a0a0a = hsl(0, 0%, 4%)    = rgb(10, 10, 10)
#0f0f0f = hsl(0, 0%, 6%)    = rgb(15, 15, 15)
#141414 = hsl(0, 0%, 8%)    = rgb(20, 20, 20)
#222222 = hsl(0, 0%, 13%)   = rgb(34, 34, 34)
#333333 = hsl(0, 0%, 20%)   = rgb(51, 51, 51)
#555555 = hsl(0, 0%, 33%)   = rgb(85, 85, 85)
#888888 = hsl(0, 0%, 53%)   = rgb(136, 136, 136)
#cccccc = hsl(0, 0%, 80%)   = rgb(204, 204, 204)
```
