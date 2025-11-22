# Periodic Table Update

## What's New

The element selection interface has been upgraded from a tree list to a **beautiful, interactive periodic table**!

## Features

### 🎨 Visual Periodic Table
- **Full periodic table layout** with all elements up to Z=108
- **Color-coded by element group**:
  - Alkali metals (red)
  - Alkaline earth metals (coral)
  - Transition metals (yellow)
  - Post-transition metals (mint)
  - Metalloids (light green)
  - Nonmetals (sky blue)
  - Halogens (plum)
  - Noble gases (lavender)
  - Lanthanides (peach)
  - Actinides (light pink)

### 🖱️ Interactive Selection
- **Click any element** to select/deselect
- **Hover** to see element name and atomic number
- **Visual feedback** with border highlighting
- **Selected elements** show with bold green border

### 🔧 Quick Actions
- **Select All** - Select all elements
- **Clear All** - Deselect all elements
- **Common XRF** - Auto-select commonly analyzed elements (Na-Bi range)

### 📊 Color Legend
- Visual legend at bottom showing element group colors
- Easy identification of element types

## Usage

1. **Launch the application**:
   ```bash
   python main.py
   ```

2. **Select elements**:
   - Click on any element in the periodic table
   - Selected elements will highlight with a green border
   - Click again to deselect

3. **Quick selection**:
   - Click "Common XRF" to select typical XRF elements
   - Click "Select All" to select everything
   - Click "Clear All" to start over

4. **View selection**:
   - Selected elements are tracked automatically
   - Element data includes symbol, atomic number, and name

## Technical Details

### New Files
- `ui/periodic_table_widget.py` - Interactive periodic table widget

### Updated Files
- `ui/element_panel.py` - Now uses periodic table instead of tree widget
- `ui/main_window.py` - Fixed toolbar warning

### Element Coverage
- **118 elements** total (H to Hs)
- **Organized by periods** (rows 1-7)
- **Lanthanides and actinides** shown separately below main table
- **All XRF-relevant elements** included

### Styling
- Custom `ElementButton` class with group-based colors
- Hover effects for better UX
- Selected state with bold border
- Responsive layout with scroll support

## Benefits

### Over Tree List
✅ **More intuitive** - Familiar periodic table layout  
✅ **Better visualization** - See element relationships  
✅ **Faster selection** - Click directly on elements  
✅ **Color coding** - Identify element types at a glance  
✅ **Complete coverage** - All elements, not just common ones  
✅ **Professional appearance** - Modern scientific interface  

## Common XRF Elements

The "Common XRF" button selects these elements:
- **Light**: Na, Mg, Al, Si, P, S, Cl, K, Ca
- **Transition**: Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn
- **Others**: As, Se, Br, Rb, Sr, Y, Zr, Nb, Mo, Ag, Cd, Sn, Sb, Ba, W, Pb, Bi

## Screenshots

### Periodic Table View
```
┌─────────────────────────────────────────────────────────┐
│         Periodic Table - Select Elements                │
├─────────────────────────────────────────────────────────┤
│  H                                                  He  │
│  Li Be                          B  C  N  O  F      Ne  │
│  Na Mg                          Al Si P  S  Cl     Ar  │
│  K  Ca Sc Ti V  Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr │
│  Rb Sr Y  Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I  Xe │
│  Cs Ba La Hf Ta W  Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn │
│  Fr Ra Ac Rf Db Sg Bh Hs                                │
│                                                          │
│  Lanthanides → Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu│
│  Actinides   → Th Pa U  Np Pu Am Cm Bk Cf               │
├─────────────────────────────────────────────────────────┤
│  [Select All] [Clear All] [Common XRF]                  │
├─────────────────────────────────────────────────────────┤
│  ■ Alkali  ■ Alkaline  ■ Transition  ■ Post-Trans.     │
│  ■ Metalloid  ■ Nonmetal  ■ Halogen  ■ Noble           │
│  ■ Lanthanide  ■ Actinide                               │
└─────────────────────────────────────────────────────────┘
```

### Element Button
```
┌──────┐
│  Fe  │  ← Symbol
│  26  │  ← Atomic number
└──────┘
Tooltip: Iron (Fe)
         Atomic Number: 26
```

## Future Enhancements

Potential additions for Phase 2+:
- Show available X-ray lines (K, L, M) on hover
- Highlight elements with detected peaks
- Filter by energy range
- Show element concentrations on buttons
- Export/import element selections

---

**The periodic table provides a much more intuitive and professional interface for element selection in XRF analysis!**
