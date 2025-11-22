# Periodic Table v2.0 - Compact & Interactive

## Updates

### 🎨 Compact Design
- **Reduced button size** from 55x55 to 35x35 pixels
- **Simplified display** - shows only element symbol (not atomic number)
- **Smaller font** for better space efficiency
- **More compact layout** - fits better in the left panel

### 🖱️ Interactive Features

#### Left-Click: Show Emission Lines
When you **click on an element**, all its X-ray emission lines appear on the spectrum plot:
- **K lines** (red) - Kα1, Kα2, Kβ1, Kβ2, Kβ3
- **L lines** (green) - Lα1, Lα2, Lβ1, Lβ2, Lβ3, Lβ4, Lγ1, Lγ2, Lγ3
- **M lines** (blue) - Mα1, Mα2, Mβ, Mγ
- **N lines** (magenta) - N series lines

Each line is displayed as a vertical dashed line with a label showing the element and line designation.

#### Right-Click: Element Information
When you **right-click on an element**, a detailed information dialog appears showing:
- **Element name** and symbol
- **Atomic number** (Z)
- **Atomic weight** (g/mol)
- **Density** (g/cm³)
- **Complete list of emission lines** organized by series (K, L, M, N)
  - Line name (e.g., Kα1, Lβ2)
  - Energy in keV

### 🎨 Color Coding

**Emission Line Series Colors:**
- **Red** - K series (highest energy)
- **Green** - L series
- **Blue** - M series
- **Magenta** - N series (lowest energy)

**Element Group Colors** (unchanged):
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

## Usage

### View Emission Lines
1. Load a spectrum (File → Open Spectrum)
2. Click on any element in the periodic table
3. All emission lines for that element appear on the plot
4. Lines are color-coded by series (K=red, L=green, M=blue, N=magenta)

### View Element Details
1. Right-click on any element in the periodic table
2. A dialog appears with:
   - Basic element information
   - Complete list of all emission lines with energies
3. Click "Close" to dismiss the dialog

### Clear Lines
- Click on a different element to show its lines (previous lines are cleared)
- Or use the spectrum plot controls to reset the view

## Technical Details

### X-ray Data Source
- **Primary**: xraylib - Comprehensive X-ray database
- **Fallback**: Approximate Moseley's law calculations if xraylib unavailable

### New Files
- `core/xray_data.py` - X-ray emission line data interface
  - `get_element_lines(symbol, z)` - Get emission lines
  - `get_element_info(symbol, z)` - Get element properties
  - Fallback data when xraylib not available

### Updated Files
- `ui/periodic_table_widget.py`
  - Reduced button size to 35x35 px
  - Added right-click context menu support
  - Added signals for element clicks and info requests

- `ui/element_panel.py`
  - Added element info dialog
  - Connected periodic table signals
  - Displays emission lines in formatted dialog

- `ui/spectrum_widget.py`
  - Added `show_element_lines()` method
  - Color-coded emission line markers
  - Vertical dashed lines with labels

- `ui/main_window.py`
  - Connected element click to show lines
  - Status bar updates when element clicked

## Benefits

✅ **More compact** - Takes up less screen space  
✅ **Interactive** - Click to see emission lines instantly  
✅ **Informative** - Right-click for detailed element data  
✅ **Color-coded** - Easy to distinguish line series  
✅ **Professional** - Uses xraylib for accurate data  
✅ **Fallback** - Works even without xraylib  

## Example Workflow

1. **Load sample data**:
   ```bash
   python main.py
   # File → Open Spectrum → sample_data/steel_sample.txt
   ```

2. **Click on Fe** in periodic table:
   - Red lines appear for Fe K-alpha (6.404 keV) and K-beta (7.058 keV)
   - Green lines appear for Fe L series

3. **Right-click on Fe**:
   - Dialog shows:
     - Iron (Fe), Z=26
     - Atomic weight: 55.845 g/mol
     - All K and L emission lines with exact energies

4. **Click on Cr**:
   - Fe lines disappear
   - Cr lines appear (K-alpha at 5.415 keV, etc.)

## Future Enhancements

- Toggle to show/hide specific line series
- Adjust line opacity
- Save/load element line configurations
- Highlight lines that match detected peaks
- Show theoretical intensities

---

**The periodic table is now more compact and highly interactive, making XRF analysis faster and more intuitive!**
