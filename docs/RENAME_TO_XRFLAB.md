# Renaming FpXrF to XRFLab

## Steps to Complete the Rename

### 1. ✅ Code Updates (DONE)
I've updated all references in the code:
- `main.py` - Application name changed to "XRFLab"
- `ui/main_window.py` - Window title and About dialog
- `README.md` - Project title and structure
- `setup.sh` - Script header

### 2. 📁 Rename Local Folder

**Option A: Using Finder (Easiest)**
1. Close this IDE/editor
2. Navigate to: `/Users/aaroncelestian/Library/Mobile Documents/com~apple~CloudDocs/Python/`
3. Rename folder `FpXrF` to `XRFLab`
4. Reopen the project in your IDE

**Option B: Using Terminal**
```bash
cd "/Users/aaroncelestian/Library/Mobile Documents/com~apple~CloudDocs/Python/"
mv FpXrF XRFLab
cd XRFLab
```

### 3. 🔄 Update Git Repository

#### A. Rename on GitHub/GitLab
1. Go to your repository on GitHub
2. Click **Settings**
3. Under "Repository name", change to `XRFLab`
4. Click **Rename**

#### B. Update Local Git Remote
After renaming on GitHub, update your local repository:

```bash
cd "/Users/aaroncelestian/Library/Mobile Documents/com~apple~CloudDocs/Python/XRFLab"

# Update the remote URL
git remote set-url origin https://github.com/YOUR_USERNAME/XRFLab.git

# Verify the change
git remote -v
```

### 4. 📝 Update Git Configuration (Optional)
If you want to update commit messages or history:

```bash
# Just commit the rename changes
git add .
git commit -m "Rename project from FpXrF to XRFLab"
git push
```

### 5. ✅ Verify Everything Works

After renaming:

```bash
cd "/Users/aaroncelestian/Library/Mobile Documents/com~apple~CloudDocs/Python/XRFLab"

# Test the application
python main.py

# Verify git status
git status
```

## What Was Changed

### Files Updated
- ✅ `main.py` - App name: "XRFLab"
- ✅ `ui/main_window.py` - Window title: "XRFLab - Fundamental Parameters Analysis"
- ✅ `ui/main_window.py` - About dialog updated
- ✅ `README.md` - Title changed to "XRFLab"
- ✅ `README.md` - Project structure diagram updated
- ✅ `setup.sh` - Header updated

### What Stays the Same
- All functionality remains unchanged
- All file paths and imports work the same
- Documentation content is preserved
- Git history is preserved

## Quick Reference

**Old Name**: FpXrF (Fundamental parameters X-ray Fluorescence)  
**New Name**: XRFLab  

**Old Folder**: `/Users/aaroncelestian/.../Python/FpXrF`  
**New Folder**: `/Users/aaroncelestian/.../Python/XRFLab`  

**Old Repo**: `github.com/YOUR_USERNAME/FpXrF`  
**New Repo**: `github.com/YOUR_USERNAME/XRFLab`  

## Troubleshooting

### If IDE shows errors after rename:
1. Close the IDE completely
2. Reopen the project from the new `XRFLab` folder
3. IDE should re-index the project

### If git push fails:
```bash
# Make sure remote URL is updated
git remote -v

# Should show:
# origin  https://github.com/YOUR_USERNAME/XRFLab.git (fetch)
# origin  https://github.com/YOUR_USERNAME/XRFLab.git (push)
```

### If imports break:
- They shouldn't! All imports use relative paths
- If issues occur, restart Python/IDE

---

**After completing these steps, your project will be fully renamed to XRFLab!** 🎉
