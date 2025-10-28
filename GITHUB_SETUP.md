# GitHub Repository Setup Guide

Follow these steps to create and push your repository to GitHub.

## Step 1: Initialize Local Git Repository

```powershell
# Navigate to your project directory (if not already there)
cd "C:\Users\carlo\OneDrive - VIDENS ANALYTICS\Prompt Injection\Experiment"

# Initialize git repository
git init

# Configure your git identity (if not done globally)
git config user.name "Your Name"
git config user.email "your.email@example.com"

# Add all files to staging
git add .

# Create initial commit
git commit -m "Initial commit: Prompt Injection Defense Framework

- Complete defense implementation (signature, rules, classifier, NeMo)
- Real-world API validation (OpenAI, Anthropic)
- Cross-model generalization testing
- Statistical analysis tools (bootstrap, McNemar)
- Paper experiment framework for IEEE Software submission
- Comprehensive documentation and results"
```

## Step 2: Create GitHub Repository (via CLI)

### Option A: Using GitHub CLI (gh)

First, check if GitHub CLI is installed:
```powershell
gh --version
```

If not installed, download from: https://cli.github.com/

Then create the repository:
```powershell
# Login to GitHub
gh auth login

# Create repository
gh repo create prompt-injection-defense --public --source=. --description="Systematic framework for evaluating prompt injection defenses with real-world API validation" --push

# That's it! Your repo is created and pushed.
```

### Option B: Manual Setup (via GitHub Website)

1. **Go to GitHub**: https://github.com/new

2. **Create repository**:
   - Repository name: `prompt-injection-defense`
   - Description: "Systematic framework for evaluating prompt injection defenses with real-world API validation"
   - Public repository (or Private if you prefer)
   - **DO NOT** initialize with README, .gitignore, or license (we already have these)
   - Click "Create repository"

3. **Link local repo to GitHub**:
```powershell
# Add remote origin (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/prompt-injection-defense.git

# Verify remote
git remote -v

# Push to GitHub
git branch -M main
git push -u origin main
```

## Step 3: Verify Upload

```powershell
# Check status
git status

# View your repository online
# https://github.com/YOUR_USERNAME/prompt-injection-defense
```

## Step 4: Add GitHub Repository Badges (Optional)

After creating the repo, update README.md with your actual repo URL:

Replace in README.md:
- `https://github.com/yourusername/prompt-injection-defense.git` 
- with `https://github.com/YOUR_USERNAME/prompt-injection-defense.git`

## Useful Git Commands for Future Updates

```powershell
# Check what files have changed
git status

# Add specific files
git add filename.py

# Add all changed files
git add .

# Commit changes
git commit -m "Description of changes"

# Push to GitHub
git push

# Pull latest changes from GitHub
git pull

# View commit history
git log --oneline

# Create a new branch for experimental features
git checkout -b feature-branch-name

# Switch back to main branch
git checkout main

# Merge branch into main
git merge feature-branch-name
```

## Important Notes

### Files to NEVER Commit

The `.gitignore` file already excludes these, but double-check:
- `.env` (contains API keys!)
- Any files with actual API keys
- Large model files (*.bin, *.pt, etc.)

### Before Each Commit

```powershell
# Review what will be committed
git status
git diff

# Make sure no secrets are included
cat .env  # Should NOT appear in git status
```

### Large Files Warning

If you have large result files, consider:
1. Using Git LFS (Large File Storage)
2. Or exclude them in .gitignore
3. Or use GitHub Releases for datasets

## Recommended GitHub Repository Settings

After creating the repo:

1. **Add topics/tags** (in repo settings):
   - `prompt-injection`
   - `llm-security`
   - `defense-evaluation`
   - `machine-learning`
   - `nlp`
   - `security`

2. **Enable GitHub Pages** (for documentation):
   - Settings → Pages → Source: main branch, /docs folder

3. **Add repository description**:
   "Systematic framework for evaluating prompt injection defenses with real-world API validation"

4. **Set up branch protection** (for main branch):
   - Require pull request reviews
   - Require status checks

## Sharing Your Work

Once published, share:
- GitHub URL: `https://github.com/YOUR_USERNAME/prompt-injection-defense`
- Clone command: `git clone https://github.com/YOUR_USERNAME/prompt-injection-defense.git`

## Archiving for DOI (Zenodo)

For academic citation:

1. **Connect GitHub to Zenodo**:
   - Go to https://zenodo.org/
   - Login with GitHub
   - Enable repository in Zenodo

2. **Create a release**:
```powershell
# Tag your current state
git tag -a v1.0.0 -m "Version 1.0.0: IEEE Software submission"
git push origin v1.0.0
```

3. **Get DOI**:
   - Zenodo automatically creates a DOI for each release
   - Add DOI to README.md and paper

## Troubleshooting

### "Permission denied" when pushing
```powershell
# Use HTTPS with token or set up SSH keys
# Generate token: GitHub → Settings → Developer settings → Personal access tokens
```

### "Large files" error
```powershell
# Remove large files from git history
git rm --cached large_file.csv
git commit -m "Remove large file"

# Or use Git LFS
git lfs install
git lfs track "*.csv"
```

### "Merge conflict"
```powershell
# Pull latest changes first
git pull

# Resolve conflicts in files (Git will mark them)
# Then commit the resolution
git add .
git commit -m "Resolve merge conflicts"
git push
```
