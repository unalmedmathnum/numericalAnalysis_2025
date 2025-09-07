## Numerical Analysis Course Codes 🧮

Welcome to the Numerical Analysis Course Codes repository! 

This repository contains all the code examples, scripts, and computational exercises from our numerical analysis course. Don't worry if you're new to programming or Git - we'll guide you through everything step by step! 😊

## What You'll Need 📋

Don't panic if you don't have these installed yet - we'll help you install everything:

- **Python 3.9 or higher** (a programming language)
- **Git** (for downloading and managing code)
- **Node.js** (for managing code quality)
- **A text editor or IDE** (we recommend VS Code)

> 💡 **New to programming?** No worries! We'll explain everything as we go.

## Step-by-Step Setup Guide 🚀

Follow these steps **in order**. Don't skip any steps!

### Step 1: Install Git 🔧

Git helps you download and manage code. Think of it like Google Drive, but for programmers.

**Windows:**
1. Go to [git-scm.com](https://git-scm.com/)
2. Click "Download for Windows"
3. Run the installer and click "Next" through all options (the defaults are fine)
4. When finished, open "Git Bash" from your Start menu

**macOS:**
1. Open Terminal (press `Cmd + Space`, type "Terminal", press Enter)
2. Type this command and press Enter:
   ```bash
   git --version
   ```
3. If Git isn't installed, your Mac will offer to install it automatically. Click "Install"
4. Or install via Homebrew (if you have it):
   ```bash
   brew install git
   ```

**Linux (Ubuntu/Debian):**
1. Open Terminal (`Ctrl + Alt + T`)
2. Type these commands one by one:
   ```bash
   sudo apt update
   sudo apt install git
   ```

**Linux (CentOS/RHEL/Fedora):**
1. Open Terminal
2. Type this command:
   ```bash
   sudo dnf install git
   ```

### Step 2: Install Python 🐍

Python is the programming language we'll use for all our numerical analysis.

**Windows:**
1. Go to [python.org](https://python.org/)
2. Click "Download Python" (get the latest version 3.9+)
3. **IMPORTANT**: When installing, check the box "Add Python to PATH"
4. Click "Install Now"
5. Test it works: Open Command Prompt and type `python --version`

**macOS:**
Option 1 (Recommended - using Homebrew):
1. First install Homebrew if you don't have it:
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```
2. Then install Python:
   ```bash
   brew install python
   ```

Option 2 (Direct download):
1. Go to [python.org](https://python.org/)
2. Download and install the latest version

**Linux:**
Most Linux systems have Python, but let's make sure you have the right version:
```bash
# Ubuntu/Debian
sudo apt update && sudo apt install python3 python3-pip python3-venv

# CentOS/RHEL/Fedora
sudo dnf install python3 python3-pip
```

**Test Python Installation:**
Open your terminal/command prompt and type:
```bash
python --version
# or if that doesn't work, try:
python3 --version
```
You should see something like "Python 3.11.0" or similar.

### Step 3: Install Node.js 📦

Node.js helps us maintain clean, consistent code. It's like a quality checker for our code.

**Windows:**
1. Go to [nodejs.org](https://nodejs.org/)
2. Click the green button "Download for Windows" (get the LTS version)
3. Run the installer and accept all defaults
4. Test: Open Command Prompt and type `node --version`

**macOS:**
Option 1 (Homebrew - recommended):
```bash
brew install node
```

Option 2 (Direct download):
1. Go to [nodejs.org](https://nodejs.org/)
2. Download and install the LTS version

**Linux:**
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install nodejs npm

# Or using snap (alternative)
sudo snap install node --classic

# CentOS/RHEL/Fedora
sudo dnf install nodejs npm
```

**Test Node.js Installation:**
```bash
node --version  # Should show v16.0.0 or higher
npm --version   # Should show 8.0.0 or higher
```

### Step 4: Get the Code 📁

Now let's download our course repository!

1. **Choose a location** for your code. We recommend creating a folder like:
   - Windows: `C:\Users\YourName\Documents\Programming\`
   - macOS/Linux: `/home/yourusername/Programming/` or `~/Programming/`

2. **Open your terminal** in that location:
   - Windows: Navigate to the folder in File Explorer, then right-click and select "Git Bash Here"
   - macOS: Right-click the folder and select "New Terminal at Folder"
   - Linux: Right-click the folder and select "Open in Terminal"

3. **Download the repository:**
   ```bash
   git clone https://github.com/Zaprovic/numericalAnalysis_2025.git
   ```

4. **Enter the project folder:**
   ```bash
   cd numericalAnalysis_2025
   ```

### Step 5: Set Up Python Environment 🔒

We'll create a "virtual environment" - think of it as a separate, clean space for our Python packages that won't interfere with other projects.

1. **Create the virtual environment:**
   ```bash
   python -m venv venv
   # If that doesn't work, try: python3 -m venv venv
   ```

2. **Activate the virtual environment:**
   
   **Windows (Command Prompt or PowerShell):**
   ```bash
   venv\Scripts\activate
   ```
   
   **Windows (Git Bash):**
   ```bash
   source venv/Scripts/activate
   ```
   
   **macOS/Linux:**
   ```bash
   source venv/bin/activate
   ```

3. **Check if it worked:**
   Your terminal prompt should now show `(venv)` at the beginning. This means you're in the virtual environment!

4. **Install Python packages:**
   ```bash
   pip install -r requirements.txt
   ```

### Step 6: Set Up Code Quality Tools ✨

This step ensures everyone's code looks consistent and follows good practices.

1. **Install Node.js dependencies:**
   ```bash
   npm install
   ```
   
   ⚠️ **This step is CRUCIAL!** If you skip this, the code quality tools won't work.

2. **Test that everything is working:**
   ```bash
   npm run commit --version
   ```

You're all set! 🎉

## Daily Workflow 💻

Every time you want to work on the project, follow these steps:

### Starting Your Work Session

1. **Open your terminal** and navigate to the project folder:
   ```bash
   cd path/to/numericalAnalysis_2025
   ```

2. **Activate your Python environment** (you'll see `(venv)` appear in your prompt):
   
   **Windows:**
   ```bash
   venv\Scripts\activate
   ```
   
   **macOS/Linux:**
   ```bash
   source venv/bin/activate
   ```

3. **You're ready to code!** 🎉

### Running Python Scripts

Navigate to the folder with the script you want to run:

```bash
# Example: Go to Part 1 exercises
cd src/parte-01

# Run a Python script
python main.py
```

### Working with Jupyter Notebooks

If you want to use Jupyter notebooks for interactive coding:

1. **Make sure your virtual environment is active** (you should see `(venv)`)

2. **Start Jupyter:**
   ```bash
   jupyter notebook
   ```

3. **Your browser will open** with the Jupyter interface

4. **Navigate to any `.ipynb` file** and start coding!

### Finishing Your Work Session

When you're done working:

```bash
# Deactivate your virtual environment
deactivate
```

The `(venv)` will disappear from your prompt.

## How to Save Your Changes (Git Commits) 💾

When you make changes to the code and want to save them properly, follow these steps:

### The Easy Way (Recommended for Beginners)

1. **Make sure you're in the project folder:**
   ```bash
   cd numericalAnalysis_2025
   ```

2. **Check what files you've changed:**
   ```bash
   git status
   ```
   This shows you which files are new or modified.

3. **Add your changes:**
   ```bash
   git add .
   ```
   This prepares all your changes to be saved.

4. **Create a commit using our guided tool:**
   ```bash
   npm run commit
   ```
   
   This will ask you questions like:
   - What type of change is this? (feature, fix, documentation, etc.)
   - What did you change?
   - Why did you change it?
   
   Just answer the questions, and it will create a properly formatted commit message!

5. **Upload your changes to GitHub:**
   ```bash
   git push
   ```

### Understanding Commit Types 📝

When using `npm run commit`, you'll be asked what type of change you made:

- **feat**: You added a new feature or function
  - Example: "Added bisection method for finding roots"
  
- **fix**: You fixed a bug or error
  - Example: "Fixed calculation error in Newton's method"
  
- **docs**: You updated documentation or comments
  - Example: "Added more explanations to README"
  
- **style**: You improved code formatting (no logic changes)
  - Example: "Fixed indentation in main.py"
  
- **refactor**: You reorganized code without changing what it does
  - Example: "Split large function into smaller ones"

### Alternative: Manual Commits (Advanced)

If you prefer typing commit messages manually, make sure they follow this format:

```
type(scope): short description

Optional longer explanation of what you changed and why.
```

Example:
```bash
git commit -m "feat(root-finding): add bisection method

Added bisection method for finding roots of continuous functions
with proper error handling and convergence criteria."
```

> ⚠️ **Important**: Make sure you ran `npm install` during setup, or the guided commit tool won't work!

## Troubleshooting Common Issues 🔧

Don't panic if something doesn't work! Here are solutions to common problems:

### "Command not found" Errors

**"git: command not found"**
- **Solution**: Git isn't installed properly
- **Fix**: Go back to Step 1 and reinstall Git
- Make sure to restart your terminal after installation

**"python: command not found"**
- **Solution**: Try `python3` instead of `python`
- **Fix**: If that doesn't work, Python isn't installed properly. Go back to Step 2
- **Windows users**: Make sure you checked "Add Python to PATH" during installation

**"npm: command not found"**
- **Solution**: Node.js isn't installed properly
- **Fix**: Go back to Step 3 and reinstall Node.js
- Restart your terminal after installation

### Virtual Environment Issues

**"(venv) doesn't appear in my terminal"**
- **Solution**: The virtual environment isn't activated
- **Fix**: Try the activation command again:
  ```bash
  # Windows
  venv\Scripts\activate
  
  # macOS/Linux
  source venv/bin/activate
  ```

**"No module named 'numpy'" (or other packages)**
- **Solution**: Either virtual environment isn't active, or packages aren't installed
- **Fix**: 
  1. Make sure you see `(venv)` in your terminal
  2. If not, activate it first
  3. Then run: `pip install -r requirements.txt`

**"Permission denied" when creating virtual environment**
- **Solution**: Don't use `sudo` with Python commands
- **Fix**: Make sure you're in your home directory or a folder you own

### Git and Commit Issues

**"You can make any commit message" (conventional commits not enforcing)**
- **Solution**: You didn't run `npm install` properly
- **Fix**: 
  1. Make sure you're in the project root directory
  2. Run `npm install` again
  3. Look for a `.husky` folder - it should exist after npm install

**"commitizen: command not found"**
- **Solution**: Node.js dependencies aren't installed
- **Fix**: Run `npm install` in the project root directory

**"fatal: not a git repository"**
- **Solution**: You're not in the right folder
- **Fix**: Navigate to the `numericalAnalysis_2025` folder using `cd`

### Installation Issues

**"Permission denied" errors on Linux/macOS**
- **Solution**: Don't use `sudo` for npm or pip in virtual environments
- **Fix**: For npm, configure a different directory:
  ```bash
  mkdir ~/.npm-global
  npm config set prefix '~/.npm-global'
  echo 'export PATH=~/.npm-global/bin:$PATH' >> ~/.bashrc
  source ~/.bashrc
  ```

**Python packages installing globally instead of in virtual environment**
- **Solution**: Virtual environment isn't activated
- **Fix**: Always check for `(venv)` in your terminal before installing packages

### Still Having Problems?

1. **Check you're in the right directory**: 
   ```bash
   pwd  # Shows current directory
   ls   # Shows files in current directory (should see README.md, package.json, etc.)
   ```

2. **Check what's installed**:
   ```bash
   git --version
   python --version  # or python3 --version
   node --version
   npm --version
   ```

3. **Start fresh**: If all else fails, delete the project folder and start from Step 4 again

4. **Ask for help**: Open an issue on GitHub with:
   - Your operating system
   - What command you ran
   - The exact error message you got

### Quick Recovery Commands

If you mess up and want to start over:

```bash
# Deactivate virtual environment
deactivate

# Delete and recreate virtual environment
rm -rf venv  # or rmdir /s venv on Windows
python -m venv venv

# Reactivate and reinstall
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Usage
**Important**: Always activate your virtual environment before running any Python scripts:

```bash
# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

Navigate to the specific topic directory and run the Python scripts:

```bash
cd root-finding/src/parte-01
python main.py  # for Python scripts
```

For Jupyter notebooks:
```bash
# Install Jupyter in your virtual environment (if not already installed)
pip install jupyter

# Start Jupyter notebook
jupyter notebook  # then navigate to the desired .ipynb file
```

**Note**: Make sure to keep your virtual environment activated throughout your work session.

## Contributing 🤝

We'd love your help making this repository even better! Here's how you can contribute:

### What We're Looking For
- **Additional examples** or alternative implementations of numerical methods
- **Improvements** in code readability and efficiency
- **Better explanations** and documentation
- **Bug fixes** and error corrections
- **New numerical analysis methods** we haven't covered yet

### How to Contribute

1. **Fork the repository** (click the "Fork" button on GitHub)
2. **Clone your fork** to your computer:
   ```bash
   git clone https://github.com/YOUR-USERNAME/numericalAnalysis_2025.git
   ```
3. **Create a new branch** for your changes:
   ```bash
   git checkout -b my-new-feature
   ```
4. **Make your changes** following our setup guide above
5. **Test your changes** to make sure they work
6. **Commit your changes** using our commit process:
   ```bash
   npm run commit
   ```
7. **Push to your fork**:
   ```bash
   git push origin my-new-feature
   ```
8. **Create a Pull Request** on GitHub

### Contribution Guidelines

- **Follow the existing code style** - look at other files to see how they're written
- **Add comments** to explain complex code
- **Test your code** before submitting
- **Write clear commit messages** using our commit tool
- **Be respectful** in discussions and reviews

### First-Time Contributors

Never contributed to an open-source project before? No problem! 

- Start with small changes like fixing typos or improving documentation
- Look for "good first issue" labels on GitHub issues
- Don't be afraid to ask questions
- We're here to help you learn!

## Getting Help 📞

Stuck? Don't worry! Here are ways to get help:

### Quick Help
- **Check the troubleshooting section** above - it covers most common issues
- **Read error messages carefully** - they often tell you exactly what's wrong
- **Google the error message** - someone else has probably had the same problem

### Course Help
- **Ask your classmates** - they might have solved the same problem
- **Office hours** - bring your laptop and we'll debug together
- **Course forum/Discord** - post your question with details

### Technical Help
- **GitHub Issues**: Open an issue on this repository for:
  - Bugs in the code
  - Problems with setup instructions
  - Suggestions for improvements
  
- **Email**: Contact the instructor for course-related questions

### When Asking for Help

Please include:
1. **Your operating system** (Windows, macOS, Linux)
2. **What you were trying to do** (step-by-step)
3. **What command you ran** (copy and paste it)
4. **The exact error message** (copy and paste it)
5. **What you've already tried** to fix it

**Good example:**
> "I'm on Windows 10, trying to activate my virtual environment. I ran `venv\Scripts\activate` but got the error 'cannot be loaded because running scripts is disabled on this system'. I tried running as administrator but same error."

---

## Summary: Your Complete Checklist ✅

Print this out or bookmark it! Here's everything you need to do:

### One-Time Setup (do this once)
- [ ] Install Git
- [ ] Install Python 3.9+
- [ ] Install Node.js
- [ ] Clone the repository: `git clone https://github.com/Zaprovic/numericalAnalysis_2025.git`
- [ ] Navigate to project: `cd numericalAnalysis_2025`
- [ ] Create virtual environment: `python -m venv venv`
- [ ] Activate virtual environment (see commands above)
- [ ] Install Python packages: `pip install -r requirements.txt`
- [ ] Install Node.js packages: `npm install`

### Every Time You Work (daily workflow)
- [ ] Navigate to project folder
- [ ] Activate virtual environment (look for `(venv)` in terminal)
- [ ] Write your code
- [ ] Test your code
- [ ] Save changes: `git add .` → `npm run commit` → `git push`
- [ ] Deactivate virtual environment when done: `deactivate`

### When You Need Help
- [ ] Check troubleshooting section
- [ ] Ask classmates or instructor
- [ ] Open GitHub issue with details

**Remember**: The most important thing is the `(venv)` in your terminal - if you don't see it, activate your virtual environment first!


