## Numerical Analysis Course Codes
Welcome to the Numerical Analysis Course Codes repository! 
This repository contains (almost) all the code examples, scripts, and computational exercises 
corresponding to the lectures in the numerical analysis course.

## Requirements
All codes are written in Python. To run the scripts, you will need:

- **Python 3.9 or higher**
- **Virtual environment** (recommended for dependency isolation)
- **Dependencies listed in `requirements.txt`**

### Python Installation
- **Windows**: Download from [python.org](https://python.org/) or install via Microsoft Store
- **macOS**: 
  ```bash
  # Using Homebrew (recommended)
  brew install python
  
  # Or download from python.org
  ```
- **Linux**: Usually pre-installed, or install via package manager:
  ```bash
  # Ubuntu/Debian
  sudo apt update && sudo apt install python3 python3-pip python3-venv
  
  # CentOS/RHEL/Fedora
  sudo dnf install python3 python3-pip
  ```

### Quick Setup (using requirements.txt)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

## Development Setup
This repository uses conventional commits to maintain a clean and standardized commit history. Follow these steps to set up your development environment:

### Prerequisites
1. **Node.js and npm**: Required for commit linting and formatting
   - **Windows**: Download and install from [nodejs.org](https://nodejs.org/)
   - **macOS**: 
     ```bash
     # Using Homebrew (recommended)
     brew install node
     
     # Or download from nodejs.org
     ```
   - **Linux (Ubuntu/Debian)**:
     ```bash
     # Using apt
     sudo apt update
     sudo apt install nodejs npm
     
     # Or using snap
     sudo snap install node --classic
     ```
   - **Linux (CentOS/RHEL/Fedora)**:
     ```bash
     # Using dnf (Fedora)
     sudo dnf install nodejs npm
     
     # Using yum (CentOS/RHEL)
     sudo yum install nodejs npm
     ```

2. **Verify installation**:
   ```bash
   node --version  # Should show v16.0.0 or higher
   npm --version   # Should show 8.0.0 or higher
   ```

### Project Setup
1. **Clone the repository**:
   ```bash
   git clone https://github.com/Zaprovic/numericalAnalysis_2025.git
   cd numericalAnalysis_2025
   ```

2. **Set up Python virtual environment** (recommended):
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate virtual environment
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   
   # Your terminal prompt should now show (venv) at the beginning
   ```

3. **Install Python dependencies**:
   ```bash
   # Install from requirements.txt (recommended)
   pip install -r requirements.txt
   
   # Or install individual packages
   # pip install numpy scipy matplotlib pandas
   ```

4. **Install Node.js project dependencies**:
   ```bash
   npm install
   ```

### Working with Virtual Environment
- **Always activate** the virtual environment before working on the project:
  ```bash
  # On Windows:
  venv\Scripts\activate
  # On macOS/Linux:
  source venv/bin/activate
  ```
- **Deactivate** when you're done:
  ```bash
  deactivate
  ```
- **Verify** you're in the virtual environment:
  ```bash
  which python  # Should show path to venv/bin/python (macOS/Linux)
  where python  # Should show path to venv\Scripts\python.exe (Windows)
  ```

### Making Commits
Instead of using `git commit -m "message"`, use our standardized commit process:

```bash
npm run commit
```

This will:
- Guide you through creating a properly formatted conventional commit
- Automatically lint your commit message
- Ensure consistency across all contributors

**Example commit types:**
- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation only changes
- `style`: Changes that do not affect the meaning of the code
- `refactor`: A code change that neither fixes a bug nor adds a feature
- `test`: Adding missing tests or correcting existing tests

### Alternative Commit Method
If you prefer using git directly, make sure your commit messages follow this format:
```
type(scope): description

[optional body]

[optional footer]
```

Example:
```
feat(root-finding): add bisection method implementation

Add bisection method for finding roots of continuous functions
with proper error handling and convergence criteria.
```

## Troubleshooting

### Common Issues

**"npm: command not found"**
- Make sure Node.js is properly installed
- Restart your terminal after installation
- Check if Node.js is in your PATH

**"Permission denied" errors on Linux/macOS**
- Avoid using `sudo npm install`
- Configure npm to use a different directory:
  ```bash
  mkdir ~/.npm-global
  npm config set prefix '~/.npm-global'
  echo 'export PATH=~/.npm-global/bin:$PATH' >> ~/.bashrc
  source ~/.bashrc
  ```

**Commit hooks not working**
- Make sure you ran `npm install` after cloning
- Check if `.husky` directory exists
- Try running: `npm run prepare`

**"commitizen: command not found"**
- Run `npm install` to install all dependencies
- Make sure you're in the project root directory

**Python/Virtual Environment Issues**
- **"python: command not found"**: 
  - Try `python3` instead of `python`
  - Make sure Python is installed and in your PATH
- **"No module named 'numpy'" (or other packages)**:
  - Make sure your virtual environment is activated
  - Run `pip install -r requirements.txt`
- **Virtual environment not activating**:
  - Make sure you're in the project root directory
  - Check if `venv` folder exists
  - Try creating a new virtual environment: `python -m venv venv`
- **Permission errors when installing packages**:
  - Use virtual environment instead of global Python
  - Never use `sudo pip install`

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

## Contributing
This repository is intended as a resource for students in the numerical analysis course. 
Contributions are welcome, especially for:
- Additional examples or alternative implementations of methods.
- Improvements in code readability and efficiency.
- Suggestions for enhancing documentation.
- If you wish to contribute, please fork the repository, make your changes, and submit a pull request.

## Contact
For questions or further information about the course materials, feel free to reach out via email 
or open an issue
