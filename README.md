## Numerical Analysis Course Codes
Welcome to the Numerical Analysis Course Codes repository! 
This repository contains (almost) all the code examples, scripts, and computational exercises 
corresponding to the lectures in the numerical analysis course.

## Requirements
All codes are written in Python. To run the scripts, you will need Python 3.9 (at least) and the following libraries:

numpy
scipy
matplotlib
pandas 

You can install all dependencies with:

```
pip install numpy scipy matplotlib pandas
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

2. **Install project dependencies**:
   ```bash
   npm install
   ```

3. **Install Python dependencies**:
   ```bash
   pip install numpy scipy matplotlib pandas
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

## Usage
Clone the repository:

```
git clone https://github.com/unalmedmathnum/numericalAnalysis_2025/
```

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
