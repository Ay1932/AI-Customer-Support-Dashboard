# Contributing to AI Customer Support Dashboard

Thank you for your interest in contributing to the AI Customer Support Dashboard project! We welcome contributions from the community.

## How to Contribute

### 1. Fork the Repository
- Click the "Fork" button on GitHub
- Clone your fork locally: `git clone https://github.com/your-username/AI-Customer-Support-Dashboard.git`

### 2. Create a Feature Branch
```bash
git checkout -b feature/your-feature-name
```

### 3. Make Your Changes
- Follow the existing code style and structure
- Add tests for new features
- Update documentation as needed

### 4. Test Your Changes
```bash
# Run the CI checks locally
pip install -r requirements.txt
python -c "import data_preprocessing; import train_models; import app"
```

### 5. Commit and Push
```bash
git add .
git commit -m "Add your descriptive commit message"
git push origin feature/your-feature-name
```

### 6. Create a Pull Request
- Go to the original repository on GitHub
- Click "New Pull Request"
- Select your feature branch
- Provide a clear description of your changes

## Development Guidelines

### Code Style
- Follow PEP 8 Python style guidelines
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Keep functions focused on single responsibilities

### Testing
- Test your changes thoroughly
- Ensure the dashboard runs without errors
- Verify model training and prediction functionality

### Documentation
- Update README.md for significant changes
- Add comments for complex logic
- Update requirements.txt if new dependencies are added

## Reporting Issues

When reporting bugs or requesting features:
- Use the provided issue templates
- Provide clear steps to reproduce bugs
- Include relevant error messages and screenshots
- Specify your Python version and operating system

## Code of Conduct

Please be respectful and constructive in all interactions. We aim to foster an inclusive and welcoming community.

## Questions?

If you have questions about contributing, feel free to open an issue or contact the maintainers.