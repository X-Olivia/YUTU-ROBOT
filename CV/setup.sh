 #!/bin/bash

echo "Setting up Detection and Tracking Environment..."

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

echo "Python 3 found: $(python3 --version)"

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Environment setup complete!"
echo "To activate virtual environment, run: source venv/bin/activate"
echo "Ready to run detection and tracking!" 