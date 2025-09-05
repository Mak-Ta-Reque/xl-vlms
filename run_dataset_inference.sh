#!/bin/bash

# Dataset Inference Runner Script
# This script provides an easy way to run dataset inference with various configurations

set -e  # Exit on any error

# Default values
DATASET_PATH="/mnt/abka03/Projects/xl-vlms/data/train"
MODEL_NAME="google/gemma-3n-e4b"
OUTPUT_CSV="dataset_inference_results.csv"
PROMPT="Describe this image."
HF_TOKEN=""
IMAGE_SIZE=""
TRUST_REMOTE_CODE="--trust_remote_code"
RESUME=""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -d, --dataset-path PATH     Path to the dataset directory (required)"
    echo "  -m, --model-name NAME       Model name or path (default: google/gemma-3n-E4B)"
    echo "  -o, --output-csv PATH       Output CSV file path (default: dataset_inference_results.csv)"
    echo "  -p, --prompt TEXT           Text prompt for the model (default: 'Describe this image.')"
    echo "  -t, --hf-token TOKEN        Hugging Face authentication token for private models"
    echo "  -s, --image-size WxH        Resize images to WxH (e.g., 512x512)"
    echo "  -r, --resume                Resume from existing CSV file"
    echo "  --no-trust-remote-code      Don't trust remote code when loading models"
    echo "  --list-models               List available model shortcuts"
    echo "  -h, --help                  Show this help message"
    echo ""
    echo "Examples:"
    echo "  # Basic usage"
    echo "  $0 -d ./data/train -m google/gemma-3n-E4B -o results.csv"
    echo ""
    echo "  # With authentication token for private models"
    echo "  $0 -d ./data/train -m google/gemma-3n-E4B-it -t your_hf_token"
    echo ""
    echo "  # With image resizing and resume"
    echo "  $0 -d ./data/train -m qwen2-vl-7b -s 512x512 -r"
    echo ""
    echo "  # Custom prompt"
    echo "  $0 -d ./data/train -p 'What objects do you see in this image?'"
    echo ""
    echo "Environment Variables:"
    echo "  HF_TOKEN                    Hugging Face token (alternative to -t option)"
}

# Function to list available models
list_models() {
    python3 inference/dataset_inference.py --list_models
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--dataset-path)
            DATASET_PATH="$2"
            shift 2
            ;;
        -m|--model-name)
            MODEL_NAME="$2"
            shift 2
            ;;
        -o|--output-csv)
            OUTPUT_CSV="$2"
            shift 2
            ;;
        -p|--prompt)
            PROMPT="$2"
            shift 2
            ;;
        -t|--hf-token)
            HF_TOKEN="$2"
            shift 2
            ;;
        -s|--image-size)
            if [[ $2 =~ ^[0-9]+x[0-9]+$ ]]; then
                WIDTH=$(echo $2 | cut -d'x' -f1)
                HEIGHT=$(echo $2 | cut -d'x' -f2)
                IMAGE_SIZE="--image_size $WIDTH $HEIGHT"
            else
                print_error "Invalid image size format. Use WIDTHxHEIGHT (e.g., 512x512)"
                exit 1
            fi
            shift 2
            ;;
        -r|--resume)
            RESUME="--resume"
            shift
            ;;
        --no-trust-remote-code)
            TRUST_REMOTE_CODE="--no_trust_remote_code"
            shift
            ;;
        --list-models)
            list_models
            exit 0
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$DATASET_PATH" ]]; then
    print_error "Dataset path is required!"
    show_usage
    exit 1
fi

# Check if dataset path exists
if [[ ! -d "$DATASET_PATH" ]]; then
    print_error "Dataset path does not exist: $DATASET_PATH"
    exit 1
fi

# Check if the inference script exists
if [[ ! -f "inference/dataset_inference.py" ]]; then
    print_error "Dataset inference script not found: inference/dataset_inference.py"
    print_error "Please run this script from the project root directory"
    exit 1
fi

# Build the command
CMD="python3 inference/dataset_inference.py"
CMD="$CMD --dataset_path \"$DATASET_PATH\""
CMD="$CMD --model_name \"$MODEL_NAME\""
CMD="$CMD --output_csv \"$OUTPUT_CSV\""
CMD="$CMD --prompt \"$PROMPT\""
CMD="$CMD $TRUST_REMOTE_CODE"

if [[ -n "$HF_TOKEN" ]]; then
    CMD="$CMD --hf_token \"$HF_TOKEN\""
fi

if [[ -n "$IMAGE_SIZE" ]]; then
    CMD="$CMD $IMAGE_SIZE"
fi

if [[ -n "$RESUME" ]]; then
    CMD="$CMD $RESUME"
fi

# Print configuration
print_info "Starting dataset inference with the following configuration:"
echo "  Dataset Path: $DATASET_PATH"
echo "  Model Name: $MODEL_NAME"
echo "  Output CSV: $OUTPUT_CSV"
echo "  Prompt: $PROMPT"
if [[ -n "$HF_TOKEN" ]]; then
    echo "  HF Token: ****** (provided)"
fi
if [[ -n "$IMAGE_SIZE" ]]; then
    echo "  Image Size: $IMAGE_SIZE"
fi
if [[ -n "$RESUME" ]]; then
    echo "  Resume: Yes"
fi
echo "  Trust Remote Code: $(if [[ "$TRUST_REMOTE_CODE" == "--trust_remote_code" ]]; then echo "Yes"; else echo "No"; fi)"
echo ""

# Create output directory if it doesn't exist
OUTPUT_DIR=$(dirname "$OUTPUT_CSV")
if [[ ! -d "$OUTPUT_DIR" && "$OUTPUT_DIR" != "." ]]; then
    print_info "Creating output directory: $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
fi

# Execute the command
print_info "Executing command:"
echo "$CMD"
echo ""

eval $CMD

# Check if the command was successful
if [[ $? -eq 0 ]]; then
    print_success "Dataset inference completed successfully!"
    print_success "Results saved to: $OUTPUT_CSV"
else
    print_error "Dataset inference failed!"
    exit 1
fi
