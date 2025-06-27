#!/bin/bash

# 🚀 Google Cloud Setup Script for DoctAI Health Hub
# This script automates the Google Cloud Console setup process

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
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

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if user is authenticated
check_auth() {
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
        print_error "You are not authenticated with Google Cloud. Please run:"
        echo "gcloud auth login"
        exit 1
    fi
}

# Function to get current project
get_current_project() {
    gcloud config get-value project 2>/dev/null || echo ""
}

# Function to create project
create_project() {
    local project_name="$1"
    local project_id="$2"

    print_status "Creating Google Cloud project: $project_name ($project_id)"

    if gcloud projects create "$project_id" --name="$project_name" --quiet; then
        print_success "Project created successfully"
        return 0
    else
        print_error "Failed to create project. It might already exist."
        return 1
    fi
}

# Function to enable APIs
enable_apis() {
    local project_id="$1"

    print_status "Enabling required APIs for project: $project_id"

    # Set the project
    gcloud config set project "$project_id"

    # List of APIs to enable
    local apis=(
        "healthcare.googleapis.com"
        "vision.googleapis.com"
        "storage.googleapis.com"
        "cloudfunctions.googleapis.com"
        "run.googleapis.com"
        "compute.googleapis.com"
        "iam.googleapis.com"
    )

    for api in "${apis[@]}"; do
        print_status "Enabling API: $api"
        if gcloud services enable "$api" --quiet; then
            print_success "Enabled $api"
        else
            print_warning "Failed to enable $api (might already be enabled)"
        fi
    done
}

# Function to create service account
create_service_account() {
    local project_id="$1"
    local service_account_name="$2"
    local service_account_email="$service_account_name@$project_id.iam.gserviceaccount.com"

    print_status "Creating service account: $service_account_name"

    # Create service account
    if gcloud iam service-accounts create "$service_account_name" \
        --display-name="DoctAI Healthcare Service Account" \
        --description="Service account for DoctAI Health Hub healthcare operations" \
        --quiet; then
        print_success "Service account created"
    else
        print_warning "Service account might already exist"
    fi

    # Assign roles
    local roles=(
        "roles/healthcare.datasetAdmin"
        "roles/healthcare.dicomStoreAdmin"
        "roles/healthcare.fhirStoreAdmin"
        "roles/healthcare.hl7V2StoreAdmin"
        "roles/storage.admin"
        "roles/vision.user"
        "roles/cloudfunctions.developer"
        "roles/run.developer"
    )

    for role in "${roles[@]}"; do
        print_status "Assigning role: $role"
        if gcloud projects add-iam-policy-binding "$project_id" \
            --member="serviceAccount:$service_account_email" \
            --role="$role" \
            --quiet; then
            print_success "Assigned $role"
        else
            print_warning "Failed to assign $role (might already be assigned)"
        fi
    done

    # Create and download key
    print_status "Creating service account key"
    local key_file="doctai-healthcare-service-key.json"

    if gcloud iam service-accounts keys create "$key_file" \
        --iam-account="$service_account_email" \
        --quiet; then
        print_success "Service account key created: $key_file"
        print_warning "Keep this file secure and never commit it to version control!"
    else
        print_error "Failed to create service account key"
        return 1
    fi
}

# Function to create healthcare dataset
create_healthcare_dataset() {
    local project_id="$1"
    local location="$2"
    local dataset_id="$3"

    print_status "Creating healthcare dataset: $dataset_id"

    if gcloud healthcare datasets create "$dataset_id" \
        --location="$location" \
        --quiet; then
        print_success "Healthcare dataset created"
    else
        print_warning "Dataset might already exist"
    fi
}

# Function to create DICOM stores
create_dicom_stores() {
    local project_id="$1"
    local location="$2"
    local dataset_id="$3"

    local stores=(
        "chest-xray-store"
        "ct-scan-store"
        "mri-store"
        "skin-lesion-store"
    )

    for store in "${stores[@]}"; do
        print_status "Creating DICOM store: $store"
        if gcloud healthcare dicom-stores create "$store" \
            --dataset="$dataset_id" \
            --location="$location" \
            --quiet; then
            print_success "Created DICOM store: $store"
        else
            print_warning "DICOM store $store might already exist"
        fi
    done
}

# Function to create storage bucket
create_storage_bucket() {
    local project_id="$1"
    local location="$2"
    local bucket_name="$3"

    print_status "Creating storage bucket: $bucket_name"

    if gsutil mb -l "$location" "gs://$bucket_name"; then
        print_success "Storage bucket created"

        # Set bucket permissions
        local service_account_email="doctai-healthcare-service@$project_id.iam.gserviceaccount.com"
        print_status "Setting bucket permissions for service account"

        if gsutil iam ch "serviceAccount:$service_account_email:objectAdmin" "gs://$bucket_name"; then
            print_success "Bucket permissions set"
        else
            print_warning "Failed to set bucket permissions"
        fi
    else
        print_warning "Storage bucket might already exist"
    fi
}

# Function to update .env file
update_env_file() {
    local project_id="$1"
    local location="$2"
    local dataset_id="$3"
    local bucket_name="$4"

    print_status "Updating .env file with Google Cloud configuration"

    # Create .env file if it doesn't exist
    if [ ! -f .env ]; then
        touch .env
    fi

    # Function to update or add environment variable
    update_env_var() {
        local key="$1"
        local value="$2"

        if grep -q "^$key=" .env; then
            # Update existing variable
            sed -i.bak "s/^$key=.*/$key=$value/" .env
        else
            # Add new variable
            echo "$key=$value" >> .env
        fi
    }

    # Update Google Cloud configuration
    update_env_var "VITE_USE_CLOUD_HEALTHCARE" "true"
    update_env_var "VITE_ENABLE_GOOGLE_HEALTHCARE" "true"
    update_env_var "VITE_PRIMARY_CLOUD_PROVIDER" "google"
    update_env_var "VITE_GOOGLE_HEALTHCARE_PROJECT_ID" "$project_id"
    update_env_var "VITE_GOOGLE_HEALTHCARE_LOCATION" "$location"
    update_env_var "VITE_GOOGLE_HEALTHCARE_DATASET_ID" "$dataset_id"
    update_env_var "VITE_GOOGLE_CLOUD_STORAGE_BUCKET" "$bucket_name"
    update_env_var "GOOGLE_APPLICATION_CREDENTIALS" "./doctai-healthcare-service-key.json"
    update_env_var "VITE_CLOUD_HEALTHCARE_TIMEOUT" "30000"
    update_env_var "VITE_CLOUD_HEALTHCARE_MAX_RETRIES" "3"
    update_env_var "VITE_CLOUD_HEALTHCARE_DEBUG" "true"
    update_env_var "VITE_CLOUD_HEALTHCARE_LOG_REQUESTS" "true"

    print_success ".env file updated"
}

# Function to create test script
create_test_script() {
    print_status "Creating Google Cloud test script"

    cat > scripts/test-google-cloud.js << 'EOF'
// Test script for Google Cloud Healthcare setup
const { CloudHealthcareService } = require('../src/services/cloudHealthcare');

async function testGoogleHealthcare() {
    try {
        console.log('🧪 Testing Google Cloud Healthcare setup...');

        const config = {
            googleHealthcare: {
                projectId: process.env.VITE_GOOGLE_HEALTHCARE_PROJECT_ID,
                location: process.env.VITE_GOOGLE_HEALTHCARE_LOCATION,
                datasetId: process.env.VITE_GOOGLE_HEALTHCARE_DATASET_ID
            }
        };

        const service = new CloudHealthcareService(config);

        // Test dataset access
        const status = await service.getServiceStatus();
        console.log('✅ Google Healthcare Status:', status);

        console.log('🎉 Google Cloud Healthcare setup is working correctly!');
        return true;
    } catch (error) {
        console.error('❌ Google Healthcare Test Failed:', error);
        return false;
    }
}

// Run test if called directly
if (require.main === module) {
    testGoogleHealthcare();
}

module.exports = { testGoogleHealthcare };
EOF

    print_success "Test script created: scripts/test-google-cloud.js"
}

# Function to display summary
display_summary() {
    local project_id="$1"
    local location="$2"
    local dataset_id="$3"
    local bucket_name="$4"

    echo ""
    echo "🎉 Google Cloud Setup Complete!"
    echo "================================"
    echo ""
    echo "📋 Configuration Summary:"
    echo "  Project ID: $project_id"
    echo "  Location: $location"
    echo "  Dataset ID: $dataset_id"
    echo "  Storage Bucket: $bucket_name"
    echo ""
    echo "🔑 Next Steps:"
    echo "  1. Review the generated .env file"
    echo "  2. Keep the service account key file secure"
    echo "  3. Test the setup with: node scripts/test-google-cloud.js"
    echo "  4. Set up billing alerts in Google Cloud Console"
    echo "  5. Review the GOOGLE_CLOUD_SETUP_GUIDE.md for additional configuration"
    echo ""
    echo "⚠️  Important Security Notes:"
    echo "  - Never commit the service account key to version control"
    echo "  - The .env file is already added to .gitignore"
    echo "  - Consider setting up VPC for additional security"
    echo ""
}

# Main function
main() {
    echo "🚀 Google Cloud Setup Script for DoctAI Health Hub"
    echo "=================================================="
    echo ""

    # Check if gcloud CLI is installed
    if ! command_exists gcloud; then
        print_error "Google Cloud CLI (gcloud) is not installed."
        echo "Please install it from: https://cloud.google.com/sdk/docs/install"
        exit 1
    fi

    # Check authentication
    check_auth

    # Get current user
    local current_user=$(gcloud auth list --filter=status:ACTIVE --format="value(account)")
    print_status "Authenticated as: $current_user"

    # Get or create project
    local project_name="DoctAI Health Hub"
    local project_id="doctai-health-hub-$(date +%s)"
    local current_project=$(get_current_project)

    if [ -n "$current_project" ]; then
        print_status "Current project: $current_project"
        read -p "Do you want to use the current project? (y/n): " use_current

        if [[ $use_current =~ ^[Yy]$ ]]; then
            project_id="$current_project"
        else
            read -p "Enter project ID (or press Enter for auto-generated): " custom_project_id
            if [ -n "$custom_project_id" ]; then
                project_id="$custom_project_id"
            fi
        fi
    else
        read -p "Enter project ID (or press Enter for auto-generated): " custom_project_id
        if [ -n "$custom_project_id" ]; then
            project_id="$custom_project_id"
        fi
    fi

    # Get location
    local location="us-central1"
    read -p "Enter location (default: us-central1): " custom_location
    if [ -n "$custom_location" ]; then
        location="$custom_location"
    fi

    # Get dataset ID
    local dataset_id="doctai-healthcare-dataset"
    read -p "Enter dataset ID (default: doctai-healthcare-dataset): " custom_dataset_id
    if [ -n "$custom_dataset_id" ]; then
        dataset_id="$custom_dataset_id"
    fi

    # Get bucket name
    local bucket_name="doctai-health-hub-images"
    read -p "Enter storage bucket name (default: doctai-health-hub-images): " custom_bucket_name
    if [ -n "$custom_bucket_name" ]; then
        bucket_name="$custom_bucket_name"
    fi

    echo ""
    print_status "Starting Google Cloud setup with the following configuration:"
    echo "  Project ID: $project_id"
    echo "  Location: $location"
    echo "  Dataset ID: $dataset_id"
    echo "  Storage Bucket: $bucket_name"
    echo ""

    read -p "Continue with this configuration? (y/n): " confirm
    if [[ ! $confirm =~ ^[Yy]$ ]]; then
        print_status "Setup cancelled"
        exit 0
    fi

    # Create project if needed
    if [ "$project_id" != "$current_project" ]; then
        create_project "$project_name" "$project_id"
    fi

    # Enable APIs
    enable_apis "$project_id"

    # Create service account
    create_service_account "$project_id" "doctai-healthcare-service"

    # Create healthcare dataset
    create_healthcare_dataset "$project_id" "$location" "$dataset_id"

    # Create DICOM stores
    create_dicom_stores "$project_id" "$location" "$dataset_id"

    # Create storage bucket
    create_storage_bucket "$project_id" "$location" "$bucket_name"

    # Update .env file
    update_env_file "$project_id" "$location" "$dataset_id" "$bucket_name"

    # Create test script
    create_test_script

    # Display summary
    display_summary "$project_id" "$location" "$dataset_id" "$bucket_name"
}

# Run main function
main "$@"
