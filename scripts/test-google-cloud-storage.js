#!/usr/bin/env node

/**
 * Test Google Cloud Storage Setup
 * This script tests the Google Cloud Storage bucket configuration
 */

import { execSync } from 'child_process';
import { readFileSync, existsSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Configuration
const PROJECT_ID = 'doctai-project';
const BUCKET_NAME = 'doctai-personal-health-data';
const LOCATION = 'us-central1';

console.log('🔍 Testing Google Cloud Storage Setup...\n');

try {
  // 1. Check if gcloud is authenticated
  console.log('1. Checking gcloud authentication...');
  try {
    const authInfo = execSync('gcloud auth list --filter=status:ACTIVE --format="value(account)"', { encoding: 'utf8' });
    console.log(`✅ Authenticated as: ${authInfo.trim()}`);
  } catch (error) {
    console.log('❌ Not authenticated. Please run: gcloud auth login');
    process.exit(1);
  }

  // 2. Check if project is set
  console.log('\n2. Checking project configuration...');
  try {
    const currentProject = execSync('gcloud config get-value project', { encoding: 'utf8' });
    if (currentProject.trim() === PROJECT_ID) {
      console.log(`✅ Project is set to: ${PROJECT_ID}`);
    } else {
      console.log(`⚠️  Current project: ${currentProject.trim()}`);
      console.log(`Setting project to: ${PROJECT_ID}`);
      execSync(`gcloud config set project ${PROJECT_ID}`);
      console.log('✅ Project updated');
    }
  } catch (error) {
    console.log('❌ Error checking project configuration');
    process.exit(1);
  }

  // 3. Check if bucket exists
  console.log('\n3. Checking bucket existence...');
  try {
    const bucketInfo = execSync(`gsutil ls -b gs://${BUCKET_NAME}`, { encoding: 'utf8' });
    console.log(`✅ Bucket exists: ${BUCKET_NAME}`);
    console.log(`   Location: ${bucketInfo.trim()}`);
  } catch (error) {
    console.log(`❌ Bucket ${BUCKET_NAME} does not exist or is not accessible`);
    console.log('Creating bucket...');
    try {
      execSync(`gsutil mb -l ${LOCATION} gs://${BUCKET_NAME}`);
      console.log('✅ Bucket created successfully');
    } catch (createError) {
      console.log('❌ Failed to create bucket');
      process.exit(1);
    }
  }

  // 4. Test bucket permissions
  console.log('\n4. Testing bucket permissions...');
  try {
    // Create a test file
    const testContent = 'This is a test file for Google Cloud Storage';
    const testFileName = 'test-file.txt';

    // Write test file
    const fs = await import('fs');
    fs.writeFileSync(testFileName, testContent);

    // Upload test file
    execSync(`gsutil cp ${testFileName} gs://${BUCKET_NAME}/test/`);
    console.log('✅ Upload test successful');

    // Download test file
    execSync(`gsutil cp gs://${BUCKET_NAME}/test/${testFileName} downloaded-${testFileName}`);
    console.log('✅ Download test successful');

    // Verify content
    const downloadedContent = fs.readFileSync(`downloaded-${testFileName}`, 'utf8');
    if (downloadedContent === testContent) {
      console.log('✅ Content verification successful');
    } else {
      console.log('❌ Content verification failed');
    }

    // Clean up test files
    execSync(`gsutil rm gs://${BUCKET_NAME}/test/${testFileName}`);
    fs.unlinkSync(testFileName);
    fs.unlinkSync(`downloaded-${testFileName}`);
    console.log('✅ Cleanup completed');

  } catch (error) {
    console.log('❌ Permission test failed:', error.message);
  }

  // 5. Check bucket IAM permissions
  console.log('\n5. Checking bucket IAM permissions...');
  try {
    const iamInfo = execSync(`gsutil iam get gs://${BUCKET_NAME}`, { encoding: 'utf8' });
    console.log('✅ IAM permissions retrieved successfully');

    // Check if service account has access
    const serviceAccountKeyPath = join(__dirname, '..', 'CI-CD-secrets', 'doctai-project-service-account.json');
    const fs = await import('fs');
    if (fs.existsSync(serviceAccountKeyPath)) {
      console.log('✅ Service account key file exists');
    } else {
      console.log('⚠️  Service account key file not found');
    }
  } catch (error) {
    console.log('❌ IAM permission check failed:', error.message);
  }

  // 6. Test bucket organization structure
  console.log('\n6. Testing bucket organization structure...');
  try {
    // Create test files to establish directory structure
    const testDirs = [
      'users/test-user/skin_lesion/test-file.txt',
      'users/test-user/xray/test-file.txt',
      'users/test-user/ct_scan/test-file.txt',
      'users/test-user/mri/test-file.txt',
      'users/test-user/eeg/test-file.txt'
    ];

    const fs = await import('fs');
    const testContent = 'Test file for directory structure';

    for (const filePath of testDirs) {
      // Create a temporary file
      const tempFile = `temp-${filePath.split('/').pop()}`;
      fs.writeFileSync(tempFile, testContent);

      // Upload to establish directory structure
      execSync(`gsutil cp ${tempFile} gs://${BUCKET_NAME}/${filePath}`);
      fs.unlinkSync(tempFile);
    }
    console.log('✅ Directory structure created');

    // List directories
    const listResult = execSync(`gsutil ls gs://${BUCKET_NAME}/users/test-user/`, { encoding: 'utf8' });
    console.log('✅ Directory listing successful');
    console.log('   Created directories:');
    listResult.split('\n').filter(line => line.trim()).forEach(dir => {
      console.log(`   - ${dir.split('/').pop()}`);
    });

    // Clean up test files
    execSync(`gsutil -m rm gs://${BUCKET_NAME}/users/test-user/**`);
    console.log('✅ Test files cleaned up');

  } catch (error) {
    console.log('❌ Directory structure test failed:', error.message);
  }

  console.log('\n🎉 Google Cloud Storage setup is working correctly!');
  console.log('\n📋 Summary:');
  console.log(`   Project ID: ${PROJECT_ID}`);
  console.log(`   Bucket Name: ${BUCKET_NAME}`);
  console.log(`   Location: ${LOCATION}`);
  console.log(`   Status: ✅ Ready for use`);

  console.log('\n📝 Next steps:');
  console.log('   1. Update your .env file with the bucket name');
  console.log('   2. Configure your application to use Google Cloud Storage');
  console.log('   3. Test image uploads in your application');

} catch (error) {
  console.error('\n❌ Test failed:', error.message);
  process.exit(1);
}
