"""
Comprehensive test script for the XL-VLMS Pipeline API.
Tests all endpoints and different scenarios.

Endpoints:
- /run - Fast VLM explainer with pre-built concepts (~30 seconds) ⚡
- /run-full - Full pipeline with concept generation (~30 minutes) ⚠️
"""
import requests
import json
import time
from pathlib import Path
from typing import Optional

API_BASE_URL = "http://localhost:8000"
TEST_IMAGE_PATH = "TKG.png"


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def test_api_reachable():
    """Test if API is running and reachable."""
    print_section("Test 1: API Reachability")
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=5)
        print(f"✅ API is reachable")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return True
    except requests.exceptions.ConnectionError:
        print(f"❌ API not reachable at {API_BASE_URL}")
        print("Make sure to start the API first:")
        print("  cd api")
        print("  python main.py")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_run_fast(image_path: str, custom_prompt: Optional[str] = None):
    """Test FAST image upload with pre-built concepts."""
    print_section("Test 2: Fast Explainer (Pre-built Concepts) ⚡")
    
    if not Path(image_path).exists():
        print(f"❌ Test image not found at {image_path}")
        return False
    
    print(f"📤 Uploading: {Path(image_path).name}")
    print("⚡ Using pre-built concepts (should take ~30 seconds)")
    if custom_prompt:
        print(f"📝 Custom prompt: {custom_prompt}")
    print()
    
    start_time = time.time()
    
    try:
        with open(image_path, "rb") as f:
            files = {"file": (Path(image_path).name, f, "image/jpeg")}
            data = {
                "prompt_mode": "unsupervised",
                "top_n": 5
            }
            if custom_prompt:
                data["prompt"] = custom_prompt
            response = requests.post(
                f"{API_BASE_URL}/run",
                files=files,
                data=data,
                timeout=300  # 5 minutes max
            )
        
        elapsed = time.time() - start_time
        print(f"\n⏱️  Time taken: {elapsed:.1f} seconds")
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("\n✅ Fast explainer completed successfully!")
            print(f"  - Success: {result.get('success')}")
            print(f"  - Message: {result.get('message')}")
            print(f"  - Prompt mode: {result.get('prompt_mode')}")
            print(f"  - Using pre-built concepts: {result.get('using_prebuilt_concepts')}")
            print(f"  - Has vlm_explanations_data: {result.get('vlm_explanations_data') is not None}")
            
            # Check for concept_index
            vlm_data = result.get('vlm_explanations_data', {})
            results_list = vlm_data.get('results', [])
            if results_list:
                per_token = results_list[0].get('per_token_concepts', [])
                if per_token:
                    has_concept_index = 'concept_index' in per_token[0]
                    print(f"  - Has concept_index in tokens: {has_concept_index}")
            
            # Save to file
            output_file = "test_result_fast.json"
            with open(output_file, "w") as f:
                json.dump(result, f, indent=2)
            print(f"\n📄 Full result saved to: {output_file}")
            return True
        else:
            print(f"\n❌ Fast explainer failed!")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print(f"\n❌ Request timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False


def test_run_full_pipeline(image_path: str):
    """Test FULL pipeline with concept generation (SLOW)."""
    print_section("Test 3: Full Pipeline with Concept Generation (SLOW) ⚠️")
    
    if not Path(image_path).exists():
        print(f"❌ Test image not found at {image_path}")
        return False
    
    custom_prompt = "What objects, animals, and items are visible in this image?"
    
    print(f"📤 Uploading: {Path(image_path).name}")
    print(f"📝 Prompt: {custom_prompt}")
    print("⚠️  This will take 30+ minutes (running full pipeline)...")
    print()
    
    start_time = time.time()
    
    try:
        with open(image_path, "rb") as f:
            files = {"file": (Path(image_path).name, f, "image/jpeg")}
            data = {
                "prompt": custom_prompt,
                "prompt_mode": "unsupervised"
            }
            response = requests.post(
                f"{API_BASE_URL}/run-full",
                files=files,
                data=data,
                timeout=7200
            )
        
        elapsed = time.time() - start_time
        print(f"\n⏱️  Time taken: {elapsed/60:.1f} minutes")
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("\n✅ Full pipeline completed successfully!")
            print(f"  - Prompt used: {result.get('prompt_used')}")
            print(f"  - Has concept_data: {result.get('concept_data') is not None}")
            print(f"  - Has vlm_explanations_data: {result.get('vlm_explanations_data') is not None}")
            
            output_file = "test_result_full_pipeline.json"
            with open(output_file, "w") as f:
                json.dump(result, f, indent=2)
            print(f"\n📄 Full result saved to: {output_file}")
            return True
        else:
            print(f"\n❌ Pipeline failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False


def test_invalid_file():
    """Test with non-image file (should fail)."""
    print_section("Test 4: Invalid File Type (Should Fail)")
    
    print("📤 Uploading text file instead of image...")
    
    try:
        # Create temporary text file
        text_content = b"This is not an image"
        files = {"file": ("test.txt", text_content, "text/plain")}
        response = requests.post(
            f"{API_BASE_URL}/run",
            files=files,
            timeout=10
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 400:
            print("✅ Correctly rejected non-image file")
            print(f"Response: {response.json()}")
            return True
        else:
            print(f"❌ Should have rejected but got: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_pipeline_busy():
    """Test concurrent request while pipeline is running."""
    print_section("Test 5: Concurrent Request (Should Return 409)")
    
    print("ℹ️  This test only works if a pipeline is currently running")
    print("Skip this test if no pipeline is running")
    
    try:
        # Try to send a dummy request
        files = {"file": ("dummy.jpg", b"fake", "image/jpeg")}
        response = requests.post(
            f"{API_BASE_URL}/run",
            files=files,
            timeout=5
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 409:
            print("✅ Correctly returned 409 (Pipeline already running)")
            print(f"Response: {response.json()}")
            return True
        else:
            print(f"ℹ️  Got {response.status_code} - no pipeline was running")
            return None  # Not really a failure
            
    except Exception as e:
        print(f"ℹ️  {e}")
        return None


def test_custom_prompt(image_path: str):
    """Test custom prompt parameter (FAST)."""
    print_section("Test 5: Custom Prompt (Fast) ⚡")
    
    if not Path(image_path).exists():
        print(f"❌ Test image not found at {image_path}")
        return False
    
    custom_prompt = "What are the main objects visible in this image? List them."
    
    print(f"📤 Uploading: {Path(image_path).name}")
    print(f"📝 Custom prompt: {custom_prompt}")
    print("⚡ Using pre-built concepts (~30 seconds)")
    
    start_time = time.time()
    
    try:
        with open(image_path, "rb") as f:
            files = {"file": (Path(image_path).name, f, "image/jpeg")}
            data = {
                "prompt_mode": "unsupervised",
                "prompt": custom_prompt,
                "top_n": 5
            }
            response = requests.post(
                f"{API_BASE_URL}/run",
                files=files,
                data=data,
                timeout=300
            )
        
        elapsed = time.time() - start_time
        print(f"\n⏱️  Time taken: {elapsed:.1f} seconds")
        
        if response.status_code == 200:
            result = response.json()
            print("\n✅ Custom prompt test completed!")
            print(f"  - Prompt mode: {result.get('prompt_mode')}")
            print(f"  - Model output: {result.get('vlm_explanations_data', {}).get('results', [{}])[0].get('model_output', 'N/A')}")
            
            output_file = "test_result_custom_prompt.json"
            with open(output_file, "w") as f:
                json.dump(result, f, indent=2)
            print(f"\n📄 Result saved to: {output_file}")
            return True
        else:
            print(f"\n❌ Failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False


def test_binary_mode(image_path: str):
    """Test binary classification mode (FAST)."""
    print_section("Test 6: Binary Classification Mode (Fast) ⚡")
    
    if not Path(image_path).exists():
        print(f"❌ Test image not found at {image_path}")
        return False
    
    print(f"📤 Uploading: {Path(image_path).name}")
    print(f"🎯 Mode: binary, Label: cat")
    print("⚡ Using pre-built concepts (~30 seconds)")
    
    start_time = time.time()
    
    try:
        with open(image_path, "rb") as f:
            files = {"file": (Path(image_path).name, f, "image/jpeg")}
            data = {
                "prompt_mode": "binary",
                "label": "cat",
                "top_n": 5
            }
            response = requests.post(
                f"{API_BASE_URL}/run",
                files=files,
                data=data,
                timeout=300
            )
        
        elapsed = time.time() - start_time
        print(f"\n⏱️  Time taken: {elapsed:.1f} seconds")
        
        if response.status_code == 200:
            result = response.json()
            print("\n✅ Binary mode completed!")
            print(f"  - Prompt mode: {result.get('prompt_mode')}")
            
            output_file = "test_result_binary.json"
            with open(output_file, "w") as f:
                json.dump(result, f, indent=2)
            print(f"\n📄 Result saved to: {output_file}")
            return True
        else:
            print(f"\n❌ Failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False


def interactive_menu():
    """Show interactive menu for test selection."""
    print("\n" + "🚀"*35)
    print("  XL-VLMS Pipeline API Test Suite")
    print("🚀"*35)
    
    print("""
Select tests to run:

1. Test API Reachability (quick)
2. Fast Explainer - Pre-built concepts (~30 sec) ⚡
3. Full Pipeline - Generate concepts (~30 min) ⚠️
4. Invalid File Type Test (quick)
5. Custom Prompt Test (~30 sec) ⚡
6. Binary Classification Mode (~30 sec) ⚡
7. Run ALL quick tests (1, 2, 4, 5, 6)
8. Run ALL tests including SLOW (WARNING: Takes 30+ minutes!)
0. Exit

Note: Test 3 runs the full pipeline and takes 30+ minutes.
Tests 2, 5 & 6 are FAST (use pre-built concepts).
""")
    
    results = {}
    
    while True:
        choice = input("Enter choice (0-8): ").strip()
        
        if choice == "0":
            print("\n👋 Exiting...")
            break
            
        elif choice == "1":
            results["reachability"] = test_api_reachable()
            
        elif choice == "2":
            results["fast"] = test_run_fast(TEST_IMAGE_PATH)
                
        elif choice == "3":
            if confirm_long_test():
                results["full_pipeline"] = test_run_full_pipeline(TEST_IMAGE_PATH)
                
        elif choice == "4":
            results["invalid_file"] = test_invalid_file()
            
        elif choice == "5":
            results["custom_prompt"] = test_custom_prompt(TEST_IMAGE_PATH)
            
        elif choice == "6":
            results["binary"] = test_binary_mode(TEST_IMAGE_PATH)
                
        elif choice == "7":
            print("\n🏃 Running quick tests...")
            results["reachability"] = test_api_reachable()
            results["fast"] = test_run_fast(TEST_IMAGE_PATH)
            results["invalid_file"] = test_invalid_file()
            results["custom_prompt"] = test_custom_prompt(TEST_IMAGE_PATH)
            results["binary"] = test_binary_mode(TEST_IMAGE_PATH)
            
        elif choice == "8":
            if confirm_long_test("ALL tests"):
                results["reachability"] = test_api_reachable()
                if results["reachability"]:
                    results["fast"] = test_run_fast(TEST_IMAGE_PATH)
                    results["invalid_file"] = test_invalid_file()
                    results["custom_prompt"] = test_custom_prompt(TEST_IMAGE_PATH)
                    results["binary"] = test_binary_mode(TEST_IMAGE_PATH)
                    results["full_pipeline"] = test_run_full_pipeline(TEST_IMAGE_PATH)
        else:
            print("❌ Invalid choice")
            continue
        
        print("\n" + "-"*70)
        print_results_summary(results)
        print("-"*70)
        print("\nPress Enter to continue...")
        input()


def confirm_long_test(test_name: str = "This test") -> bool:
    """Ask user to confirm long-running test."""
    response = input(f"\n⚠️  {test_name} will take 30+ minutes. Continue? (y/N): ")
    return response.strip().lower() == 'y'


def print_results_summary(results: dict):
    """Print summary of test results."""
    if not results:
        return
    
    print("\n📊 Test Results Summary:")
    for test_name, result in results.items():
        if result is True:
            status = "✅ PASS"
        elif result is False:
            status = "❌ FAIL"
        else:
            status = "⚠️  SKIP"
        print(f"  {status} - {test_name}")


if __name__ == "__main__":
    interactive_menu()
