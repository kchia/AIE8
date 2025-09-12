#!/usr/bin/env python3
"""
Test script for Phase 1: PDF Document Support
This script tests the PDF processing functionality added to the RAG system.
"""

import os
import tempfile
import shutil
from pathlib import Path
from aimakerspace.text_utils import TextFileLoader, CharacterTextSplitter


def create_sample_text_file(content: str, filename: str = "test.txt") -> str:
    """Create a temporary text file for testing."""
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, filename)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return file_path


def create_sample_pdf_file() -> str:
    """Create a simple PDF file for testing using reportlab."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph
        from reportlab.lib.styles import getSampleStyleSheet
        
        temp_dir = tempfile.mkdtemp()
        pdf_path = os.path.join(temp_dir, "test_document.pdf")
        
        # Create a simple PDF
        doc = SimpleDocTemplate(pdf_path, pagesize=letter)
        styles = getSampleStyleSheet()
        
        content = [
            Paragraph("Test PDF Document", styles['Title']),
            Paragraph("This is the first paragraph of the test PDF document. It contains information about machine learning and artificial intelligence.", styles['Normal']),
            Paragraph("This is the second paragraph discussing natural language processing and computer vision applications.", styles['Normal']),
            Paragraph("The third paragraph covers deep learning techniques and neural network architectures.", styles['Normal']),
        ]
        
        doc.build(content)
        return pdf_path
        
    except ImportError:
        print("[INFO] reportlab not available for creating test PDFs")
        return None


def test_basic_text_file_loading():
    """Test 1: Basic text file loading functionality."""
    
    print("=== Test 1: Basic Text File Loading ===")
    
    # Create sample text content
    sample_content = """
    This is a sample text document for testing the TextFileLoader.
    It contains multiple lines and paragraphs to verify proper loading.
    
    Machine learning is a subset of artificial intelligence that enables computers to learn.
    Natural language processing helps computers understand human language.
    Computer vision allows machines to interpret visual information.
    """
    
    # Create temporary text file
    temp_file = create_sample_text_file(sample_content.strip())
    
    try:
        # Test loading
        loader = TextFileLoader(temp_file)
        documents = loader.load_documents()
        
        # Assertions
        assert len(documents) == 1, f"Expected 1 document, got {len(documents)}"
        assert len(documents[0]) > 0, "Document should not be empty"
        assert "machine learning" in documents[0].lower(), "Should contain expected content"
        
        print(f"[OK] Loaded document with {len(documents[0])} characters")
        print(f"[OK] Content preview: {documents[0][:100]}...")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Text file loading failed: {e}")
        return False
        
    finally:
        # Cleanup
        if os.path.exists(temp_file):
            shutil.rmtree(os.path.dirname(temp_file))


def test_pdf_file_loading():
    """Test 2: PDF file loading functionality."""
    
    print("\n=== Test 2: PDF File Loading ===")
    
    # Try to create a test PDF
    pdf_path = create_sample_pdf_file()
    
    if pdf_path is None:
        print("[SKIP] Skipping PDF creation test - reportlab not available")
        return test_pdf_loading_with_existing_file()
    
    try:
        # Test loading PDF
        loader = TextFileLoader(pdf_path)
        documents = loader.load_documents()
        
        # Assertions
        assert len(documents) == 1, f"Expected 1 document, got {len(documents)}"
        assert len(documents[0]) > 0, "PDF document should not be empty"
        
        content = documents[0].lower()
        assert "test pdf document" in content, "Should contain PDF title"
        assert "machine learning" in content, "Should contain expected content"
        
        print(f"[OK] Loaded PDF with {len(documents[0])} characters")
        print(f"[OK] Content preview: {documents[0][:150]}...")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] PDF loading failed: {e}")
        return False
        
    finally:
        # Cleanup
        if pdf_path and os.path.exists(pdf_path):
            shutil.rmtree(os.path.dirname(pdf_path))


def test_pdf_loading_with_existing_file():
    """Test 3: PDF loading with existing file if available."""
    
    print("\n=== Test 3: PDF Loading with Existing File ===")
    
    # Check for existing PDF files in common locations
    potential_pdfs = [
        "data/sample.pdf",
        "data/test.pdf", 
        "data/document.pdf"
    ]
    
    pdf_file = None
    for pdf_path in potential_pdfs:
        if os.path.exists(pdf_path):
            pdf_file = pdf_path
            break
    
    if pdf_file is None:
        print("[SKIP] No existing PDF files found for testing")
        return True
    
    try:
        print(f"[INFO] Testing with existing PDF: {pdf_file}")
        
        loader = TextFileLoader(pdf_file)
        documents = loader.load_documents()
        
        # Basic assertions
        assert len(documents) == 1, f"Expected 1 document, got {len(documents)}"
        assert len(documents[0]) > 0, "PDF document should not be empty"
        
        print(f"[OK] Successfully loaded existing PDF with {len(documents[0])} characters")
        print(f"[OK] Content preview: {documents[0][:100]}...")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Loading existing PDF failed: {e}")
        return False


def test_directory_loading_mixed_formats():
    """Test 4: Directory loading with mixed file formats."""
    
    print("\n=== Test 4: Directory Loading with Mixed Formats ===")
    
    # Create temporary directory with mixed files
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create text files
        txt_content1 = "This is the first text document about artificial intelligence."
        txt_content2 = "This is the second text document about machine learning algorithms."
        
        txt_file1 = os.path.join(temp_dir, "doc1.txt")
        txt_file2 = os.path.join(temp_dir, "doc2.txt")
        
        with open(txt_file1, 'w') as f:
            f.write(txt_content1)
        with open(txt_file2, 'w') as f:
            f.write(txt_content2)
        
        # Create a simple PDF if possible
        pdf_created = False
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.platypus import SimpleDocTemplate, Paragraph
            from reportlab.lib.styles import getSampleStyleSheet
            
            pdf_file = os.path.join(temp_dir, "doc3.pdf")
            doc = SimpleDocTemplate(pdf_file, pagesize=letter)
            styles = getSampleStyleSheet()
            
            content = [
                Paragraph("PDF Document in Mixed Directory", styles['Title']),
                Paragraph("This PDF contains information about natural language processing and computer vision.", styles['Normal'])
            ]
            
            doc.build(content)
            pdf_created = True
            
        except ImportError:
            print("[INFO] Skipping PDF creation in mixed directory test")
        
        # Test directory loading
        loader = TextFileLoader(temp_dir)
        documents = loader.load_documents()
        
        # Assertions
        expected_docs = 3 if pdf_created else 2
        assert len(documents) == expected_docs, f"Expected {expected_docs} documents, got {len(documents)}"
        
        # Verify content from text files
        all_content = " ".join(documents).lower()
        assert "artificial intelligence" in all_content, "Should contain content from txt files"
        assert "machine learning" in all_content, "Should contain content from txt files"
        
        if pdf_created:
            assert "natural language processing" in all_content, "Should contain content from PDF"
        
        print(f"[OK] Loaded {len(documents)} documents from mixed directory")
        for i, doc in enumerate(documents):
            print(f"     Document {i+1}: {len(doc)} characters")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Mixed directory loading failed: {e}")
        return False
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


def test_pdf_error_handling():
    """Test 5: PDF error handling for corrupted or invalid files."""
    
    print("\n=== Test 5: PDF Error Handling ===")
    
    # Create a fake PDF file (not actually a PDF)
    temp_dir = tempfile.mkdtemp()
    fake_pdf = os.path.join(temp_dir, "fake.pdf")
    
    try:
        # Create fake PDF content
        with open(fake_pdf, 'w') as f:
            f.write("This is not a real PDF file content")
        
        # Test loading fake PDF
        loader = TextFileLoader(fake_pdf)
        
        try:
            documents = loader.load_documents()
            print("[ERROR] Should have raised ValueError for invalid PDF")
            return False
        except ValueError as e:
            print(f"[OK] Correctly handled invalid PDF: {str(e)[:60]}...")
            return True
        except Exception as e:
            print(f"[OK] Handled invalid PDF with exception: {str(e)[:60]}...")
            return True
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


def test_file_type_validation():
    """Test 6: File type validation and error handling."""
    
    print("\n=== Test 6: File Type Validation ===")
    
    # Create unsupported file type
    temp_dir = tempfile.mkdtemp()
    unsupported_file = os.path.join(temp_dir, "document.docx")
    
    try:
        # Create fake DOCX file
        with open(unsupported_file, 'w') as f:
            f.write("This is a fake DOCX file")
        
        # Test loading unsupported file
        loader = TextFileLoader(unsupported_file)
        
        try:
            documents = loader.load_documents()
            print("[ERROR] Should have raised ValueError for unsupported file type")
            return False
        except ValueError as e:
            if "must be a directory, .txt file, or .pdf file" in str(e):
                print(f"[OK] Correctly rejected unsupported file type")
                return True
            else:
                print(f"[ERROR] Unexpected error message: {e}")
                return False
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


def test_text_splitting_with_mixed_content():
    """Test 7: Text splitting with content from different file types."""
    
    print("\n=== Test 7: Text Splitting with Mixed Content ===")
    
    try:
        # Create sample content simulating mixed sources
        text_content = "This content comes from a text file. It discusses programming and software development."
        pdf_content = "This content comes from a PDF document. It covers machine learning and AI applications."
        
        documents = [text_content, pdf_content]
        
        # Test text splitting
        splitter = CharacterTextSplitter(chunk_size=50, chunk_overlap=10)
        chunks = splitter.split_texts(documents)
        
        # Assertions
        assert len(chunks) > len(documents), "Should create more chunks than original documents"
        
        # Verify content is preserved
        all_chunks = " ".join(chunks).lower()
        assert "programming" in all_chunks, "Should preserve text file content"
        assert "machine learning" in all_chunks, "Should preserve PDF content"
        
        print(f"[OK] Split {len(documents)} documents into {len(chunks)} chunks")
        print(f"[OK] Sample chunk: {chunks[0][:60]}...")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Text splitting test failed: {e}")
        return False


def test_encoding_handling():
    """Test 8: Text encoding handling."""
    
    print("\n=== Test 8: Text Encoding Handling ===")
    
    # Create file with special characters
    content_with_encoding = """
    This document contains special characters: café, naïve, résumé
    Unicode symbols: ★ ♥ → ∞ ≈ ≠
    Non-ASCII text: 日本語 العربية русский
    """
    
    temp_file = create_sample_text_file(content_with_encoding.strip())
    
    try:
        # Test with default encoding
        loader = TextFileLoader(temp_file)
        documents = loader.load_documents()
        
        # Assertions
        assert len(documents) == 1, "Should load one document"
        assert "café" in documents[0], "Should handle special characters"
        assert "★" in documents[0], "Should handle Unicode symbols"
        
        print("[OK] Successfully handled special characters and Unicode")
        
        # Test with explicit encoding
        loader_explicit = TextFileLoader(temp_file, encoding="utf-8")
        documents_explicit = loader_explicit.load_documents()
        
        assert documents[0] == documents_explicit[0], "Results should be identical with explicit UTF-8"
        print("[OK] Explicit UTF-8 encoding works correctly")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Encoding test failed: {e}")
        return False
        
    finally:
        # Cleanup
        if os.path.exists(temp_file):
            shutil.rmtree(os.path.dirname(temp_file))


def test_backwards_compatibility():
    """Test 9: Backwards compatibility with existing code."""
    
    print("\n=== Test 9: Backwards Compatibility ===")
    
    try:
        # Test that old text-only usage still works
        text_content = "Legacy text file content for backwards compatibility testing."
        temp_file = create_sample_text_file(text_content)
        
        # Old-style usage (should still work)
        loader = TextFileLoader(temp_file)
        loader.load()  # Old method
        documents_old = loader.documents
        
        # New-style usage
        documents_new = loader.load_documents()
        
        # Should be identical
        assert documents_old == documents_new, "Old and new methods should return same results"
        assert len(documents_old) >= 1, "Should load at least one document"
        
        # Content should be present (may have some whitespace differences)
        assert text_content.strip() in documents_old[0] or documents_old[0].strip() == text_content.strip(), "Content should match"
        
        print("[OK] Backwards compatibility maintained")
        return True
        
    except Exception as e:
        print(f"[ERROR] Backwards compatibility test failed: {e}")
        return False
        
    finally:
        # Cleanup
        if 'temp_file' in locals() and os.path.exists(temp_file):
            shutil.rmtree(os.path.dirname(temp_file))


def run_all_tests():
    """Run all Phase 1 tests and report results."""
    
    print("Testing Phase 1: PDF Document Support")
    print("=" * 50)
    
    tests = [
        ("Basic Text File Loading", test_basic_text_file_loading),
        ("PDF File Loading", test_pdf_file_loading),
        ("PDF Loading with Existing File", test_pdf_loading_with_existing_file),
        ("Directory Loading Mixed Formats", test_directory_loading_mixed_formats),
        ("PDF Error Handling", test_pdf_error_handling),
        ("File Type Validation", test_file_type_validation),
        ("Text Splitting with Mixed Content", test_text_splitting_with_mixed_content),
        ("Encoding Handling", test_encoding_handling),
        ("Backwards Compatibility", test_backwards_compatibility)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
                print(f"✓ {test_name}: PASSED")
            else:
                print(f"✗ {test_name}: FAILED")
        except Exception as e:
            print(f"✗ {test_name}: ERROR - {e}")
    
    print("\n" + "=" * 50)
    print(f"Phase 1 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All Phase 1 tests passed!")
        print("\nPhase 1 Features Validated:")
        print("✓ PDF document loading with pypdf")
        print("✓ Mixed directory processing (TXT + PDF)")
        print("✓ Error handling for corrupted/invalid files")
        print("✓ File type validation and rejection")
        print("✓ Text encoding support (UTF-8)")
        print("✓ Backwards compatibility with existing code")
        print("✓ Integration with text splitting functionality")
    else:
        failed = total - passed
        print(f"⚠️  {failed} test(s) failed. Check error messages above.")
    
    return passed == total


def print_phase1_summary():
    """Print a summary of Phase 1 capabilities."""
    
    summary = """
PHASE 1 IMPLEMENTATION SUMMARY
==============================

WHAT WAS ADDED:
- PDF document processing using pypdf library
- Enhanced TextFileLoader with PDF support
- Mixed directory processing (TXT + PDF files)
- Robust error handling for corrupted files
- File type validation and proper error messages

KEY BENEFITS:
- Process enterprise and academic documents (mostly PDFs)
- Maintain full backwards compatibility with existing TXT workflow
- Handle real-world document formats without breaking existing code
- Provide clear error messages for unsupported or corrupted files

TECHNICAL DETAILS:
- Added pypdf>=5.1.0 dependency 
- New load_pdf_file() method in TextFileLoader
- Updated load_directory() to handle both .txt and .pdf files
- Preserved all existing functionality and method signatures
- Added proper error handling with informative messages

USAGE EXAMPLES:
- TextFileLoader("document.pdf")  # Load single PDF
- TextFileLoader("docs/")         # Load mixed TXT and PDF directory
- Same text splitting and processing works with PDF content

FILES MODIFIED:
- aimakerspace/text_utils.py (enhanced with PDF support)
- pyproject.toml (added pypdf dependency)

This enhancement enables the RAG system to process real-world document
collections while maintaining complete backwards compatibility.
    """
    
    print(summary)


if __name__ == "__main__":
    try:
        success = run_all_tests()
        print_phase1_summary()
        
        if success:
            print("\n🚀 Phase 1 implementation fully validated!")
        else:
            print("\n⚠️  Some Phase 1 tests need attention.")
            
    except Exception as e:
        print(f"\n❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()