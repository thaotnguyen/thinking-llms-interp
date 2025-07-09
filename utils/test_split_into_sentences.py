"""
Tests for the split_into_sentences function.

Run this file directly to execute all tests:
    python test_split_into_sentences.py
"""

from utils import split_into_sentences


def test_basic_sentence_splitting():
    """Test basic sentence splitting with periods."""
    test_text = "This is the first sentence. This is the second sentence. This is the third sentence."
    result = split_into_sentences(test_text)
    expected = ["This is the first sentence", "This is the second sentence", "This is the third sentence"]
    assert result == expected, f"Test failed: expected {expected}, got {result}"
    print("✓ Basic sentence splitting test passed")


def test_different_punctuation_marks():
    """Test splitting with different punctuation marks."""
    test_text = "What is this? This is amazing! I think so; maybe not."
    result = split_into_sentences(test_text, min_words=2)
    expected = ["What is this", "This is amazing", "I think so", "maybe not"]
    assert result == expected, f"Test failed: expected {expected}, got {result}"
    print("✓ Different punctuation marks test passed")


def test_short_sentence_filtering():
    """Test filtering out sentences with less than 3 words."""
    test_text = "This is a good sentence. Yes. No way. This is another good sentence."
    result = split_into_sentences(test_text)
    expected = ["This is a good sentence", "This is another good sentence"]
    assert result == expected, f"Test failed: expected {expected}, got {result}"
    print("✓ Short sentence filtering test passed")


def test_whitespace_handling():
    """Test proper trimming of leading/trailing spaces."""
    test_text = "  First sentence here.   Second sentence there.  "
    result = split_into_sentences(test_text)
    expected = ["First sentence here", "Second sentence there"]
    assert result == expected, f"Test failed: expected {expected}, got {result}"
    print("✓ Whitespace handling test passed")


def test_empty_string():
    """Test edge case for empty input."""
    test_text = ""
    result = split_into_sentences(test_text)
    expected = []
    assert result == expected, f"Test failed: expected {expected}, got {result}"
    print("✓ Empty string test passed")


def test_single_sentence():
    """Test when no splitting is needed."""
    test_text = "This is a single sentence with more than three words"
    result = split_into_sentences(test_text)
    expected = ["This is a single sentence with more than three words"]
    assert result == expected, f"Test failed: expected {expected}, got {result}"
    print("✓ Single sentence test passed")


def test_consecutive_punctuation():
    """Test handling of multiple punctuation marks."""
    test_text = "What?! This is crazy!! Are you sure? Yes I am."
    result = split_into_sentences(test_text, min_words=1)
    expected = ["What", "This is crazy", "Are you sure", "Yes I am"]
    assert result == expected, f"Test failed: expected {expected}, got {result}"
    print("✓ Consecutive punctuation test passed")


def test_mixed_content():
    """Test combination of short and long sentences."""
    test_text = "I think this is good. Ok. But what about this longer sentence? Sure thing! No."
    result = split_into_sentences(test_text, min_words=2)
    expected = ["I think this is good", "But what about this longer sentence", "Sure thing"]
    assert result == expected, f"Test failed: expected {expected}, got {result}"
    print("✓ Mixed content test passed")


def test_only_whitespace_sentences():
    """Test sentences that become empty after stripping."""
    test_text = "Good sentence here.   .   Another good sentence."
    result = split_into_sentences(test_text)
    expected = ["Good sentence here", "Another good sentence"]
    assert result == expected, f"Test failed: expected {expected}, got {result}"
    print("✓ Whitespace-only sentences test passed")


def test_no_punctuation():
    """Test text without sentence-ending punctuation."""
    test_text = "This is a sentence without punctuation"
    result = split_into_sentences(test_text)
    expected = ["This is a sentence without punctuation"]
    assert result == expected, f"Test failed: expected {expected}, got {result}"
    print("✓ No punctuation test passed")


def test_decimal_numbers():
    """Test that decimal numbers don't cause sentence splits."""
    test_text = "Since the P-value is 0.075, which is less than 0.05, that suggests that the t-statistic is pretty large in magnitude, right?"
    result = split_into_sentences(test_text)
    expected = ["Since the P-value is 0.075, which is less than 0.05, that suggests that the t-statistic is pretty large in magnitude, right"]
    assert result == expected, f"Test failed: expected {expected}, got {result}"
    print("✓ Decimal numbers test passed")


def test_mixed_decimals_and_sentences():
    """Test sentences with decimal numbers and actual sentence breaks."""
    test_text = "The value is 3.14159 which is pi. The other value is 2.718 for Euler's number! What about 0.5 as a fraction?"
    result = split_into_sentences(test_text)
    expected = ["The value is 3.14159 which is pi", "The other value is 2.718 for Euler's number", "What about 0.5 as a fraction"]
    assert result == expected, f"Test failed: expected {expected}, got {result}"
    print("✓ Mixed decimals and sentences test passed")


def run_all_tests():
    """Run all test functions."""
    print("Running tests for split_into_sentences function...\n")
    
    test_functions = [
        test_basic_sentence_splitting,
        test_different_punctuation_marks,
        test_short_sentence_filtering,
        test_whitespace_handling,
        test_empty_string,
        test_single_sentence,
        test_consecutive_punctuation,
        test_mixed_content,
        test_only_whitespace_sentences,
        test_no_punctuation,
        test_decimal_numbers,
        test_mixed_decimals_and_sentences,
    ]
    
    passed = 0
    total = len(test_functions)
    
    for test_func in test_functions:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test_func.__name__} failed: {e}")
        except Exception as e:
            print(f"✗ {test_func.__name__} error: {e}")
    
    print(f"\nTest Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed successfully!")
    else:
        print("❌ Some tests failed. Please check the implementation.")


if __name__ == "__main__":
    run_all_tests() 