import sys
import os

print(f"Python interpreter: {sys.executable}")
print(f"Python version: {sys.version}")

libraries = [
    "torch",
    "transformers",
    "datasets",
    "sentence_transformers",
    "textstat",
    "tabulate",
    "scipy",
    "pandas",
    "seaborn",
    "spacy",
    "matplotlib"
]

print("\n--- Verifying library imports ---")
all_ok = True
for lib in libraries:
    try:
        __import__(lib)
        print(f"[OK]  {lib}")
    except ImportError as e:
        print(f"[ERR] {lib} - {e}")
        all_ok = False

if all_ok:
    print("\n[SUCCESS] All key libraries are successfully installed and importable!")
else:
    print("\n[WARNING] Some libraries could not be imported.")

try:
    import spacy
    print("\nChecking SpaCy English model 'en_core_web_sm'...")
    nlp = spacy.load("en_core_web_sm")
    print("[OK]  SpaCy 'en_core_web_sm' is installed and loaded successfully!")
except Exception as e:
    print(f"[ERR] SpaCy 'en_core_web_sm' could not be loaded: {e}")
