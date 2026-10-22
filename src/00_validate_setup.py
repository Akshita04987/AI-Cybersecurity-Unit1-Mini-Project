"""Task 1 - Validate that the environment is set up correctly."""
import sys
import numpy, pandas, sklearn, matplotlib, seaborn

print("Python      :", sys.version.split()[0])
for lib in (numpy, pandas, sklearn, matplotlib, seaborn):
    print(f"{lib.__name__:<12}: {lib.__version__}")

# Tiny sanity check: fit a model on random data
from sklearn.linear_model import LogisticRegression
X = numpy.random.rand(50, 3)
y = (X.sum(axis=1) > 1.5).astype(int)
print("Sample model accuracy:", LogisticRegression().fit(X, y).score(X, y))
print("Setup OK")
