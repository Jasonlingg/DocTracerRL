"""Test-session setup.

Must run before torch/faiss are imported, so it lives in conftest.py rather
than in any individual test module.
"""

import os

# faiss-cpu and torch each bundle their own OpenMP runtime. On macOS, loading
# both into one process and then doing a torch backward pass deadlocks — the
# GRPO loss tests hang indefinitely, but only when they run AFTER a test that
# has imported faiss (which is why they pass in isolation). Pinning OpenMP to a
# single thread avoids the conflict. Verified: with faiss imported first, the
# GRPO tests hang past 45s without this and pass in ~4s with it.
# KMP_DUPLICATE_LIB_OK=TRUE, the usual workaround, does NOT help here.
os.environ.setdefault("OMP_NUM_THREADS", "1")
