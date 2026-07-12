"""The orchestration layer: decides which runtime + model solves each task.

This package is the research artifact. Runtimes (fusion, Stirrup, pi) are
interchangeable workers; the orchestrator owns the patterns (single agent now;
Scout delegation and confidence-gated routing land here next) and the
cost-vs-capability measurement.
"""
