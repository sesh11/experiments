"""Pluggable agent runtimes: interchangeable harnesses behind one interface.

Each runtime (fusion's own loop, Stirrup, pi) solves a task inside a Workspace
and records its token usage into a shared Ledger, so the orchestrator can mix
runtimes and models while keeping cost accounting comparable.
"""
