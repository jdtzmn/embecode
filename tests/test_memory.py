"""Memory leak detection test for full indexing workflow.

This test measures memory consumption during a full index of ~2000 synthetic
source files to catch regressions in memory usage. The test uses real Database
and Indexer implementations with a mock Embedder to avoid model loading overhead.

The test must fail on the current codebase (which has a known memory leak) and
pass only after the leak is fixed.
"""

from __future__ import annotations

import os
import random
import string
import threading
import time
from pathlib import Path

import pytest

from embecode.config import load_config
from embecode.db import Database
from embecode.indexer import Indexer

# ============================================================================
# Mock Embedder
# ============================================================================


class MockEmbedder:
    """Mock embedder that returns fixed-dimension zero vectors.

    This avoids loading a real sentence-transformers model while still
    exercising the embedding storage and database paths.
    """

    def __init__(self, dimension: int = 768) -> None:
        """Initialize mock embedder with specified dimension."""
        self._dimension = dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate mock embeddings (zero vectors)."""
        return [[0.0] * self._dimension for _ in texts]

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimension


# ============================================================================
# Synthetic Codebase Generator
# ============================================================================


def generate_python_file_content(num_lines: int) -> str:
    """Generate valid Python source code with varied content.

    Args:
        num_lines: Approximate number of lines to generate (150-300)

    Returns:
        Python source code as string
    """
    lines = []

    # Add varied imports to create diversity
    import_choices = [
        "from typing import Any, Dict, List, Optional",
        "import json",
        "import os",
        "from pathlib import Path",
        "from datetime import datetime",
        "import logging",
    ]
    lines.extend(random.sample(import_choices, k=random.randint(2, 4)))
    lines.append("")

    # Generate random identifiers with varying content
    def random_identifier(prefix: str = "") -> str:
        suffix = "".join(random.choices(string.ascii_lowercase, k=random.randint(5, 12)))
        return f"{prefix}{suffix}" if prefix else suffix

    # Generate varied class definitions
    num_classes = max(1, num_lines // 50)
    for _ in range(num_classes):
        class_name = random_identifier("Class").capitalize()
        lines.append(f"class {class_name}:")
        lines.append(f'    """Generated class {class_name}."""')
        lines.append("")

        # Add __init__ method
        lines.append("    def __init__(self, *args, **kwargs) -> None:")
        lines.append(f'        """Initialize {class_name}."""')
        lines.append(
            f"        self.{random_identifier('attr_')} = kwargs.get('{random_identifier()}', None)"
        )
        lines.append(f"        self.{random_identifier('data_')} = {{}}")
        lines.append("")

        # Add 2-4 methods per class
        for _ in range(random.randint(2, 4)):
            method_name = random_identifier("method_")
            param_name = random_identifier("param_")
            lines.append(f"    def {method_name}(self, {param_name}: Any) -> Dict[str, Any]:")
            lines.append(f'        """Process {param_name}."""')
            lines.append(
                f"        # Generated method with unique content: {random.randint(1000, 9999)}"
            )
            lines.append(f"        result = {{'status': 'ok', 'data': {param_name}}}")
            lines.append("        return result")
            lines.append("")

    # Generate standalone functions
    num_functions = max(1, num_lines // 40)
    for _ in range(num_functions):
        func_name = random_identifier("function_")
        arg1 = random_identifier("arg_")
        arg2 = random_identifier("val_")
        lines.append(f"def {func_name}({arg1}: str, {arg2}: int = 0) -> Optional[str]:")
        lines.append('    """Generated function with unique content."""')
        lines.append(f"    # Unique identifier: {random.randint(10000, 99999)}")
        lines.append(f"    if not {arg1}:")
        lines.append("        return None")
        lines.append(f"    return f'{{{arg1}}}_{{{arg2}}}'")
        lines.append("")

    # Pad to approximate target line count with comments
    while len(lines) < num_lines:
        lines.append(f"# Generated comment line with unique content: {random.randint(1, 99999)}")

    return "\n".join(lines)


def generate_typescript_file_content(num_lines: int) -> str:
    """Generate valid TypeScript source code with varied content.

    Args:
        num_lines: Approximate number of lines to generate (100-200)

    Returns:
        TypeScript source code as string
    """
    lines = []

    # Add varied imports
    import_choices = [
        "import { Component } from 'react';",
        "import type { ReactNode } from 'react';",
        "import axios from 'axios';",
        "import { useState, useEffect } from 'react';",
    ]
    lines.extend(random.sample(import_choices, k=random.randint(1, 3)))
    lines.append("")

    # Generate random identifiers
    def random_identifier(prefix: str = "") -> str:
        suffix = "".join(random.choices(string.ascii_lowercase, k=random.randint(5, 12)))
        return f"{prefix}{suffix}" if prefix else suffix

    # Generate interfaces
    num_interfaces = random.randint(1, 3)
    for _ in range(num_interfaces):
        interface_name = random_identifier("Interface").capitalize()
        lines.append(f"interface {interface_name} {{")
        for _ in range(random.randint(2, 5)):
            prop_name = random_identifier("prop_")
            prop_type = random.choice(["string", "number", "boolean", "any"])
            lines.append(f"  {prop_name}: {prop_type};")
        lines.append("}")
        lines.append("")

    # Generate type aliases
    type_name = random_identifier("Type").capitalize()
    lines.append(f"type {type_name} = {{")
    lines.append("  id: number;")
    lines.append("  name: string;")
    lines.append(f"  // Unique identifier: {random.randint(1000, 9999)}")
    lines.append("};")
    lines.append("")

    # Generate class with methods
    class_name = random_identifier("Service").capitalize()
    lines.append(f"class {class_name} {{")
    lines.append(f"  private {random_identifier('data_')}: Map<string, any>;")
    lines.append("")
    lines.append("  constructor() {")
    lines.append(f"    this.{random_identifier('data_')} = new Map();")
    lines.append(f"    // Unique init: {random.randint(10000, 99999)}")
    lines.append("  }")
    lines.append("")

    # Add methods
    for _ in range(random.randint(2, 4)):
        method_name = random_identifier("method_")
        param_name = random_identifier("param_")
        lines.append(f"  public {method_name}({param_name}: string): any {{")
        lines.append(f"    // Process {param_name} - unique: {random.randint(1000, 9999)}")
        lines.append(f"    const result = {{ status: 'ok', data: {param_name} }};")
        lines.append("    return result;")
        lines.append("  }")
        lines.append("")

    lines.append("}")
    lines.append("")

    # Generate standalone functions
    num_functions = max(1, num_lines // 50)
    for _ in range(num_functions):
        func_name = random_identifier("function_")
        arg_name = random_identifier("arg_")
        lines.append(f"function {func_name}({arg_name}: string): boolean {{")
        lines.append(f"  // Unique function identifier: {random.randint(10000, 99999)}")
        lines.append(f"  return {arg_name}.length > 0;")
        lines.append("}")
        lines.append("")

    # Pad to approximate target line count
    while len(lines) < num_lines:
        lines.append(f"// Generated comment with unique content: {random.randint(1, 99999)}")

    return "\n".join(lines)


def generate_javascript_file_content(num_lines: int) -> str:
    """Generate valid JavaScript source code with varied content.

    Args:
        num_lines: Approximate number of lines to generate (80-150)

    Returns:
        JavaScript source code as string
    """
    lines = []

    # Add varied imports/requires
    import_choices = [
        "const express = require('express');",
        "const axios = require('axios');",
        "const fs = require('fs');",
        "const path = require('path');",
    ]
    lines.extend(random.sample(import_choices, k=random.randint(1, 3)))
    lines.append("")

    # Generate random identifiers
    def random_identifier(prefix: str = "") -> str:
        suffix = "".join(random.choices(string.ascii_lowercase, k=random.randint(5, 12)))
        return f"{prefix}{suffix}" if prefix else suffix

    # Generate class using ES6 syntax
    class_name = random_identifier("Handler").capitalize()
    lines.append(f"class {class_name} {{")
    lines.append("  constructor(config) {")
    lines.append(f"    this.{random_identifier('config_')} = config || {{}};")
    lines.append(f"    this.{random_identifier('state_')} = new Map();")
    lines.append(f"    // Unique init: {random.randint(10000, 99999)}")
    lines.append("  }")
    lines.append("")

    # Add methods
    for _ in range(random.randint(2, 4)):
        method_name = random_identifier("handle_")
        param_name = random_identifier("data_")
        lines.append(f"  {method_name}({param_name}) {{")
        lines.append(f"    // Process {param_name} - unique: {random.randint(1000, 9999)}")
        lines.append(f"    const result = {{ status: 'ok', data: {param_name} }};")
        lines.append("    return result;")
        lines.append("  }")
        lines.append("")

    lines.append("}")
    lines.append("")

    # Generate standalone functions
    num_functions = max(1, num_lines // 40)
    for _ in range(num_functions):
        func_name = random_identifier("function_")
        arg1 = random_identifier("arg_")
        arg2 = random_identifier("val_")
        lines.append(f"function {func_name}({arg1}, {arg2} = 0) {{")
        lines.append(f"  // Unique function identifier: {random.randint(10000, 99999)}")
        lines.append(f"  if (!{arg1}) {{")
        lines.append("    return null;")
        lines.append("  }")
        lines.append(f"  return `${{{arg1}}}_${{{arg2}}}`;")
        lines.append("}")
        lines.append("")

    # Generate arrow functions
    arrow_func_name = random_identifier("handler_")
    lines.append(f"const {arrow_func_name} = (input) => {{")
    lines.append(f"  // Arrow function with unique content: {random.randint(1000, 9999)}")
    lines.append("  return { processed: input };")
    lines.append("};")
    lines.append("")

    # Export statement
    lines.append(f"module.exports = {{ {class_name}, {arrow_func_name} }};")
    lines.append("")

    # Pad to approximate target line count
    while len(lines) < num_lines:
        lines.append(f"// Generated comment with unique content: {random.randint(1, 99999)}")

    return "\n".join(lines)


def generate_synthetic_codebase(base_path: Path) -> None:
    """Generate ~2000 synthetic source files in a nested directory structure.

    File distribution:
    - 1200 Python files (~150-300 lines each)
    - 400 TypeScript files (~100-200 lines each)
    - 400 JavaScript files (~80-150 lines each)

    Directory structure: minimum 4 levels of nesting (e.g., src/core/auth/models/)

    Args:
        base_path: Root directory for generated files
    """
    # Define directory structures with at least 4 levels of nesting
    directory_templates = [
        "src/core/auth/models",
        "src/core/auth/services",
        "src/core/auth/controllers",
        "src/core/data/repositories",
        "src/core/data/migrations",
        "src/core/api/endpoints",
        "src/core/api/middleware",
        "src/utils/validation/rules",
        "src/utils/validation/schemas",
        "src/utils/formatting/helpers",
        "src/components/ui/buttons",
        "src/components/ui/forms",
        "src/components/ui/modals",
        "src/components/layout/headers",
        "src/components/layout/footers",
        "src/services/external/api/clients",
        "src/services/external/storage/providers",
        "src/services/internal/cache/handlers",
        "src/services/internal/queue/workers",
        "tests/unit/core/auth",
        "tests/unit/core/data",
        "tests/integration/api/endpoints",
        "tests/integration/services/external",
    ]

    # Create all directory structures
    for dir_template in directory_templates:
        (base_path / dir_template).mkdir(parents=True, exist_ok=True)

    # Generate Python files (1200 total)
    python_files_per_dir = 1200 // len(directory_templates)
    for dir_template in directory_templates:
        dir_path = base_path / dir_template
        for i in range(python_files_per_dir):
            file_name = f"module_{i:04d}.py"
            file_path = dir_path / file_name
            num_lines = random.randint(150, 300)
            content = generate_python_file_content(num_lines)
            file_path.write_text(content, encoding="utf-8")

    # Generate remaining Python files to reach exactly 1200
    remaining_python = 1200 - (python_files_per_dir * len(directory_templates))
    for i in range(remaining_python):
        dir_template = random.choice(directory_templates)
        dir_path = base_path / dir_template
        file_name = f"extra_module_{i:04d}.py"
        file_path = dir_path / file_name
        num_lines = random.randint(150, 300)
        content = generate_python_file_content(num_lines)
        file_path.write_text(content, encoding="utf-8")

    # Generate TypeScript files (400 total)
    ts_files_per_dir = 400 // len(directory_templates)
    for dir_template in directory_templates:
        dir_path = base_path / dir_template
        for i in range(ts_files_per_dir):
            file_name = f"component_{i:04d}.ts"
            file_path = dir_path / file_name
            num_lines = random.randint(100, 200)
            content = generate_typescript_file_content(num_lines)
            file_path.write_text(content, encoding="utf-8")

    # Generate remaining TypeScript files to reach exactly 400
    remaining_ts = 400 - (ts_files_per_dir * len(directory_templates))
    for i in range(remaining_ts):
        dir_template = random.choice(directory_templates)
        dir_path = base_path / dir_template
        file_name = f"extra_component_{i:04d}.ts"
        file_path = dir_path / file_name
        num_lines = random.randint(100, 200)
        content = generate_typescript_file_content(num_lines)
        file_path.write_text(content, encoding="utf-8")

    # Generate JavaScript files (400 total)
    js_files_per_dir = 400 // len(directory_templates)
    for dir_template in directory_templates:
        dir_path = base_path / dir_template
        for i in range(js_files_per_dir):
            file_name = f"handler_{i:04d}.js"
            file_path = dir_path / file_name
            num_lines = random.randint(80, 150)
            content = generate_javascript_file_content(num_lines)
            file_path.write_text(content, encoding="utf-8")

    # Generate remaining JavaScript files to reach exactly 400
    remaining_js = 400 - (js_files_per_dir * len(directory_templates))
    for i in range(remaining_js):
        dir_template = random.choice(directory_templates)
        dir_path = base_path / dir_template
        file_name = f"extra_handler_{i:04d}.js"
        file_path = dir_path / file_name
        num_lines = random.randint(80, 150)
        content = generate_javascript_file_content(num_lines)
        file_path.write_text(content, encoding="utf-8")


# ============================================================================
# Memory Measurement Infrastructure
# ============================================================================


class MemorySample:
    """A single memory measurement with associated metadata.

    Attributes:
        timestamp: When the sample was taken (seconds since epoch)
        rss_mb: Resident Set Size in megabytes
        files_indexed: Approximate number of files indexed at sample time (if known)
    """

    def __init__(self, timestamp: float, rss_mb: float, files_indexed: int | None = None):
        """Initialize memory sample.

        Args:
            timestamp: Time when sample was taken (seconds since epoch)
            rss_mb: Memory usage in MB
            files_indexed: Number of files indexed at this point (if determinable)
        """
        self.timestamp = timestamp
        self.rss_mb = rss_mb
        self.files_indexed = files_indexed


def get_memory_usage_mb() -> float:
    """Get current process RSS (Resident Set Size) in megabytes.

    Returns:
        Memory usage in MB
    """
    import psutil

    process = psutil.Process(os.getpid())
    # RSS (Resident Set Size) in bytes, convert to MB
    return process.memory_info().rss / (1024 * 1024)


def find_peak_memory_sample(samples: list[MemorySample]) -> tuple[float, int | None]:
    """Find the peak memory sample and return its RSS and file index.

    Args:
        samples: List of memory samples collected during monitoring

    Returns:
        Tuple of (peak_rss_mb, files_indexed_at_peak)
        files_indexed_at_peak will be None if not determinable
    """
    if not samples:
        return 0.0, None

    peak_sample = max(samples, key=lambda s: s.rss_mb)
    return peak_sample.rss_mb, peak_sample.files_indexed


# ============================================================================
# Diagnostic Output
# ============================================================================


def print_diagnostics(
    files_generated: int,
    files_indexed: int,
    chunks_stored: int,
    baseline_mb: float,
    peak_mb: float,
    final_mb: float,
    duration_sec: float,
    peak_at_file_index: int | None = None,
) -> None:
    """Print comprehensive diagnostic information from the memory test.

    Args:
        files_generated: Total number of files generated in synthetic codebase
        files_indexed: Total number of files successfully indexed
        chunks_stored: Total number of chunks stored in database
        baseline_mb: Memory usage before indexing started (MB)
        peak_mb: Peak memory usage during indexing (MB)
        final_mb: Memory usage after indexing completed (MB)
        duration_sec: Wall-clock duration of indexing in seconds
        peak_at_file_index: Approximate file index when peak memory was observed (if determinable)
    """
    print("\n" + "=" * 70)
    print("MEMORY LEAK TEST DIAGNOSTICS")
    print("=" * 70)
    print("\nFile Statistics:")
    print(f"  Total files generated:  {files_generated:6,}")
    print(f"  Total files indexed:    {files_indexed:6,}")
    print(f"  Total chunks stored:    {chunks_stored:6,}")
    print("\nMemory Usage (MB):")
    print(f"  Baseline (before):      {baseline_mb:8.2f} MB")
    if peak_at_file_index is not None:
        print(
            f"  Peak (during):          {peak_mb:8.2f} MB (at approx. file #{peak_at_file_index})"
        )
    else:
        print(f"  Peak (during):          {peak_mb:8.2f} MB")
    print(f"  Final (after):          {final_mb:8.2f} MB")
    print(f"  Delta (final - base):   {final_mb - baseline_mb:8.2f} MB")
    print(f"  Peak delta:             {peak_mb - baseline_mb:8.2f} MB")
    print("\nTiming:")
    print(f"  Indexing duration:      {duration_sec:8.2f} seconds")
    if duration_sec > 0:
        print(f"  Files per second:       {files_indexed / duration_sec:8.2f}")
    print("\n" + "=" * 70 + "\n")


# ============================================================================
# Main Memory Leak Test
# ============================================================================


@pytest.mark.slow
@pytest.mark.memory
def test_memory_leak_during_full_index(tmp_path: Path) -> None:
    """Test that full indexing of ~2000 files stays under 2 GB memory.

    This test:
    1. Generates ~2000 synthetic source files
    2. Measures baseline memory before indexing
    3. Tracks peak memory during indexing (polled every 2s)
    4. Measures final memory after indexing completes
    5. Asserts both peak and final memory stay below 2.0 GB
    6. Prints diagnostic information regardless of pass/fail

    The test currently fails on the unfixed codebase due to unbounded
    memory growth during indexing.
    """
    # Skip test if psutil is not available
    pytest.importorskip("psutil")

    # Step 1: Generate synthetic codebase
    codebase_path = tmp_path / "codebase"
    codebase_path.mkdir()
    generate_synthetic_codebase(codebase_path)
    files_generated = 2000  # As specified: 1200 Python + 400 TypeScript + 400 JavaScript

    # Step 2: Set up Database, Indexer, and config
    db_path = tmp_path / "test.duckdb"
    db = Database(db_path)
    db.connect()

    # Load config with include=[] and exclude defaults
    config = load_config(project_path=codebase_path)

    # Create mock embedder (returns zero vectors without loading a model)
    embedder = MockEmbedder(dimension=768)

    # Create indexer
    indexer = Indexer(project_path=codebase_path, config=config, db=db, embedder=embedder)

    # Step 3: Capture baseline memory (before indexing starts)
    baseline_mb = get_memory_usage_mb()

    # Step 4: Start background peak memory monitoring
    peak_samples: list[MemorySample] = []
    monitoring_active = threading.Event()
    monitoring_active.set()

    def monitor_peak_memory() -> None:
        """Background thread to monitor peak memory every 2 seconds."""
        while monitoring_active.is_set():
            current_mb = get_memory_usage_mb()
            timestamp = time.time()
            # Try to get current file index from indexer status
            try:
                status = indexer.get_status()
                # We don't have exact file index, but we can use files_indexed as proxy
                file_index = status.files_indexed if status.files_indexed > 0 else None
            except Exception:
                file_index = None
            peak_samples.append(MemorySample(timestamp, current_mb, file_index))
            time.sleep(2.0)

    monitor_thread = threading.Thread(target=monitor_peak_memory, daemon=True)
    monitor_thread.start()

    # Step 5: Run full index with timing (foreground mode for deterministic completion)
    start_time = time.perf_counter()
    indexer.start_full_index(background=False)
    end_time = time.perf_counter()
    duration_sec = end_time - start_time

    # Step 6: Stop peak monitoring and capture final memory
    monitoring_active.clear()
    monitor_thread.join(timeout=5.0)  # Wait for monitoring thread to finish
    final_mb = get_memory_usage_mb()

    # Step 7: Extract peak memory and file index
    peak_mb, peak_at_file_index = find_peak_memory_sample(peak_samples)
    # Peak might also be the final value if it occurred after last poll
    if final_mb > peak_mb:
        peak_mb = final_mb
        # Peak was at the end, use final file count
        status = indexer.get_status()
        peak_at_file_index = status.files_indexed

    # Get final statistics from database and indexer
    status = indexer.get_status()
    files_indexed = status.files_indexed
    chunks_stored = status.total_chunks

    # Step 8: Print diagnostics (always printed regardless of pass/fail)
    print_diagnostics(
        files_generated=files_generated,
        files_indexed=files_indexed,
        chunks_stored=chunks_stored,
        baseline_mb=baseline_mb,
        peak_mb=peak_mb,
        final_mb=final_mb,
        duration_sec=duration_sec,
        peak_at_file_index=peak_at_file_index,
    )

    # Step 9: Clean up database connection
    db.close()

    # Step 10: Assert memory constraints
    # Both peak and final memory must stay below 2.0 GB (2000 MB)
    assert peak_mb < 2000.0, f"Peak memory {peak_mb:.2f} MB exceeded 2000 MB limit"
    assert final_mb < 2000.0, f"Final memory {final_mb:.2f} MB exceeded 2000 MB limit"
