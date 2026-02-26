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
import tempfile
import time
from pathlib import Path
from typing import Any

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
            lines.append(f"        return result")
            lines.append("")

    # Generate standalone functions
    num_functions = max(1, num_lines // 40)
    for _ in range(num_functions):
        func_name = random_identifier("function_")
        arg1 = random_identifier("arg_")
        arg2 = random_identifier("val_")
        lines.append(f"def {func_name}({arg1}: str, {arg2}: int = 0) -> Optional[str]:")
        lines.append(f'    """Generated function with unique content."""')
        lines.append(f"    # Unique identifier: {random.randint(10000, 99999)}")
        lines.append(f"    if not {arg1}:")
        lines.append(f"        return None")
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
    lines.append(f"  id: number;")
    lines.append(f"  name: string;")
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
        lines.append(f"    return result;")
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
        lines.append(f"    return result;")
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
        lines.append(f"    return null;")
        lines.append(f"  }}")
        lines.append(f"  return `${{{arg1}}}_${{{arg2}}}`;")
        lines.append("}")
        lines.append("")

    # Generate arrow functions
    arrow_func_name = random_identifier("handler_")
    lines.append(f"const {arrow_func_name} = (input) => {{")
    lines.append(f"  // Arrow function with unique content: {random.randint(1000, 9999)}")
    lines.append(f"  return {{ processed: input }};")
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


def get_memory_usage_mb() -> float:
    """Get current process RSS (Resident Set Size) in megabytes.

    Returns:
        Memory usage in MB
    """
    import psutil

    process = psutil.Process(os.getpid())
    # RSS (Resident Set Size) in bytes, convert to MB
    return process.memory_info().rss / (1024 * 1024)


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
    """
    print("\n" + "=" * 70)
    print("MEMORY LEAK TEST DIAGNOSTICS")
    print("=" * 70)
    print(f"\nFile Statistics:")
    print(f"  Total files generated:  {files_generated:6,}")
    print(f"  Total files indexed:    {files_indexed:6,}")
    print(f"  Total chunks stored:    {chunks_stored:6,}")
    print(f"\nMemory Usage (MB):")
    print(f"  Baseline (before):      {baseline_mb:8.2f} MB")
    print(f"  Peak (during):          {peak_mb:8.2f} MB")
    print(f"  Final (after):          {final_mb:8.2f} MB")
    print(f"  Delta (final - base):   {final_mb - baseline_mb:8.2f} MB")
    print(f"  Peak delta:             {peak_mb - baseline_mb:8.2f} MB")
    print(f"\nTiming:")
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

    # TODO: Implement test body
    # - Generate synthetic codebase
    # - Set up Database, Indexer, and config
    # - Capture baseline memory
    # - Start background peak memory monitoring
    # - Run full index (with timing)
    # - Stop peak monitoring and capture final memory

    # Placeholder values for diagnostic output (will be replaced with real values)
    files_generated = 0
    files_indexed = 0
    chunks_stored = 0
    baseline_mb = 0.0
    peak_mb = 0.0
    final_mb = 0.0

    # Track wall-clock duration of indexing
    start_time = time.perf_counter()
    # TODO: Call indexer.start_full_index() here
    end_time = time.perf_counter()
    duration_sec = end_time - start_time

    # Print diagnostics (always printed regardless of pass/fail)
    print_diagnostics(
        files_generated=files_generated,
        files_indexed=files_indexed,
        chunks_stored=chunks_stored,
        baseline_mb=baseline_mb,
        peak_mb=peak_mb,
        final_mb=final_mb,
        duration_sec=duration_sec,
    )

    # TODO: Assert memory constraints
    # - Assert peak_mb < 2000.0 (2 GB)
    # - Assert final_mb < 2000.0 (2 GB)

    assert False, "Test not yet implemented"
