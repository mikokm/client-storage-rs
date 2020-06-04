#!/usr/bin/python3

import subprocess

iterations = 3
executable = "target/release/client-storage-rs"
common_args = [
    "--benchmark --apple-format --texture-storage",
    "--benchmark --apple-format --texture-storage --texture-rectangle",
    "--benchmark --apple-format --texture-storage --texture-array",
    "--benchmark --apple-format --texture-rectangle",
    "--benchmark --apple-format --texture-array",
    "--benchmark --swizzle --texture-storage",
]

upload_methods = [
    "--pbo 1",
    "--pbo 1 --pbo-reallocate-buffer",
    "--pbo 2",
    "--pbo 2 --pbo-reallocate-buffer",
    "--client-storage"
]

subprocess.run(["cargo", "build", "--release"])

for common in common_args:
    for method in upload_methods:
        cmd = [executable, *common.split(), *method.split()];
        print(cmd)
        for i in range(iterations):
            subprocess.run(cmd)
