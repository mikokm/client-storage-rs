#!/usr/bin/python3

import subprocess

iterations = 3
executable = ["target/release/client-storage-rs", "--benchmark"]

texture_formats = [
    "--apple-format",
    "--apple-format --texture-storage",
    "--swizzle",
    "--swizzle --texture-storage"
]

texture_types = [
    "--texture-rectangle",
    "--texture-array"
]

upload_methods = [
    "--pbo 1",
    "--pbo 1 --pbo-reallocate-buffer",
    "--pbo 1 --pbo-no-copy",
    "--pbo 2",
    "--pbo 2 --pbo-reallocate-buffer",
    "--pbo 2 --pbo-no-copy",
    "--client-storage"
]

subprocess.run(["cargo", "build", "--release"])

for texture_format in texture_formats:
    for texture_type in texture_types:
        for upload_method in upload_methods:
            cmd = executable + texture_format.split() + texture_type.split() + upload_method.split()
            print(cmd)
            for i in range(iterations):
                subprocess.run(cmd)
