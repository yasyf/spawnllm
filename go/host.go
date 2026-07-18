package spawnllm

import (
	"os"
	"runtime"
)

func platform() string { return runtime.GOOS }

func home() string { return os.Getenv("HOME") }

func configDirEnv() string { return os.Getenv("CLAUDE_CONFIG_DIR") }
