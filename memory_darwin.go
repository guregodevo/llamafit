package spartacus

import (
	"os/exec"
	"strconv"
	"strings"
)

func detectDarwinMemory() uint64 {
	out, err := exec.Command("sysctl", "-n", "hw.memsize").Output()
	if err != nil {
		return 16 * 1024 * 1024 * 1024
	}
	v, err := strconv.ParseUint(strings.TrimSpace(string(out)), 10, 64)
	if err != nil {
		return 16 * 1024 * 1024 * 1024
	}
	return v
}
