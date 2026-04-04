package spartacus

import (
	"os"
	"strconv"
	"strings"
)

func detectTotalMemory() uint64 {
	// On Linux, read /proc/meminfo
	data, err := os.ReadFile("/proc/meminfo")
	if err != nil {
		return 16 * 1024 * 1024 * 1024
	}
	for _, line := range strings.Split(string(data), "\n") {
		if strings.HasPrefix(line, "MemTotal:") {
			fields := strings.Fields(line)
			if len(fields) >= 2 {
				kb, _ := strconv.ParseUint(fields[1], 10, 64)
				return kb * 1024
			}
		}
	}
	return 16 * 1024 * 1024 * 1024
}
