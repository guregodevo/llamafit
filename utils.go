package llamafit

import "strings"

// StringInSlice checks if a string is in a slice.
func StringInSlice(s string, list []string) bool {
	for _, v := range list {
		if v == s {
			return true
		}
	}
	return false
}

// ReverseString reverses a string.
func ReverseString(s string) string {
	r := []rune(s)
	for i, j := 0, len(r)-1; i < j; i, j = i+1, j-1 {
		r[i], r[j] = r[j], r[i]
	}
	return string(r)
}

// IsEmpty checks if a string is empty.
func IsEmpty(s string) bool { return len(strings.TrimSpace(s)) == 0 }
