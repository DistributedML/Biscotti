// +build vartime

package shuffle

import (
	"testing"

	"github.com/dedis/kyber/group/nist"
)

func BenchmarkBiffleP256(b *testing.B) {
	biffleTest(nist.NewBlakeSHA256P256(), b.N)
}

func Benchmark2PairShuffleP256(b *testing.B) {
	shuffleTest(nist.NewBlakeSHA256P256(), 2, b.N)
}

func Benchmark10PairShuffleP256(b *testing.B) {
	shuffleTest(nist.NewBlakeSHA256P256(), 10, b.N)
}
