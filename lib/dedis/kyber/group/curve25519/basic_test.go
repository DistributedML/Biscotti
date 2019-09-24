// +build experimental vartime

package curve25519

import (
	"testing"

	"github.com/dedis/kyber/util/test"
)

// Test the basic implementation of the Ed25519 curve.

func TestBasic25519(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping test in short mode.")
	} else {
		test.GroupTest(t, new(BasicCurve).Init(Param25519(), false))
	}
}

// Test ProjectiveCurve versus BasicCurve implementations

func TestCompareBasicProjective25519(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping test in short mode.")
	} else {
		test.CompareGroups(t, testSuite.XOF,
			new(BasicCurve).Init(Param25519(), false),
			new(ProjectiveCurve).Init(Param25519(), false))
	}
}

func TestCompareBasicProjectiveE382(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping test in short mode.")
	} else {
		test.CompareGroups(t, testSuite.XOF,
			new(BasicCurve).Init(ParamE382(), false),
			new(ProjectiveCurve).Init(ParamE382(), false))
	}
}

func TestCompareBasicProjective41417(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping test in short mode.")
	} else {
		test.CompareGroups(t, testSuite.XOF,
			new(BasicCurve).Init(Param41417(), false),
			new(ProjectiveCurve).Init(Param41417(), false))
	}
}

func TestCompareBasicProjectiveE521(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping test in short mode.")
	} else {
		test.CompareGroups(t, testSuite.XOF,
			new(BasicCurve).Init(ParamE521(), false),
			new(ProjectiveCurve).Init(ParamE521(), false))
	}
}

// Benchmark contrasting implementations of the Ed25519 curve

var basicBench = test.NewGroupBench(new(BasicCurve).Init(Param25519(), false))

func BenchmarkPointAddBasic(b *testing.B)     { basicBench.PointAdd(b.N) }
func BenchmarkPointMulBasic(b *testing.B)     { basicBench.PointMul(b.N) }
func BenchmarkPointBaseMulBasic(b *testing.B) { basicBench.PointBaseMul(b.N) }
func BenchmarkPointEncodeBasic(b *testing.B)  { basicBench.PointEncode(b.N) }
func BenchmarkPointDecodeBasic(b *testing.B)  { basicBench.PointDecode(b.N) }
func BenchmarkPointPickBasic(b *testing.B)    { basicBench.PointPick(b.N) }
