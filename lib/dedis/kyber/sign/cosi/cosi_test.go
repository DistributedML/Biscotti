package cosi

import (
	"crypto/cipher"
	"crypto/sha512"
	"errors"
	"hash"
	"testing"

	"github.com/dedis/kyber"
	"github.com/dedis/kyber/group/edwards25519"
	"github.com/dedis/kyber/sign/eddsa"
	"github.com/dedis/kyber/util/key"
	"github.com/dedis/kyber/xof/blake2xb"
)

// Specify cipher suite using AES-128, SHA512, and the Edwards25519 curve.
type cosiSuite struct {
	Suite
	r kyber.XOF
}

func (m *cosiSuite) Hash() hash.Hash {
	return sha512.New()
}
func (m *cosiSuite) RandomStream() cipher.Stream { return m.r }

var testSuite = &cosiSuite{edwards25519.NewBlakeSHA256Ed25519(), blake2xb.New(nil)}

func TestCoSi(t *testing.T) {
	testCoSi(t, 2, 0)
	testCoSi(t, 5, 0)
	testCoSi(t, 5, 2)
	testCoSi(t, 5, 4)
}

func testCoSi(t *testing.T, n, f int) {
	message := []byte("Hello World Cosi")

	// Generate key pairs
	var kps []*key.Pair
	var privates []kyber.Scalar
	var publics []kyber.Point
	for i := 0; i < n; i++ {
		kp := key.NewKeyPair(testSuite)
		kps = append(kps, kp)
		privates = append(privates, kp.Private)
		publics = append(publics, kp.Public)
	}

	// Init masks
	var masks []*Mask
	var byteMasks [][]byte
	for i := 0; i < n-f; i++ {
		m, err := NewMask(testSuite, publics, publics[i])
		if err != nil {
			t.Fatal(err)
		}
		masks = append(masks, m)
		byteMasks = append(byteMasks, masks[i].Mask())
	}

	// Compute commitments
	var v []kyber.Scalar // random
	var V []kyber.Point  // commitment
	for i := 0; i < n-f; i++ {
		x, X := Commit(testSuite)
		v = append(v, x)
		V = append(V, X)
	}

	// Aggregate commitments
	aggV, aggMask, err := AggregateCommitments(testSuite, V, byteMasks)
	if err != nil {
		t.Fatal(err)
	}

	// Set aggregate mask in nodes
	for i := 0; i < n-f; i++ {
		masks[i].SetMask(aggMask)
	}

	// Compute challenge
	var c []kyber.Scalar
	for i := 0; i < n-f; i++ {
		ci, err := Challenge(testSuite, aggV, masks[i].AggregatePublic, message)
		if err != nil {
			t.Fatal(err)
		}
		c = append(c, ci)
	}

	// Compute responses
	var r []kyber.Scalar
	for i := 0; i < n-f; i++ {
		ri, _ := Response(testSuite, privates[i], v[i], c[i])
		r = append(r, ri)
	}

	// Aggregate responses
	aggr, err := AggregateResponses(testSuite, r)
	if err != nil {
		t.Fatal(err)
	}

	for i := 0; i < n-f; i++ {
		// Sign
		sig, err := Sign(testSuite, aggV, aggr, masks[i])
		if err != nil {
			t.Fatal(err)
		}
		// Set policy depending on threshold f and then Verify
		var p Policy
		if f == 0 {
			p = nil
		} else {
			p = NewThresholdPolicy(n - f)
		}
		if err := Verify(testSuite, publics, message, sig, p); err != nil {
			t.Fatal(err)
		}
		// cosi signature should follow the same format as EdDSA except it has no mask
		maskLen := len(masks[i].Mask())
		if err := eddsa.Verify(masks[i].AggregatePublic, message, sig[0:len(sig)-maskLen]); err != nil {
			t.Fatal(err)
		}
	}
}

func TestMask(t *testing.T) {
	n := 17

	// Generate key pairs
	var kps []*key.Pair
	var privates []kyber.Scalar
	var publics []kyber.Point
	for i := 0; i < n; i++ {
		kp := key.NewKeyPair(testSuite)
		kps = append(kps, kp)
		privates = append(privates, kp.Private)
		publics = append(publics, kp.Public)
	}

	// Init masks and aggregate them
	var masks []*Mask
	var aggr []byte
	for i := 0; i < n; i++ {
		m, err := NewMask(testSuite, publics, publics[i])
		if err != nil {
			t.Fatal(err)
		}
		masks = append(masks, m)

		if i == 0 {
			aggr = masks[i].Mask()
		} else {
			aggr, err = AggregateMasks(aggr, masks[i].Mask())
			if err != nil {
				t.Fatal(err)
			}
		}
	}

	// Set and check aggregate mask
	if err := masks[0].SetMask(aggr); err != nil {
		t.Fatal(err)
	}

	if masks[0].CountEnabled() != n {
		t.Fatal(errors.New("unexpected number of active indices"))
	}

	if _, err := masks[0].KeyEnabled(masks[0].AggregatePublic); err == nil {
		t.Fatal(err)
	}

	for i := 0; i < n; i++ {
		b, err := masks[0].KeyEnabled(publics[i])
		if err != nil {
			t.Fatal(err)
		}
		if !b {
			t.Fatal(errors.New("mask bit not properly set"))
		}
	}
}
